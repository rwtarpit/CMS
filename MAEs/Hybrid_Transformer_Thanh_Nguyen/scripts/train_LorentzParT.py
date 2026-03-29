import os
import time
import modal

trace_volume = modal.Volume.from_name("profiler-traces", create_if_missing=True)
data_volume  = modal.Volume.from_name("jetclass-data",   create_if_missing=True)
ckpt_volume  = modal.Volume.from_name("jetclass-ckpts",  create_if_missing=True)

TRACE_DIR = "/traces"
DATA_DIR  = "/datasets"
CKPT_DIR  = "/ckpts"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .run_commands("echo 'bust cache v1'")
    .add_local_dir(".", remote_path="/app",
                   ignore=["data", "logs", "assets", "jobs", "notebooks",
                            "venv", ".git", "tests", "__pycache__",
                            "**/__pycache__", "*.pyc", "*.pt", "*.pth",
                            "*.root", "*.npy", "*.tar", "*.png", "*.ipynb"])
)

app = modal.App("jetclass-trainer", image=image)


def _train_worker(
    rank: int,
    world_size: int,
    seed: int,
    config_path: str,
    checkpoint_path: str,
    train_data_dir: str,
    val_data_dir: str,
    num_epochs: int,
    use_compile: bool,
    run_name: str,
):
    import sys
    sys.path.insert(0, "/app")

    import yaml
    import json
    import warnings
    import torch
    import torch._dynamo

    from src.configs import LorentzParTConfig, TrainConfig
    from src.engine import JetClassTrainer, MaskedModelTrainer
    from src.models import LorentzParT
    from src.utils import accuracy_metric_ce, set_seed, setup_ddp, cleanup_ddp
    from src.utils.data import LazyJetClassDataset

    import lgatr.utils.einsum as lgatr_einsum
    import lgatr.primitives.linear as lgatr_linear

    # fix torch._VF.einsum graph break
    def patched_custom_einsum(equation, *operands, path=None):
        return torch.einsum(equation, *operands)

    lgatr_einsum.custom_einsum = patched_custom_einsum
    lgatr_linear.custom_einsum = patched_custom_einsum

    # fix posix.lstat graph breeak
    _basis_cpu = lgatr_linear._compute_pin_equi_linear_basis(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    def _make_patched_basis(basis_cpu):
        def _patched_basis(device=torch.device("cpu"), dtype=torch.float32):
            return basis_cpu.to(device=device, dtype=dtype)
        return _patched_basis

    lgatr_linear._compute_pin_equi_linear_basis = _make_patched_basis(_basis_cpu)

    warnings.filterwarnings("ignore")
    set_seed(seed)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = LorentzParTConfig.from_dict(config["model"])
    model_config.weights = None

    train_config = TrainConfig.from_dict(config["train"])
    train_config.num_epochs  = num_epochs
    train_config.save_ckpt   = True
    train_config.save_best   = True
    train_config.save_fig    = False
    train_config.logging_dir = CKPT_DIR
    train_config.pin_memory  = True
    train_config.num_workers = 4

    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    _basis_gpu = _basis_cpu.to(device=device)

    def _patched_basis_gpu(device=torch.device("cpu"), dtype=torch.float32):
        return _basis_gpu.to(dtype=dtype)

    lgatr_linear._compute_pin_equi_linear_basis = _patched_basis_gpu

    normalize = [True, False, False, True]
    norm_dict = {
        "pT":     (92.72917175292969,      105.83937072753906),
        "eta":    (0.0005733045982196927,   0.9174848794937134),
        "phi":    (-0.00041169871110469103, 1.8136887550354004),
        "energy": (133.8745574951172,       167.528564453125),
    }

    obj_list = [norm_dict]
    torch.distributed.broadcast_object_list(obj_list, src=0)
    norm_dict = obj_list[0]

    mask_mode = "biased" if model_config.mask else None
    train_dataset = LazyJetClassDataset(train_data_dir, normalize, norm_dict, mask_mode=mask_mode)
    val_dataset   = LazyJetClassDataset(val_data_dir,   normalize, norm_dict, mask_mode=mask_mode)

    model = LorentzParT(config=model_config).to(device)

    # torch.compile
    if use_compile:
        print(f"[rank {rank}] Applying torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    if model_config.mask:
        trainer = MaskedModelTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            config=train_config,
        )
    else:
        trainer = JetClassTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            metric=accuracy_metric_ce,
            config=train_config,
        )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[rank {rank}] Resuming from checkpoint: {checkpoint_path}")
        try:
            trainer.load_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"[rank {rank}] Error loading checkpoint: {e}")

    start = time.time()
    history, _ = trainer.train()
    elapsed = time.time() - start

    if rank == 0:
        results = {
            "run_name":    run_name,
            "use_compile": use_compile,
            "num_epochs":  num_epochs,
            "elapsed_sec": elapsed,
            "history":     history,
        }
        out_path = os.path.join(CKPT_DIR, "results", f"{run_name}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[rank 0] Training complete in {elapsed:.1f}s")
        print(f"[rank 0] Results saved to {out_path}")

    cleanup_ddp()

# modal setup
@app.function(
    gpu="A100:2",
    scaledown_window=60,
    volumes={
        TRACE_DIR: trace_volume,
        DATA_DIR:  data_volume,
        CKPT_DIR:  ckpt_volume,
    },
    timeout=3600,
)
def run_training(
    seed:            int  = 42,
    config_path:     str  = "/app/configs/pretrain_LorentzParT.yaml",
    checkpoint_path: str  = None,
    train_data_dir:  str  = "/datasets/100k/train",
    val_data_dir:    str  = "/datasets/100k/val",
    num_epochs:      int  = 1,
    use_compile:     bool = False,
):
    import torch
    import torch.multiprocessing as mp

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    world_size = torch.cuda.device_count()
    print(f"Launching on {world_size} GPUs")

    run_name = "graph_break_benchmark_tc_ro"
    print(f"Run: {run_name}")

    worker_args = (
        world_size, seed, config_path, checkpoint_path,
        train_data_dir, val_data_dir, num_epochs,
        use_compile, run_name,
    )

    if world_size > 1:
        mp.spawn(_train_worker, args=worker_args, nprocs=world_size)
    else:
        _train_worker(0, *worker_args)

    ckpt_volume.commit()
    print(f"Done. Results in ckpts volume at /results/{run_name}.json")


@app.local_entrypoint()
def main(
    seed:            int  = 42,
    config_path:     str  = "/app/configs/pretrain_LorentzParT.yaml",
    checkpoint_path: str  = None,
    train_data_dir:  str  = "/datasets/100k/train",
    val_data_dir:    str  = "/datasets/100k/val",
    num_epochs:      int  = 5,
    use_compile:     bool = False
):
    run_training.remote(
        seed=seed,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        num_epochs=num_epochs,
        use_compile=use_compile,
    )