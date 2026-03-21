import os
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
    .add_local_dir(".", remote_path="/app",
                   ignore=["data", "logs", "assets", "jobs", "notebooks",
                           "venv", ".git", "tests", "__pycache__",
                           "**/__pycache__", "*.pyc", "*.pt", "*.pth",
                           "*.root", "*.npy", "*.tar", "*.png", "*.ipynb"])
)

app = modal.App("jetclass-profiler", image=image)


def _profile_worker(
    rank: int,
    world_size: int,
    seed: int,
    config_path: str,
    checkpoint_path: str,
    train_data_dir: str,
    val_data_dir: str,
    tb_log_dir: str,
    wait: int,
    warmup: int,
    active: int,
    steps: int,
    debug_compile: bool,
):
    import sys
    sys.path.insert(0, "/app")

    import yaml
    import warnings
    import torch

    from src.configs import LorentzParTConfig, TrainConfig
    from src.engine import JetClassTrainer, MaskedModelTrainer
    from src.models import LorentzParT
    from src.utils import accuracy_metric_ce, set_seed, setup_ddp, cleanup_ddp
    from src.utils.data import LazyJetClassDataset

    warnings.filterwarnings("ignore")
    set_seed(seed)

    if debug_compile and rank == 0:
        inductor_debug_dir = os.path.join(TRACE_DIR, "inductor_debug")
        os.makedirs(inductor_debug_dir, exist_ok=True)

        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(TRACE_DIR, "inductor_cache")

        import torch._inductor.config as inductor_cfg
        inductor_cfg.debug = True
        inductor_cfg.trace.enabled = True
        inductor_cfg.trace.debug_dir = inductor_debug_dir

        os.environ["TORCH_LOGS"] = "graph_breaks,recompiles,graph"
        os.environ["TORCHDYNAMO_VERBOSE"] = "1"

        log_path = os.path.join(TRACE_DIR, f"dynamo_rank{rank}.log")
        _log_file = open(log_path, "w")
        sys.stdout = _log_file
        sys.stderr = _log_file
        print(f"[rank {rank}] compile debug : {inductor_debug_dir}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = LorentzParTConfig.from_dict(config["model"])
    model_config.weights = None  

    train_config = TrainConfig.from_dict(config["train"])
    train_config.num_epochs    = 9999   
    train_config.steps         = steps  
    train_config.save_ckpt     = False
    train_config.save_best     = False
    train_config.save_fig      = False
    train_config.logging_dir   = CKPT_DIR
    train_config.logging_steps = 1      
    train_config.callbacks     = []   

    print(f"[rank {rank}] steps={steps}, num_epochs={train_config.num_epochs}")

    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    normalize = [True, False, False, True]
    norm_dict = {
        "pT":     (92.72917175292969,      105.83937072753906),
        "eta":    (0.0005733045982196927,   0.9174848794937134),
        "phi":    (-0.00041169871110469103, 1.8136887550354004),
        "energy": (133.8745574951172,       167.528564453125),
    }
    if world_size > 1:
        obj_list = [norm_dict]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        norm_dict = obj_list[0]

    mask_mode = "biased" if model_config.mask else None
    train_dataset = LazyJetClassDataset(
        train_data_dir, normalize, norm_dict, mask_mode=mask_mode
    )
    val_dataset = LazyJetClassDataset(
        val_data_dir, normalize, norm_dict, mask_mode=mask_mode
    )

    model = LorentzParT(config=model_config).to(device)
    compile_mode = "default" if debug_compile else "reduce-overhead"
    model = torch.compile(model, mode=compile_mode)

    meta_config = dict(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        config=train_config,
    )
    if model_config.mask:
        trainer = MaskedModelTrainer(**meta_config)
    else:
        trainer = JetClassTrainer(**meta_config, metric=accuracy_metric_ce)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[rank {rank}] Resuming from checkpoint: {checkpoint_path}")
        try:
            trainer.load_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"[rank {rank}] Error loading checkpoint: {e}")

    if rank == 0:
        trainer.profile(
            tb_log_dir=tb_log_dir,
            wait=wait,
            warmup=warmup,
            active=active,
        )
    else:
        trainer.train()

    cleanup_ddp()

    if debug_compile and rank == 0:
        sys.stdout.flush()
        sys.stderr.flush()


@app.function(
    gpu="A100:2",
    scaledown_window=60,
    volumes={
        "/traces":   trace_volume,
        "/datasets": data_volume,
        "/ckpts":    ckpt_volume,
    },
    timeout=900,
)
def run_profiler(
    seed:            int  = 42,
    config_path:     str  = "/app/configs/train_LorentzParT.yaml",
    checkpoint_path: str  = None,
    train_data_dir:  str  = "/datasets/val_5M",
    val_data_dir:    str  = "/datasets/val_5M",
    wait:            int  = 2,
    warmup:          int  = 3,
    active:          int  = 5,
    steps:           int  = 15,
):
    import torch
    import torch.multiprocessing as mp

    required = wait + warmup + active
    if steps < required:
        raise ValueError(
            f"steps={steps} must be >= wait+warmup+active={required}. "
            f"Set steps >= {required}."
        )

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    world_size = torch.cuda.device_count()
    print(f"Launching on {world_size} GPUs")

    run_name   = f"jetclass_a100x{world_size}_w{wait}_wm{warmup}_a{active}"
    tb_log_dir = os.path.join(TRACE_DIR, run_name)
    os.makedirs(tb_log_dir, exist_ok=True)

    print(f"Profiling run : {run_name}")
    print(f"  schedule : wait={wait}, warmup={warmup}, active={active}")
    print(f"  steps : {steps}")
    print(f"  traces -> : {tb_log_dir}")

    worker_args = (
        world_size,
        seed,
        config_path,
        checkpoint_path,
        train_data_dir,
        val_data_dir,
        tb_log_dir,
        wait,
        warmup,
        active,
        steps,
    )

    if world_size > 1:
        mp.spawn(_profile_worker, args=worker_args, nprocs=world_size)
    else:
        _profile_worker(0, *worker_args)

    import time
    time.sleep(10) 
    trace_volume.commit()

    print(f"\nDone. Traces written to: {tb_log_dir}")


@app.local_entrypoint()
def main(
    seed:            int  = 42,
    config_path:     str  = "/app/configs/train_LorentzParT.yaml",
    checkpoint_path: str  = None,
    train_data_dir:  str  = "/datasets/val_5M",
    val_data_dir:    str  = "/datasets/val_5M",
    wait:            int  = 2,
    warmup:          int  = 3,
    active:          int  = 5,
    steps:           int  = 15,
):
    run_profiler.remote(
        seed=seed,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        wait=wait,
        warmup=warmup,
        active=active,
        steps=steps,
    )