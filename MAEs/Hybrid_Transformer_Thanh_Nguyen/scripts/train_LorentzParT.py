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

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = LorentzParTConfig.from_dict(config["model"])
    model_config.weights = None

    train_config = TrainConfig.from_dict(config["train"])
    train_config.num_epochs  = 9999
    train_config.save_ckpt   = False
    train_config.save_best   = False
    train_config.save_fig    = False
    train_config.logging_dir = CKPT_DIR
    
    print(f"[rank {rank}] steps={steps}, num_epochs={train_config.num_epochs}")

    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    normalize = [True, False, False, True]
    norm_dict = {
        "pT":     (92.72917175292969,     105.83937072753906),
        "eta":    (0.0005733045982196927,  0.9174848794937134),
        "phi":    (-0.00041169871110469103, 1.8136887550354004),
        "energy": (133.8745574951172,      167.528564453125),
    }

    obj_list = [norm_dict]
    torch.distributed.broadcast_object_list(obj_list, src=0)
    norm_dict = obj_list[0]

    mask_mode = "biased" if model_config.mask else None
    train_dataset = LazyJetClassDataset(train_data_dir, normalize, norm_dict, mask_mode=mask_mode)
    val_dataset   = LazyJetClassDataset(val_data_dir,   normalize, norm_dict, mask_mode=mask_mode)

    model = LorentzParT(config=model_config).to(device)
    model = torch.compile(model)

    if model_config.mask:
        trainer = MaskedModelTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            config=train_config,
            steps=steps
        )
    else:
        trainer = JetClassTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            metric=accuracy_metric_ce,
            config=train_config,
            steps=steps
        )

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
        trainer._train_loop(profile_run=False) 

    cleanup_ddp()


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
    seed:            int = 42,
    config_path:     str = "/app/configs/train_LorentzParT.yaml",
    checkpoint_path: str = None,
    train_data_dir: str = "/datasets/val_5M",
    val_data_dir:   str = "/datasets/val_5M",
    wait:   int = 2,
    warmup: int = 3,
    active: int = 5,
    steps:  int = 15,
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

    run_name   = f"jetclass_a100_w{wait}_wm{warmup}_a{active}"
    tb_log_dir = os.path.join(TRACE_DIR, run_name)
    os.makedirs(tb_log_dir, exist_ok=True)

    print(f"Profiling run : {run_name}")
    print(f"  schedule    : wait={wait}, warmup={warmup}, active={active}")
    print(f"  steps       : {steps}")
    print(f"  traces →    : {tb_log_dir}")


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
    print(f"\nDone. Traces written to volume at: {tb_log_dir}")
    print("Serve TensorBoard:  modal serve modal_profile.py::tensorboard_app")


@app.function(
    volumes={TRACE_DIR: trace_volume},
    scaledown_window=300,
)
@modal.wsgi_app()
def tensorboard_app():
    import time
    from tensorboard.backend import application
    from tensorboard import default

    class _VolumeRefreshMiddleware:
        def __init__(self, wsgi_app):
            self._app = wsgi_app

        def __call__(self, environ, start_response):
            trace_volume.reload()
            return self._app(environ, start_response)

    wsgi_app = application.TensorBoardWSGIApp(
        flags=None,
        plugins=default.get_plugins(),
        data_provider=None,
        assets_zip_provider=None,
    )
    time.sleep(1)
    return _VolumeRefreshMiddleware(wsgi_app)


@app.local_entrypoint()
def main(
    seed:            int = 42,
    config_path:     str = "/app/configs/train_LorentzParT.yaml",
    checkpoint_path: str = None,
    train_data_dir:  str = "/datasets/val_5M",  
    val_data_dir:    str = "/datasets/val_5M",
    wait:   int = 2,
    warmup: int = 3,
    active: int = 5,
    steps:  int = 15,
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