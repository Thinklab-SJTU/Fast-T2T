"""The handler for training and evaluation."""

import os
from argparse import ArgumentParser

import torch
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from diffusion.pl_tsp_model import TSPModel
from diffusion.pl_mis_model import MISModel

torch.cuda.amp.autocast(enabled=True)
torch.cuda.empty_cache()

import warnings
warnings.filterwarnings("ignore")


def arg_parser():
    parser = ArgumentParser(
        description="Train a Pytorch-Lightning diffusion model on a TSP dataset."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--training_split", type=str, default=None)
    parser.add_argument(
        "--training_split_label_dir",
        type=str,
        default=None,
        help="Directory containing labels for training split (used for MIS).",
    )
    parser.add_argument("--validation_split", type=str, default=None)
    parser.add_argument(
        "--validation_split_label_dir",
        type=str,
        default=None,
        help="Directory containing labels for validation split (used for MIS).",
    )
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument(
        "--test_split_label_dir",
        type=str,
        default=None,
        help="Directory containing labels for test split (used for MIS).",
    )
    parser.add_argument("--validation_examples", type=int, default=64)
    parser.add_argument(
        "--graph_type",
        type=str,
        default="undirected",
        choices=["undirected", "directed"],
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")

    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_activation_checkpoint", action="store_true")

    parser.add_argument("--diffusion_schedule", type=str, default="linear")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_schedule", type=str, default="cosine")
    parser.add_argument("--inference_trick", type=str, default="ddim")
    parser.add_argument("--sequential_sampling", type=int, default=1)
    parser.add_argument("--parallel_sampling", type=int, default=1)

    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sparse_factor", type=int, default=-1)
    parser.add_argument("--aggregation", type=str, default="sum")
    parser.add_argument("--two_opt_iterations", type=int, default=0)
    parser.add_argument("--save_numpy_heatmap", action="store_true")

    parser.add_argument("--project_name", type=str, default="tsp_diffusion")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_logger_name", type=str, default=None)
    parser.add_argument(
        "--resume_id", type=str, default=None, help="Resume training on wandb."
    )
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume_weight_only", action="store_true")

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--rewrite_ratio", type=float, default=0.25)
    parser.add_argument("--norm", action="store_true", default=False)
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--rewrite_steps", type=int, default=1)
    parser.add_argument("--steps_inf", type=int, default=1)
    parser.add_argument("--guided", action="store_true")

    parser.add_argument(
        "--consistency", action="store_true", help="used for consistency training"
    )
    parser.add_argument("--boundary_func", default="truncate")
    parser.add_argument("--alpha", type=float)

    parser.add_argument("--c1", type=float, default=50, help="coefficient of F1")
    parser.add_argument("--c2", type=float, default=50, help="coefficient of F2")

    parser.add_argument(
        "--offline", action="store_true", help="set true when use offline wandb"
    )

    args = parser.parse_args()
    return args


def main(args):
    print(args)
    epochs = args.num_epochs
    project_name = args.project_name

    if args.offline or ("WANDB_MODE" in os.environ and os.environ["WANDB_MODE"] == "offline"):
        os.environ["WANDB_MODE"] = "offline"
        wandb.init()
    else:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    if args.task == "tsp":
        model_class = TSPModel
        saving_mode = "min"
    elif args.task == "mis":
        model_class = MISModel
        saving_mode = "max"
    else:
        raise NotImplementedError

    model = model_class(param_args=args)
    os.makedirs(os.path.join(args.storage_path), exist_ok=True)

    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=project_name,
        entity=args.wandb_entity,
        save_dir=os.path.join(args.storage_path),
        id=args.resume_id or wandb_id,
    )
    rank_zero_info(
        f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}"
    )
    rank_zero_info(args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/solved_cost",
        mode=saving_mode,
        save_top_k=1,
        save_last=True,
        dirpath=os.path.join(
            wandb_logger.save_dir,
            args.wandb_logger_name,
            wandb_logger._id,
            "checkpoints",
        ),
    )
    lr_callback = LearningRateMonitor(logging_interval="step")
    trainer = Trainer(
        accelerator="auto",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        max_epochs=epochs,
        callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback, lr_callback],
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        strategy=DDPStrategy(static_graph=True),
        precision=16 if args.fp16 else 32,
        inference_mode=False,
    )

    ckpt_path = args.ckpt_path
    if args.do_train:
        if args.do_test:
            trainer.test(model)
        if args.resume_weight_only:
            model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)
    elif args.do_test:
        trainer.test(model, ckpt_path=ckpt_path)
    trainer.logger.finalize("success")


if __name__ == "__main__":
    args = arg_parser()
    main(args)
