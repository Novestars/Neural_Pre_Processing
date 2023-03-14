import argparse
import random
import shutil
import sys
import os
import torch
from torch.utils.data import DataLoader
from models.model import NPP
from dataset.mri_dataset_affine import Generate_dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar
import glob

def save_checkpoint(state, is_best, filename="checkpoint_dae.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'_best_loss')

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, required=False, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=60,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint",default="checkpoint_dae_{quality}.pth.tar".format(quality = 24))

    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    train_dataset,val_dataset = Generate_dataset()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=( "cuda"),persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        num_workers=2,
        shuffle=False,
        pin_memory=( "cuda"),persistent_workers=True
    )

    root_dir_path = os.path.join('checkpoint', "np_128_tv_hyper6_mean")

    trainer = pl.Trainer(
        default_root_dir=root_dir_path,
        devices=1,
        max_epochs=args.epochs,
        precision=16, accelerator="gpu",
        callbacks=[
            ModelCheckpoint(mode="max",every_n_epochs = 1),
            LearningRateMonitor("epoch"),RichProgressBar(),
        ],
        benchmark=True,
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    files = glob.glob('checkpoint/np_128_tv_hyper5_mean/lightning_logs/*/checkpoints/*')

    sorted_by_mtime_descending = sorted(files, key=lambda t: -os.stat(t).st_mtime)
    if len(sorted_by_mtime_descending)>0:
        pretrained_filename = sorted_by_mtime_descending[0]
    else:
        pretrained_filename = ''

    resume_path = pretrained_filename
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        state_dict = torch.load(pretrained_filename)
        model = NPP(args.learning_rate)
        model.load_state_dict(state_dict['state_dict'],strict=False)
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = NPP(args.learning_rate)
        trainer.fit(model, train_dataloader, val_dataloader,ckpt_path= resume_path)
    return model


if __name__ == "__main__":
    #[1,1e-1,1e-2,1e-3]
    main(sys.argv[1:])
