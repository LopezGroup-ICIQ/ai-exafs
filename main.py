"""
Copyright (c) 2026 - ICIQ (Institute of Chemical Research of Catalonia)

Developed at ICIQ and funded by NCCR Catalysis.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os

# Pytorch
import torch

# Lightning
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.data.lightning import LightningDataset

# Datasets
from ai_exafs.datasets.lmdb_datasets import (
    LmdbGraphDataset,
    get_dataset_splits,
    fit_scalers_on_train,
)

# Model and Callbacks
from ai_exafs.models.gnn import ExafsGNN
from ai_exafs.callbacks.callbacks import ExafsFTOut

# Config
import hydra
from omegaconf import DictConfig

# Warnings
import warnings

warnings.filterwarnings("ignore")

# Ensure that all operations are deterministic on GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("medium")

# Device
device = (
    torch.device("cuda:0")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# Trainer
@hydra.main(config_path="./configs", config_name="config")
def train_model(cfg: DictConfig) -> Trainer:
    # Seed
    pl.seed_everything(cfg.experiment.seed, workers=True)
    # Run directory (checkpoints + normalization state)
    ckpt_dir = os.path.join(
        f"{cfg.paths.root_dir}{cfg.paths.checkpoint_path}", cfg.experiment.id
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    # Dataset
    dataset = LmdbGraphDataset(
        root=f"{cfg.paths.root_dir}/data",
        db_path=f"{cfg.paths.root_dir}{cfg.paths.db_path}",
    )
    # Split dataset into train/val/test before any normalization
    train_data, val_data, test_data = get_dataset_splits(
        dataset,
        val_ratio=cfg.dataset.val_ratio,
        test_ratio=cfg.dataset.test_ratio,
        random_state=cfg.dataset.random_state,
    )
    # Min-Max scalers fitted on the training split only
    scalers = fit_scalers_on_train(
        dataset,
        train_data,
        lo=cfg.dataset.norm_lo,
        hi=cfg.dataset.norm_hi,
    )
    scalers.save(os.path.join(ckpt_dir, "element_scalers.json"))
    # Diagnostic, how much of each split falls outside the training range
    if cfg.dataset.norm_coverage_report:
        for name, split in (
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ):
            print(scalers.coverage(split, name=name))
    # Datamodule
    graph_module = LightningDataset(
        train_data,
        val_data,
        test_data,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    )
    # Logger
    logger = WandbLogger(
        name=cfg.experiment.id,
        save_dir=cfg.paths.root_dir,
        project=cfg.logging.wandb_project,
        offline=cfg.logging.offline,
    )
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=ckpt_dir,
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=cfg.training.max_epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpt_dir,
                save_weights_only=True,
                mode="min",
                monitor="val_rfactor_loss",
                save_top_k=1,
            ),
            LearningRateMonitor(logging_interval="epoch"),
            RichProgressBar(refresh_rate=1),
            EarlyStopping(
                monitor="val_rfactor_loss",
                patience=cfg.training.early_stopping_patience,
                mode="min",
            ),
            ExafsFTOut(
                filename=f"./test_{cfg.experiment.id}",
                scalers=scalers,
                fft=cfg.callbacks.fft,
                kmin=cfg.callbacks.kmin,
                kmax=cfg.callbacks.kmax,
                kweight=cfg.callbacks.kweight,
                dk=cfg.callbacks.dk,
                window=cfg.callbacks.window,
                rmax_out=cfg.callbacks.rmax_out,
            ),
        ],
        logger=logger,
    )
    # Graph Neural Network
    model = ExafsGNN(
        in_features=cfg.model.in_features,
        out_features=cfg.model.out_features,
        spectra_dim=cfg.model.spectra_dim,
        residual_block=cfg.model.residual_block,
        n_linear=cfg.model.n_linear,
        n_conv=cfg.model.n_conv,
        act_fun=cfg.model.act_fun,
        edge_dim=cfg.model.edge_dim,
        g_aggr=cfg.model.g_aggr,
        g_batch_norm=cfg.model.g_batch_norm,
        g_bias=cfg.model.g_bias,
        max_num_nodes=cfg.model.max_num_nodes,
        pool_heads=cfg.model.pool_heads,
        pool_layer_norm=cfg.model.pool_layer_norm,
        pool_dropout=cfg.model.pool_dropout,
        bias=cfg.model.bias,
        lr=cfg.training.lr,
        batch_size=cfg.dataset.batch_size,
        weight_decay=cfg.training.weight_decay,
    )
    trainer.fit(model, datamodule=graph_module)
    model = ExafsGNN.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    # Test
    trainer.test(model, datamodule=graph_module)
    logger.experiment.finish()
    return trainer


# Run!
if __name__ == "__main__":
    trainer = train_model()
