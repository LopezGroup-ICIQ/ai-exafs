"""
Copyright (c) 2026 - ICIQ (Institute of Chemical Research of Catalonia)

Developed at ICIQ and funded by NCCR Catalysis.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os

# Pytorch
import torch

# Pytorch Lightning
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from torch_geometric.data.lightning import LightningDataset

# Datasets
from ai_exafs.datasets.lmdb_datasets import (
    LmdbGraphDataset,
    get_dataset_splits,
)
from ai_exafs.models.gnn import ExafsGNN

# Optuna
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Warnings
import warnings

warnings.filterwarnings("ignore")

# Setting the seed
pl.seed_everything(42)

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


# GLOBAL variables
ROOT_DIR = "/home/USER/YOUR_PATH/ai_exafs"
DB_PATH = "/data/wf_tmQM_wB97MV.lmdb"
PRE_TRAINED_PATH = "/pre_trained"
MAX_EPOCHS = 50


def _calc_head(out_features, heads):
    """Helper function to correct num heads for attention layer"""
    if out_features % heads == 0:
        return heads
    else:
        heads_list = []
        for i in range(1, int(heads)):
            if out_features % i == 0:
                heads_list.append(i)
        return max(heads_list)


def objective(trial: optuna.trial.Trial) -> float:
    # GNN
    out_features = trial.suggest_int("out_features", 92, 256)
    residual_block = trial.suggest_categorical("residual_block", [True, False])
    n_linear = trial.suggest_int("n_linear", 0, 10)
    n_conv = trial.suggest_int("n_conv", 1, 10)
    act_fun = trial.suggest_categorical(
        "act_fun", ["relu", "elu", "leakyrelu", "softplus", "tanh"]
    )
    g_aggr = trial.suggest_categorical("g_aggr", ["add", "mean", "max"])
    g_batch_norm = trial.suggest_categorical("g_batch_norm", [True, False])
    g_bias = trial.suggest_categorical("g_bias", [True, False])
    pool_heads = trial.suggest_int("pool_heads", 1, 20)
    pool_layer_norm = trial.suggest_categorical(
        "pool_layer_norm", [True, False]
    )
    pool_dropout = trial.suggest_float("pool_dropout", 0.0, 0.5)
    bias = trial.suggest_categorical("bias", [True, False])
    # Training
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-8, 1e-3)

    # Correct pool_heads
    pool_heads_new = _calc_head(out_features=out_features, heads=pool_heads)
    if pool_heads_new != pool_heads:
        print(f"Re-scaling Pool Heads to: {pool_heads_new}")

    # Dataset
    dataset = LmdbGraphDataset(
        root=ROOT_DIR + "/data",
        db_path=ROOT_DIR + DB_PATH,
    )

    # Splitting between train/val/test
    train_data, val_data, test_data = get_dataset_splits(
        dataset,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    # Generate lightning data module
    graph_module = LightningDataset(
        train_data,
        val_data,
        test_data,
        batch_size=batch_size,
        num_workers=0,
    )

    # Model
    model = ExafsGNN(
        # GNN
        in_features=92,
        out_features=out_features,
        target_dim=928,
        residual_block=residual_block,
        n_linear=n_linear,
        n_conv=n_conv,
        act_fun=act_fun,
        edge_dim=1,
        g_aggr=g_aggr,
        g_batch_norm=g_batch_norm,
        g_bias=g_bias,
        max_num_nodes=2000,
        pool_heads=pool_heads_new,
        pool_layer_norm=pool_layer_norm,
        pool_dropout=pool_dropout,
        bias=bias,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )

    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            EarlyStopping(
                monitor="val_avg_loss",
                patience=15,
                mode="min",
            ),
            PyTorchLightningPruningCallback(
                trial=trial,
                monitor="val_avg_loss",
            ),
        ],
    )

    # Hparams
    hparams = dict(
        in_features=92,
        out_features=out_features,
        target_dim=928,
        residual_block=residual_block,
        n_linear=n_linear,
        n_conv=n_conv,
        act_fun=act_fun,
        edge_dim=1,
        g_aggr=g_aggr,
        g_batch_norm=g_batch_norm,
        g_bias=g_bias,
        max_num_nodes=2000,
        pool_heads=pool_heads_new,
        pool_layer_norm=pool_layer_norm,
        pool_dropout=pool_dropout,
        bias=bias,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
    )
    print("Hparams: ", hparams)

    # Fit
    trainer.logger.log_hyperparams(hparams)
    trainer.fit(model, datamodule=graph_module)

    return trainer.logged_metrics["val_rfactor_loss"].item()


# Run!
if __name__ == "__main__":

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=4,
        n_warmup_steps=15,
        interval_steps=5,
    )

    study = optuna.create_study(
        study_name="ai_exafs_hpo_experiment_1",
        storage="sqlite:///ai_exafs_hpo.db",
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=300)

    print("Best Trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")

    print("Params: ")
    for k, v in trial.params.items():
        print(f"{k}: {v}")
