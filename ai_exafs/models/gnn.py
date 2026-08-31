"""
Copyright (c) 2026 - ICIQ (Institute of Chemical Research of Catalonia)

Developed at ICIQ and funded by NCCR Catalysis.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric.nn as tgnn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.aggr.utils import PoolingByMultiheadAttention
import lightning.pytorch as pl


act_fun_dict = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "leakyrelu": nn.LeakyReLU(),
    "softplus": nn.Softplus(),
    "tanh": nn.Tanh(),
}


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, units: int, n_layers: int, act_fun, bias=False):
        super().__init__()

        self.hidden = nn.Sequential()

        for i in range(n_layers):
            # Adding Linear layer
            self.hidden.add_module(
                name=f"Linear_{str(i)}",
                module=nn.Linear(units, units, bias=bias),
            )
            # Adding Activation function
            self.hidden.add_module(name=f"Act_{str(i)}", module=act_fun)

    def forward(self, x):
        out = self.hidden(x)
        out = x + out
        return out


# Graph Neural Network
class GNN(nn.Module):
    def __init__(
        self,
        in_features: int = 92,
        out_features: int = 160,
        spectra_dim: int = 928,
        residual_block: bool = False,
        n_linear: int = 0,
        n_conv: int = 3,
        act_fun: str = "relu",
        edge_dim: int = 4,
        g_aggr: str = "add",
        g_batch_norm: bool = False,
        g_bias: bool = False,
        max_num_nodes: int = 1000,
        pool_heads: int = 1,
        pool_layer_norm: bool = True,
        pool_dropout: float = 0.0,
        bias: bool = False,
    ):
        super(GNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spectra_dim = spectra_dim
        self.residual_block = residual_block
        self.n_linear = n_linear
        self.n_conv = n_conv
        self.act_fun = act_fun_dict[act_fun]
        self.edge_dim = edge_dim
        self.g_aggr = g_aggr
        self.g_batch_norm = g_batch_norm
        self.g_bias = g_bias
        self.max_num_nodes = max_num_nodes
        self.pool_heads = pool_heads
        self.pool_layer_norm = pool_layer_norm
        self.pool_dropout = pool_dropout
        self.bias = bias

        # Input Layer
        self.input_layer = nn.Linear(
            self.in_features, self.out_features, bias=self.bias
        )

        # Linear Residual Block
        if self.residual_block:
            self.linear_block = ResidualBlock(
                self.out_features,
                n_layers=self.n_linear,
                act_fun=self.act_fun,
                bias=self.bias,
            )
        else:
            self.linear_block = nn.ModuleList(
                [
                    nn.Linear(
                        self.out_features, self.out_features, bias=self.bias
                    )
                    for _ in range(self.n_linear)
                ]
            )

        # GraphConv Block
        self.graph_conv_block = nn.ModuleList(
            [
                tgnn.CGConv(
                    channels=self.out_features,
                    dim=edge_dim,
                    aggr=self.g_aggr,
                    batch_norm=self.g_batch_norm,
                    bias=self.g_bias,
                )
                for _ in range(self.n_conv)
            ]
        )

        # Linear before pooling
        self.lin_a = nn.Linear(
            self.out_features, self.out_features, bias=self.bias
        )

        # Pooling layer with Attention
        self.pma = PoolingByMultiheadAttention(
            channels=self.out_features,
            heads=self.pool_heads,
            layer_norm=self.pool_layer_norm,
            dropout=self.pool_dropout,
        )

        # Linear after pooling
        self.lin_b = nn.Linear(
            self.out_features, self.spectra_dim, bias=self.bias
        )

    def forward(self, data):
        # Input Layer
        out = self.act_fun(self.input_layer(data.x))

        # Linear Block
        if self.residual_block:
            # Linear Residual Block
            out = self.linear_block(out)
        else:
            # Non-residual block
            for n_layer in range(self.n_linear):
                out = self.act_fun(self.linear_block[n_layer](out))

        # GraphConv Block
        edge_attr = data.edge_attr if self.edge_dim else None
        for g_layer in range(self.n_conv):
            out = self.act_fun(
                self.graph_conv_block[g_layer](
                    x=out,
                    edge_index=data.edge_index,
                    edge_attr=edge_attr,
                )
            )

        # Linear before pooling
        out = self.lin_a(out)

        # To Dense batch
        batch_x, mask = to_dense_batch(
            x=out,
            batch=data.batch,
            fill_value=0.0,
            max_num_nodes=self.max_num_nodes,
            batch_size=None,
        )

        # Pooling
        out = self.pma(x=batch_x, mask=mask)

        # Linear after pooling
        out = self.lin_b(out.squeeze(1))
        return torch.sigmoid(out)


# Wrap all into a PL model
class ExafsGNN(pl.LightningModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        spectra_dim: int,
        residual_block: bool,
        n_linear: int,
        n_conv: int,
        act_fun: str,
        edge_dim: int,
        g_aggr: str,
        g_batch_norm: bool,
        g_bias: bool,
        max_num_nodes: int,
        pool_heads: int,
        pool_layer_norm: bool,
        pool_dropout: float,
        bias: bool,
        lr: float,
        batch_size: int,
        weight_decay: float,
    ):
        super().__init__()
        # To save hyperparameters inside model
        self.save_hyperparameters()

        # Graph Neural Network
        self.gnn = GNN(
            in_features=in_features,
            out_features=out_features,
            spectra_dim=spectra_dim,
            residual_block=residual_block,
            n_linear=n_linear,
            n_conv=n_conv,
            act_fun=act_fun,
            edge_dim=edge_dim,
            g_aggr=g_aggr,
            g_batch_norm=g_batch_norm,
            g_bias=g_bias,
            max_num_nodes=max_num_nodes,
            pool_heads=pool_heads,
            pool_layer_norm=pool_layer_norm,
            pool_dropout=pool_dropout,
            bias=bias,
        )

    def forward(self, data):
        out = self.gnn(data)
        return out

    def _sid(self, s1, s2):
        """Spectral Information Divergence"""
        p = (s1 / torch.sum(s1)) + torch.finfo(torch.float32).eps
        q = (s2 / torch.sum(s2)) + torch.finfo(torch.float32).eps
        return torch.sum(p * torch.log(p / q) + q * torch.log(q / p))

    def _sis(self, s1, s2):
        """Spectral Information Similarity"""
        return 1.0 / (1 + self._sid(s1, s2))

    def _rfactor(self, y_hat, y):
        """EXAFS R-Factor metric"""
        p = torch.sum(torch.square(y - y_hat))
        q = torch.sum(torch.square(y))
        return torch.sqrt(torch.divide(p, q))

    def training_step(self, data, batch_idx):
        # GNN as Encoder
        y_hat = self.forward(data)
        # Target as full EXAFS
        y = data.target_y.view(data.batch_size, self.hparams.spectra_dim)
        # Loss
        mae_loss = F.l1_loss(y_hat, y)
        mse_loss = F.mse_loss(y_hat, y)
        sid_loss = self._sid(y_hat, y)
        rfactor_loss = self._rfactor(y_hat, y)
        # Logging
        self.log(
            "train_sid_loss",
            sid_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_rfactor_loss",
            rfactor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_mae_loss",
            mae_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_mse_loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return rfactor_loss

    def validation_step(self, data, batch_idx):
        # GNN
        y_hat = self.forward(data)
        # Target as full EXAFS
        y = data.target_y.view(data.batch_size, self.hparams.spectra_dim)
        # Loss
        mae_loss = F.l1_loss(y_hat, y)
        mse_loss = F.mse_loss(y_hat, y)
        sid_loss = self._sid(y_hat, y)
        rfactor_loss = self._rfactor(y_hat, y)
        # Logging
        self.log(
            "val_sid_loss",
            sid_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val_rfactor_loss",
            rfactor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_mae_loss",
            mae_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_mse_loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return rfactor_loss

    def test_step(self, data, batch_idx):
        # GNN as Encoder
        y_hat = self.forward(data)
        # Target as full EXAFS
        y = data.target_y.view(data.batch_size, self.hparams.spectra_dim)
        # MAE
        mae_loss = F.l1_loss(y_hat, y)
        mse_loss = F.mse_loss(y_hat, y)
        sid_loss = self._sid(y_hat, y)
        sis_loss = self._sis(y_hat, y)
        rfactor_loss = self._rfactor(y_hat, y)
        # Logging
        self.log(
            "test_sid_loss",
            sid_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_rfactor_loss",
            rfactor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_sis_loss",
            sis_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_mae_loss",
            mae_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_mse_loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return rfactor_loss

    def configure_optimizers(self):
        # Adam optimizer
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # Learning rate scheduler
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=5,
                min_lr=1e-6,
            ),
            "monitor": "val_rfactor_loss",
        }
        return [optimizer], [lr_scheduler]
