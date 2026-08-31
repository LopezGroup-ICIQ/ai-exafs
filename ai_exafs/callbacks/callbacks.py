"""
Copyright (c) 2026 - ICIQ (Institute of Chemical Research of Catalonia)

Developed at ICIQ and funded by NCCR Catalysis.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
from typing import Optional

import numpy as np
import scipy.interpolate as interp

import torch
import lightning.pytorch as pl

from larch import Group
from ai_exafs.preprocessing.xafsft import xas_fft
from ai_exafs.preprocessing.normalization import ElementScalers


class ExafsFTOut(pl.Callback):
    """
    Callback that runs at the end of testing to easily
    evaluate the model performance by exporting a json
    file with the following structure:

    {"id": {"target":, "prediction":,}}

    This callback assumes that the model predicts the
    K-space that needs to be transformed to R.

    Args:
        filename        : Using the WANDB run id as an identifier.
        scalers         : Element-wise scalers (fitted on the training split
                          only) used to de-normalize the spectra.
        fft             : To do or not the Forward Fourier Transform
        kmin            : Starting K value for FT window.
        kmax            : Ending K value for FT window.
        dk              : Tapering parameter for FT window.
        dk2             : Second tapering parameter for FT window.
        window          : Name of the window type.
        rmax_out        : Maximum number for R-space (default: 6 angstrom).

    Returns:
        Export JSON file with test dataset target and pred.
    """

    def __init__(
        self,
        filename: str,
        scalers: ElementScalers,
        fft: bool = True,
        kmin: float = 3.0,
        kmax: float = 15.0,
        kweight: float = 2.0,
        dk: float = 1.0,
        dk2: Optional[bool] = None,
        window: str = "hanning",
        rmax_out: float = 6.0,
    ):
        self.filename = filename
        self.scalers = scalers
        self.fft = fft
        self.kmin = kmin
        self.kmax = kmax
        self.kweight = kweight
        self.dk = dk
        self.dk2 = dk2
        self.window = window
        self.rmax_out = rmax_out

    def _interpolate(self, arr, dim):
        """Method to interpolate to a given dimension"""
        # Build interpolation method
        y_interp = interp.interp1d(np.arange(arr.size), arr)
        # Interpolate to a new dimension
        y_new = y_interp(np.linspace(0, arr.size - 1, dim))
        return y_new

    def on_test_end(self, trainer, pl_module):
        # Switch model to eval
        model = pl_module.eval()
        # Inference on test loader
        pred_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainer.test_dataloaders):
                # Move batch to GPU
                batch = batch.cuda()
                # Prediction
                y_hats = model.forward(batch)
                # Target
                targets = batch.target_y.reshape(
                    batch.batch_size, model.hparams.spectra_dim
                )
                # Material Id, struct_type and K-space
                material_ids = batch.material_id
                target_x = batch.target_x.reshape(
                    batch.batch_size, model.hparams.spectra_dim
                )

                # FIXME: struct_types = batch.struct_type

                # Loop
                for mat_id, y_hat, target, target_x in zip(
                    material_ids, y_hats, targets, target_x
                ):
                    # Move to cpu and array
                    y_hat = y_hat.cpu().numpy()
                    target = target.cpu().numpy()
                    target_x = target_x.cpu().numpy()
                    # Do FT or not
                    if not self.fft:
                        pred_list.append(
                            {
                                "id": mat_id,
                                "target": target.tolist(),
                                "predictions": y_hat.tolist(),
                                "wavenumber": target_x.tolist(),
                            }
                        )
                    else:
                        # De-normalization
                        abs_atom = mat_id.split("_")[1]
                        scaler = self.scalers[abs_atom]

                        target_denorm = scaler.inverse_transform(target)
                        y_hat_denorm = scaler.inverse_transform(y_hat)

                        # Diagnostic: a target outside [0, 1]
                        target_oob = bool(
                            target.min() < 0.0 or target.max() > 1.0
                        )

                        # Build Larch group object for Target
                        group_target = Group()
                        group_target.k = target_x
                        group_target.chi = target_denorm

                        # FT on Target
                        xas_fft(
                            group_target.k,
                            group_target.chi,
                            kmin=self.kmin,
                            kmax=self.kmax,
                            kweight=self.kweight,
                            dk=self.dk,
                            dk2=self.dk2,
                            window=self.window,
                            rmax_out=self.rmax_out,
                            group=group_target,
                        )

                        # Build Larch group object for Predicted
                        group_pred = Group()
                        group_pred.k = target_x
                        group_pred.chi = y_hat_denorm

                        # FT on Predicted
                        xas_fft(
                            group_pred.k,
                            group_pred.chi,
                            kmin=self.kmin,
                            kmax=self.kmax,
                            kweight=self.kweight,
                            dk=self.dk,
                            dk2=self.dk2,
                            window=self.window,
                            rmax_out=self.rmax_out,
                            group=group_pred,
                        )

                        # Interpolation dim
                        dim = pl_module.hparams.spectra_dim

                        # Append
                        pred_list.append(
                            {
                                "id": mat_id,
                                # FIXME: "struct_type": struct_type,
                                "target_out_of_unit_range": target_oob,
                                # Before FT
                                "target_chi": self._interpolate(
                                    target_denorm, dim=dim
                                ).tolist(),
                                "target_k": self._interpolate(
                                    target_x, dim=dim
                                ).tolist(),
                                "pred_chi": self._interpolate(
                                    y_hat_denorm, dim=dim
                                ).tolist(),
                                # After FT
                                "target_chir_mag": self._interpolate(
                                    group_target.chir_mag, dim=dim
                                ).tolist(),
                                "target_r": self._interpolate(
                                    group_target.r, dim=dim
                                ).tolist(),
                                "pred_chir_mag": self._interpolate(
                                    group_pred.chir_mag, dim=dim
                                ).tolist(),
                                "pred_r": self._interpolate(
                                    group_pred.r, dim=dim
                                ).tolist(),
                            }
                        )

        # Export json file
        print("Exporting JSON file...")
        with open(f"{self.filename}.json", "w") as j_file:
            json.dump(pred_list, j_file)
        j_file.close()
        return
