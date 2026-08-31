"""
Copyright (c) 2026 - ICIQ (Institute of Chemical Research of Catalonia)

Developed at ICIQ and funded by NCCR Catalysis.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from ai_exafs.preprocessing.graph import EXAFSGraph

# Paths
data_path = "./data/wf_tmQM_wB97MV.pkl"
db_path = "./data/wf_tmQM_wB97MV.lmdb"

# LMDB generator
extractor = EXAFSGraph(
    data_path=data_path,
    db_path=db_path,
    kmax=12.0,
    max_z=92,
    spectra_dim=928,
)

# Run
extractor.get_lmdb()
# extractor.get_pickle()
