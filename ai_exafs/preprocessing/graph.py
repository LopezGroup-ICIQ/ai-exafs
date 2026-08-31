"""
Copyright (c) 2026 - ICIQ (Institute of Chemical Research of Catalonia)

Developed at ICIQ and funded by NCCR Catalysis.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pickle
import lmdb
from tqdm import tqdm

import numpy as np
import scipy.interpolate as interp
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt

import torch
from torch_geometric.utils import to_undirected, to_networkx
from torch_geometric.data import Data
import networkx as nx

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ai_exafs.preprocessing.embeddings import COV_RADII


class EXAFSGraph:
    """
    General method that builds the graph representation, transforms EXAFS spectra
    and stores dataset as LMDB or pickle file.

    Args:
        data_path   : Path directing to the pickle file with raw data.
        db_path     : Path to store the lmdb file (e.g., data)
        kmax        : To cut the EXAFS spectra up until this value (default: 15.0)
        max_z       : To build the one hot embeddings for atoms (default: 92)
        spectra_dim : To interpolate the EXAFS spectra to a given dimension (default: 928)

    Returns:
        LMDB dataset or Pickle file.
    """

    def __init__(
        self,
        data_path: str,
        db_path: str,
        kmax: float = 15.0,
        max_z: int = 92,
        spectra_dim: int = 928,
    ):
        self.data_path = data_path
        self.db_path = db_path
        self.kmax = kmax
        self.max_z = max_z
        self.spectra_dim = spectra_dim

        # Raw Data
        self.raw_data = self._load_pickle()
        self.length = len(list(self.raw_data.keys()))

        # Generate the elements ONE_HOT_EMBEDDINGS dict
        self.ONE_HOT_EMBEDDINGS = self._generate_one_hot_embeddings()

        # Group ids by absorbing atom
        self.abs_atom_ids = self._get_abs_atom_ids()
        self.abs_atom_length = len(self.abs_atom_ids.keys())

        # Interpolate the EXAFS spectra - without normalization
        self.exafs_dict, self.k_interp = self._prepare_exafs()

        # Transform to Graphs
        self.graph_data, self.filtered_ids = self._transform()
        self.length = len(self.graph_data)
        self.length_filtered = len(self.filtered_ids)
        print(f"Filtered Graphs: {self.length_filtered}")

    def _load_pickle(self):
        """Method to load pickle file"""
        with open(self.data_path, "rb") as p_file:
            data = pickle.load(p_file)
        p_file.close()
        return data

    def _get_absorbing_atom_symbol_index(self, absorbing_atom, structure):
        """Method that get the site index from structure providing the absorbing atom symbol"""
        return structure.indices_from_symbol(absorbing_atom)[0]

    def _struct_2_atoms(self, structure, absorbing_atom):
        """Method that convert PMG structure to ASE atoms object"""
        # Initialize method and get atoms object
        ase_adpt = AseAtomsAdaptor()
        atoms = ase_adpt.get_atoms(structure)
        # Center structure
        for atom in atoms:
            if atom.symbol == absorbing_atom:
                abs_atom_pos = np.array(atom.position)
                atoms.translate(-1 * abs_atom_pos)
        return atoms

    def _get_edge_features(self, atoms, abs_atom_idx, tol=0.5, sf=1.5):
        """Method that uses voronoi and distance filters including single scattering paths"""
        # Get coordinates
        coords = atoms.positions
        # Get Voronoi tesellation
        vor_bonds = Voronoi(coords)
        pairs_corr = np.unique(np.sort(vor_bonds.ridge_points, axis=1), axis=0)
        # Filter bond pairs by distance
        bond_pairs = [[], []]
        for pair in pairs_corr:
            distance = atoms.get_distance(pair[0], pair[1])
            if abs_atom_idx in pair:
                threshold = (
                    COV_RADII[atoms[pair[0]].symbol]
                    + COV_RADII[atoms[pair[1]].symbol]
                    + tol
                    + (sf - 1.0)
                )
            else:
                threshold = (
                    COV_RADII[atoms[pair[0]].symbol]
                    + COV_RADII[atoms[pair[1]].symbol]
                    + tol
                )
            if distance <= threshold:
                bond_pairs[0] += [pair[0]]
                bond_pairs[1] += [pair[1]]

        # Get Single Scattering paths with abs_atom_idx as origin.
        ss_pairs = [[], []]
        for atom_idx in range(len(atoms)):
            if atom_idx != abs_atom_idx:
                # Get interaction
                ss_pairs[0] += [abs_atom_idx]
                ss_pairs[1] += [atom_idx]

        # Merge bond_pairs and ss_pairs in a unique list of interactions
        bond_pairs_transpose = np.array(bond_pairs, dtype=int).T
        ss_pairs_transpose = np.array(ss_pairs, dtype=int).T

        # Vertical stack
        combined_array = np.vstack((bond_pairs_transpose, ss_pairs_transpose))

        # Sorting
        sorted_pairs = np.sort(combined_array, axis=1)

        # Only unique indexes
        unique_pairs = np.unique(sorted_pairs, axis=0)

        # Get pair distance as edge attribute from the combined one
        pair_distances = []
        for pair in unique_pairs:
            distance = atoms.get_distance(pair[0], pair[1])
            pair_distances.append(distance)

        return unique_pairs.T, np.array(pair_distances)

    def _generate_one_hot_embeddings(self):
        """Method to generate encoded vectors for atomic number up to max_z"""
        node_embeddings = {}
        for z in range(1, self.max_z + 1):
            one_hot_vector = [0] * self.max_z
            one_hot_vector[z - 1] = 1
            node_embeddings[z] = one_hot_vector
        return node_embeddings

    def _get_node_features(self, atoms):
        """Method to get the node features from an atoms object"""
        return np.stack(
            [
                self.ONE_HOT_EMBEDDINGS[atoms[i].number]
                for i in range(len(atoms))
            ]
        )

    def _interpolate(self, arr):
        """Method to interpolate to a give dimension"""
        # Build interpolation method
        y_interp = interp.interp1d(np.arange(arr.size), arr)
        # Interpolate to a new dimension
        y_new = y_interp(np.linspace(0, arr.size - 1, self.spectra_dim))
        return y_new

    def _limit_krange_by_kmax(self, exafs_data):
        """Cut EXAFS spectra in k-space by kmax variable"""
        # Extract (x,y)
        wavenumber = exafs_data[0, :]
        chi_kweight = exafs_data[1, :]
        # Cut spectra until kmax
        wavenumber_kmax = np.array([x for x in wavenumber if x <= self.kmax])
        chi_kmax = np.array(
            [chi for k, chi in zip(wavenumber, chi_kweight) if k <= self.kmax]
        )
        # exafs_data_kmax new array
        exafs_data_kmax = np.array([wavenumber_kmax, chi_kmax])
        return exafs_data_kmax

    def _plot_graph(self, graph):
        """Helper method to plot the given graph"""
        g = to_networkx(graph)
        nx.draw(g, pos=nx.kamada_kawai_layout(g), with_labels=True)
        plt.show()
        return

    def _get_abs_atom_ids(self):
        """Method that groups the unique ids by absorbing atom"""
        abs_atom_ids_dict = {}
        # Loop over raw data
        for idx, (mp_id, data) in enumerate(self.raw_data.items()):
            # Absorbing atom
            absorbing_atom = mp_id.split("_")[1]
            # Appending
            if absorbing_atom not in abs_atom_ids_dict.keys():
                abs_atom_ids_dict.update({absorbing_atom: [mp_id]})
            else:
                abs_atom_ids_dict[absorbing_atom].append(mp_id)
        return abs_atom_ids_dict

    def _prepare_exafs(self):
        """Method that slices and interpolates the EXAFS spectra without normalization"""
        exafs_dict = {}
        k_interp = None
        pbar = tqdm(
            total=self.length,
            desc="Preparing EXAFS spectra",
        )
        # Loop over raw data, grouped by absorbing atom, to transform spectra
        for abs_atom, mp_ids in self.abs_atom_ids.items():
            for mp_id in mp_ids:
                # Get K-space from raw data as already (chi*k**2)
                exafs_data = self.raw_data[mp_id]["k_space"]
                # Slice EXAFS up until kmax
                exafs_data_kmax = self._limit_krange_by_kmax(
                    exafs_data=exafs_data,
                )
                # Extract chi and k for exafs_data_kmax
                exafs_data_k = exafs_data_kmax[0, :]
                exafs_data_chi = exafs_data_kmax[1, :]
                # Interpolate EXAFS in K-space
                k_interp = self._interpolate(exafs_data_k)
                chi_mag_interp = self._interpolate(exafs_data_chi)
                # Store the raw spectrum
                exafs_dict[mp_id] = chi_mag_interp

                # Progress Bar
                pbar.update(1)

        # The k-axis shared by every graph: must match the published model
        print(f"k_interp: [{k_interp[0]:.5f} .. {k_interp[-1]:.5f}]")
        return exafs_dict, k_interp

    def _get_graph(
        self,
        material_id,
        absorbing_atom,
        struct_dict,
        target_y,
        target_x,
    ):
        """Method that returns the PyG Data object as graph representation"""
        # Convert struct_dict into PMG structure
        struct = Structure.from_dict(struct_dict)
        # Convert Pymatgen structure to ASE atoms object
        atoms = self._struct_2_atoms(struct, absorbing_atom)
        abs_atom_index = self._get_absorbing_atom_symbol_index(
            absorbing_atom=absorbing_atom,
            structure=struct,
        )
        # Get edge features and convert to undirected
        bond_index, bond_attr = self._get_edge_features(atoms, abs_atom_index)
        bond_idx_undirected, bond_attr_undirected = to_undirected(
            edge_index=torch.LongTensor(bond_index),
            edge_attr=torch.Tensor(bond_attr),
        )
        bond_attr_undirected = torch.unsqueeze(
            bond_attr_undirected, 1
        )  # (n, 1)
        # Get atom features
        atom_features = self._get_node_features(atoms)
        atom_features = torch.Tensor(atom_features)
        # Get coordinates
        coords = torch.Tensor(atoms.positions)
        # Targets
        target_y = torch.Tensor(target_y)
        target_x = torch.Tensor(target_x)
        # Build PyG Data object
        graph = Data(
            x=atom_features,
            edge_index=bond_idx_undirected,
            edge_attr=bond_attr_undirected,
            pos=coords,
            atom_idx=[idx for idx, _ in enumerate(atoms)],
            atom_symbols=atoms.get_chemical_symbols(),
            material_id=material_id,
            target_y=target_y,
            target_x=target_x,
        )
        return graph

    def _remove_isolated_nodes(self, graph):
        """Helper function to remove isolated nodes from graph checking metal and subgraph length"""
        g_nx = to_networkx(graph, to_undirected=True)
        g_connects = list(nx.connected_components(g_nx))
        g_lens = [len(g) for g in g_connects]
        max_len = max(g_lens)
        for idx, (g_connect, g_len) in enumerate(zip(g_connects, g_lens)):
            graph_sub = graph.subgraph(torch.LongTensor(list(g_connect)))
            has_metal = [
                x
                for x in graph_sub.atom_symbols
                if Element(x).is_transition_metal
            ]
            if has_metal and g_len == max_len:
                graph_sub = graph.subgraph(
                    torch.LongTensor(list(g_connects[idx]))
                )
                return graph_sub
            else:
                return None

    def _transform(self):
        """Method that loop over raw data to transform into graph representation"""
        graph_data_list = []
        filtered_ids = []

        # Progress Bar
        pbar = tqdm(
            total=self.length,
            desc="Transforming structure to Graph representation",
        )
        # Loop over raw data to transform structures into graphs
        pbar_idx = 0
        for idx, (mp_id, data) in enumerate(self.raw_data.items()):
            # Absorbing atom from label
            absorbing_atom = mp_id.split("_")[1]
            # Get the normalized EXAFS spectra
            target_y = self.exafs_dict[mp_id]
            # Build the graph
            graph = self._get_graph(
                material_id=mp_id,
                absorbing_atom=absorbing_atom,
                struct_dict=data["structure"],
                target_y=target_y,
                target_x=self.k_interp,
            )
            # Clean graph of disconnected parts
            if graph.has_isolated_nodes():
                graph = self._remove_isolated_nodes(graph)
            # Append
            if graph:
                graph_data_list.append(graph)
            else:
                filtered_ids.append(mp_id)

            # Progress Bar
            pbar_idx += 1
            pbar.update(1)

        return graph_data_list, filtered_ids

    def get_lmdb(self):
        """Create LMDB file with graph data objects"""
        # Initialize DB
        db = lmdb.open(
            self.db_path,
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )

        # Progress Bar
        pbar = tqdm(
            total=self.length,
            desc="Transferring Graph data into LMDB",
        )

        # Loop over graph data
        pbar_idx = 0
        for idx, graph in enumerate(self.graph_data):
            # Write in the LMDB file
            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(graph, protocol=-1))
            txn.commit()

            # Update pbar
            pbar_idx += 1
            pbar.update(1)

        # Save some metadata inside LMDB
        txn = db.begin(write=True)
        txn.put("length".encode("ascii"), pickle.dumps(self.length, protocol=-1))
        txn.put(
            "filtered".encode("ascii"),
            pickle.dumps(self.filtered_ids, protocol=-1),
        )

        # The LMDB holds raw spectra
        txn.put(
            "normalization".encode("ascii"),
            pickle.dumps(
                {
                    "target_units": "raw_chi_k2",
                    "kmax": self.kmax,
                    "spectra_dim": self.spectra_dim,
                },
                protocol=-1,
            ),
        )
        txn.commit()

        # Close DB
        db.sync()
        db.close()
        return

    def get_pickle(self):
        """Create pickle file with graph data objects"""
        with open(f"{self.db_path}.pkl", "wb") as p_file:
            pickle.dump(self.graph_data, p_file)
        p_file.close()
        return
