#**********************************************************************************
# Copyright (c) 2020 Process Systems Engineering (AVT.SVT), RWTH Aachen University
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.
#
# SPDX-License-Identifier: EPL-2.0
#
# The source code can be found here:
# https://git.rwth-aachen.de/avt.svt/private/graph_neural_network_for_fuel_ignition_quality.git
#
#*********************************************************************************

import os
import os.path as osp

import time

import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import rdBase
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit.Chem import Draw
    rdBase.DisableLog('rdApp.error')
except ImportError:
    rdkit = None


def process(smiles):
    if rdkit is None:
        print('Please install `rdkit` to process the raw data.')
        return None

    mol = Chem.rdmolfiles.MolFromSmiles(smiles)
    mol_block = Chem.MolToMolBlock(mol)
    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None:
        print('Invalid molecule (None)')
        return None

    N = mol.GetNumAtoms()

    # only consider molecules with more than one atom -> need at least one bond for a valid graph
    if N <= 1:
        print('Warning: molecule skipped because it contains only 1 atom')
        return None

    types = {'H': 0, 'C': 1, 'O': 2} # atom types
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3} #bond types


    # atom features
    type_idx = []
    aromatic = []
    ring = []
    sp = []
    sp2 = []
    sp3 = []
    sp3d = []
    sp3d2 = []
    num_hs = []
    num_neighbors = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        ring.append(1 if atom.IsInRing() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
        sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)
        sp3d2.append(1 if hybridization == HybridizationType.SP3D2 else 0)
        num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
        num_neighbors.append(len(atom.GetNeighbors()))

    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor([aromatic, ring, sp, sp2, sp3, sp3d, sp3d2], dtype=torch.float).t().contiguous()
    x3 = F.one_hot(torch.tensor(num_neighbors), num_classes=6)
    x4 = F.one_hot(torch.tensor(num_hs), num_classes=5)
    x = torch.cat([x1.to(torch.float), x2, x3.to(torch.float),x4.to(torch.float)], dim=-1)

    # bond features
    row, col, bond_idx, conj, ring, stereo = [], [], [], [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_idx += 2 * [bonds[bond.GetBondType()]]
        conj.append(bond.GetIsConjugated())
        conj.append(bond.GetIsConjugated())
        ring.append(bond.IsInRing())
        ring.append(bond.IsInRing())
        stereo.append(bond.GetStereo())
        stereo.append(bond.GetStereo())

    edge_index = torch.tensor([row, col], dtype=torch.long)
    e1 = F.one_hot(torch.tensor(bond_idx),num_classes=len(bonds)).to(torch.float)
    e2 = torch.tensor([conj, ring], dtype=torch.float).t().contiguous()
    e3 = F.one_hot(torch.tensor(stereo),num_classes=6).to(torch.float)
    edge_attr = torch.cat([e1, e2, e3], dim=-1)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

    # transform SMILES into ascii data type and store it in a name tensor
    name = str(Chem.MolToSmiles(mol))
    ascii_name = []
    for c in name:
        ascii_name.append(int(ord(c)))

    # if fails, increase range
    for i in range(len(ascii_name), 1000):
        ascii_name.append(0)

    ascii_name = torch.tensor([ascii_name], dtype=torch.float).contiguous()

    # print current molecule with target data
    print(str(name))

    # save data
    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, mol_id=ascii_name)

    return data
