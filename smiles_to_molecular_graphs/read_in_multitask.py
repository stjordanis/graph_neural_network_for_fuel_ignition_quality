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


class FUELNUMBERS(InMemoryDataset):
    r""" DCN,MON,RON data

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    raw_url = ''
    processed_url = ''


    if rdkit is not None:
        types = {'H': 0, 'C': 1, 'O': 2} # atom types
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3} #bond types

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(FUELNUMBERS, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'raw.pt' if rdkit is None else 'raw.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

        #url = self.processed_url if rdkit is None else self.raw_url
        #file_path = download_url(url, self.raw_dir)
        #extract_zip(file_path, self.raw_dir)
        #os.unlink(file_path)

    def process(self):
        if rdkit is None:
            print('Using a pre-processed version of the dataset. Please '
                  'install `rdkit` to alternatively process the raw data.')

            self.data, self.slices = torch.load(self.raw_paths[0])
            data_list = [data for data in self]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            return

        molecules = []
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1] #skip first line
            molecules = [[str(x) for x in line.split(";")[1:2]] for line in data]    #('<separator of letter>') in split, whitespace corresponds to blank parentheses 
            writer = Chem.SDWriter(str(self.root) + '/raw/raw.sdf')
            for m in molecules:
                mol = Chem.rdmolfiles.MolFromSmiles(m[0])
                #mol = Chem.rdmolops.AddHs(mol) ### explicit trivial Hs (excluded)
                writer.write(mol)
            del writer

            # read target data
            target = []
            for line in data:
                # targets initialize with placeholder (infinity is excluded from loss calculation for later training)
                tmp_dcn, tmp_mon, tmp_ron = float("Inf"),float("Inf"),float("Inf")
                
                targets = line.split(";")[2:5]
                
                # add target data, if present in raw data file
                if targets[0] is not '':
                    tmp_dcn = float(targets[0])
                if targets[1] is not '':
                    tmp_mon = float(targets[1])
                if targets[2] is not '':
                    tmp_ron = float(targets[2])

                target.append([tmp_dcn, tmp_mon, tmp_ron])

            target = torch.tensor(target, dtype=torch.float)


        # delay for proper saving of sdf file
        time.sleep(10)
            
        dataset = str(self.root) + '/raw/raw.sdf'
        suppl = Chem.SDMolSupplier(dataset, removeHs=False)
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

        data_list = []
        max_len = 0
        max_name = ''
        if len(target) == len(suppl):
            print('True')
        else:
            print('Fail: number of target data points does not match number of molecules')

        for i, mol in enumerate(suppl):
            if mol is None:
                print('Invalid molecule (None)')
                continue

            text = suppl.GetItemText(i)
            N = mol.GetNumAtoms()

            # only consider molecules with more than one atom -> need at least one bond for a valid graph
            if N <= 1:
                print('Warning: molecule skipped because it contains only 1 atom')
                continue

            # spatial positions of atom (Note: not included in training, can be used for future implementations)
            pos = text.split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)

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
                type_idx.append(self.types[atom.GetSymbol()])
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

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
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
                bond_idx += 2 * [self.bonds[bond.GetBondType()]]
                conj.append(bond.GetIsConjugated())
                conj.append(bond.GetIsConjugated())
                ring.append(bond.IsInRing())
                ring.append(bond.IsInRing())
                stereo.append(bond.GetStereo())
                stereo.append(bond.GetStereo())

            edge_index = torch.tensor([row, col], dtype=torch.long)
            e1 = F.one_hot(torch.tensor(bond_idx),num_classes=len(self.bonds)).to(torch.float)
            e2 = torch.tensor([conj, ring], dtype=torch.float).t().contiguous()
            e3 = F.one_hot(torch.tensor(stereo),num_classes=6).to(torch.float)
            edge_attr = torch.cat([e1, e2, e3], dim=-1)
            edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

            # target data
            mol_id = i+1
            y = target[i].unsqueeze(0)

            # transform SMILES into ascii data type and store it in a name tensor
            name = str(Chem.MolToSmiles(mol))
            ascii_name = []
            for c in name:
                ascii_name.append(int(ord(c)))

            if len(ascii_name) > max_len:
                max_len = len(ascii_name)
                max_name = name


            # if fails, increase range
            for i in range(len(ascii_name), 300):
                ascii_name.append(0)

            ascii_name = torch.tensor([ascii_name], dtype=torch.float).contiguous()

            # print current molecule with target data
            print(str(name) + ': ' + ','.join('{}'.format(x) for x in y.tolist()))

            # save data
            data = Data(x=x, pos=pos, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, mol_id=ascii_name)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
