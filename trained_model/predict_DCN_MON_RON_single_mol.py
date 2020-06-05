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

import os.path as osp
import os
import sys
sys.path.insert(0,'..')

import argparse
import torch
from torch.nn import Sequential, Linear, ReLU, GRU
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_max
import torch_geometric.transforms as T
from torch_geometric.nn import NNConv
from smiles_to_molecular_graphs.read_in_multitask import FUELNUMBERS
from smiles_to_molecular_graphs.single_molecule_conversion import process
from k_gnn import GraphConv, DataLoader, avg_pool
from k_gnn import TwoLocal

from datetime import datetime
import csv

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes > 1  # Remove graphs with less than 2 nodes

class MyPreTransform(object):
    def __call__(self, data):
        x = data.x
        data.x = data.x[:, :3]   # only consider atom types (H,C,O) of atom features vectors for determining isomorphic type in kgnn
        data = TwoLocal()(data)   # create higher-dimensional graph (2)
        data.x = x
        return data

# Define model settings
conv_type = 2   # number of graph convolutions in message passing for 1-GNN
conv_type2 = 2   # number of graph convolutions in message passing for 2-GNN
dim = 64   # size of nodes' hidden state vectors
pool_type = 'add'   # pooling function for combining nodes' hidden state vectors into molecular fingerprint
data_scaling = 'standard'   # data scaling (only standardization to zero mean and standard deviation of 1 is implemented yet)
parser = argparse.ArgumentParser()
parser.add_argument('--mol', default='CCc1ccc(OC)c(O)c1')   # define target property for single task learning (DCN, MON, or RON)
args = parser.parse_args()
smiles = str(args.mol)

print('---- Multitask ensemble prediction for DCN, MON, RON, Pooling: ADD ----')

# Path for training dataset
dataset_folder_name = '/DCN_MON_RON_joined/'
path = osp.join(osp.dirname(osp.realpath(__file__)), '../Data' + dataset_folder_name)

# Path for external test dataset
dataset_folder_name = '/DCN_MON_RON_joined/'
ext_path = osp.join(osp.dirname(osp.realpath(__file__)), '../Data' + dataset_folder_name)

# Load training dataset (only necessary for preprocessing kgnn) 
data = process(smiles)

dataset = FUELNUMBERS(
    path + 'Default/Train/',
    pre_transform=MyPreTransform(),
    pre_filter=MyFilter())

# Load external test dataset
ext_test_dataset = FUELNUMBERS(
    ext_path + 'Default/Test/',
    pre_transform=MyPreTransform(),
    pre_filter=MyFilter())

# Preprocessing kgnn
#torch.set_printoptions(profile="full")
#print(dataset.data.iso_type_2)
#torch.set_printoptions(profile="default")
dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
num_i_2 = int(dataset.data.iso_type_2.max().item() + 1)

x = data.x
data.x = data.x[:, :3]   # only consider atom types (H,C,O) of atom features vectors for determining isomorphic type in kgnn (H is only implicit in current molecular graphs, only included for future implementations where H is treated explicitly)
data = TwoLocal()(data)   # create higher-dimensional graph (2)
data.x = x

# TODO: for more general model, all possible isomorphism types should be included already in training (problem: if no molecules containing these additional isomorphism types are included in training, part of network will not be trained)
tmp_iso_type_2 = torch.tensor([4,5,8,13,14] + data.iso_type_2.tolist())   # workaround to include isomorphism types for 2-GNN included in the training data (CC, CO, OO, CC bonded, CO bonded)
tmp_iso_type_2 = torch.unique(tmp_iso_type_2, True, True)[1]
data.iso_type_2 = tmp_iso_type_2[5:]
data.iso_type_2 = F.one_hot(data.iso_type_2, num_classes=num_i_2).to(torch.float)

# model structure
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)

        nn = Sequential(Linear(12, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='add')
        self.gru = GRU(dim, dim)

        self.lin2 = torch.nn.Linear(dim + num_i_2, dim)
        self.conv4 = GraphConv(dim, dim)
        self.gru2 = GRU(dim, dim)


        self.fc1 = torch.nn.Linear(2*dim, 64)
        self.fc11 = torch.nn.Linear(64, 32)
        self.fc12 = torch.nn.Linear(32, 16)
        self.fc13 = torch.nn.Linear(16, 1)
        self.fc2 = torch.nn.Linear(2*dim, 64)
        self.fc21 = torch.nn.Linear(64, 32)
        self.fc22 = torch.nn.Linear(32, 16)
        self.fc23 = torch.nn.Linear(16, 1)
        self.fc3 = torch.nn.Linear(2*dim, 64)
        self.fc31 = torch.nn.Linear(64, 32)
        self.fc32 = torch.nn.Linear(32, 16)
        self.fc33 = torch.nn.Linear(16, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(conv_type):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        x_forward = out

        x_1 = scatter_add(x_forward, data.batch, dim=0)

        if pool_type is 'mean':
            x_1 = scatter_mean(x_forward, data.batch, dim=0)
        if pool_type is 'max':
            x_1 = scatter_max(x_forward, data.batch, dim=0)[0]


        data.x = avg_pool(x_forward, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        out = F.relu(self.lin2(data.x))
        h = out.unsqueeze(0)
        for i in range(conv_type2):
            m = F.relu(self.conv4(out, data.edge_index_2))
            out, h = self.gru2(m.unsqueeze(0), h)
            out = out.squeeze(0)
        x = out

        x_2 = scatter_add(x, data.batch_2, dim=0)

        if pool_type is 'mean':
            x_2 = scatter_mean(x, data.batch, dim=0)
        if pool_type is 'max':
            x_2 = scatter_max(x, data.batch, dim=0)[0]


        x = torch.cat([x_1, x_2], dim=1)

        x1 = F.elu(self.fc1(x))
        x1 = F.elu(self.fc11(x1))
        x1 = F.elu(self.fc12(x1))
        x1 = self.fc13(x1)
        x2 = F.elu(self.fc2(x))
        x2 = F.elu(self.fc21(x2))
        x2 = F.elu(self.fc22(x2))
        x2 = self.fc23(x2)
        x3 = F.elu(self.fc3(x))
        x3 = F.elu(self.fc31(x3))
        x3 = F.elu(self.fc32(x3))
        x3 = self.fc33(x3)
        x = torch.cat([x1, x2, x3], dim=1)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Operating on following hardware: ' + str(device))
model = Net().to(device)

def predict(loader):
    model.eval()
    mol_id, pred, real_value, mol_names, pred_list, errors = None, None, None, [], [], []
    for data in loader:
        mol_id = data.mol_id.tolist()
        for mol in mol_id:
            tmp_mol_name = ''
            for i in mol: 
                if int(i) is not 0:
                    tmp_mol_name += chr(int(i))
            mol_names.append(tmp_mol_name)
        data = data.to(device)
        pred = model(data).tolist()

        for c, k in enumerate(pred):
            if data_scaling is 'standard':
                pred_list.append([mol_names[c],(pred[c][0]*std[0]+mean[0]).item(),(pred[c][1]*std[1]+mean[1]).item(),(pred[c][2]*std[2]+mean[2]).item()])
        
    return pred_list

# read in mean and standard deviations of training data of trained models
mean_dict = {}
std_dict = {}

with open('Model_Parameters/DCN_MON_RON_multi_task/standardization.txt', 'r') as stand:
    for line in stand:
        model_id = line.split(":")[0].split("-")
        if model_id[1] == 'mean':
             mean_dict[int(model_id[0])] = torch.tensor(eval(line.split(":")[1].split()[0]))
        if model_id[1] == 'std':
             std_dict[int(model_id[0])] = torch.tensor(eval(line.split(":")[1].split()[0]))

pred_dict = {}
num_models = 0

# Calculate predictions for each of the trained models 
for i in range(1,41):

    num_models += 1
    tmp_pred = []
    
    mean = mean_dict[i]
    std = std_dict[i]

    my_dataset = [data] # create your datset
    test_loader = DataLoader(my_dataset)

    model.load_state_dict(torch.load('Model_Parameters/DCN_MON_RON_multi_task/base_model_' + str(i) + '.pt', map_location=torch.device(device)))
    tmp_pred = predict(test_loader)

    for pred in tmp_pred:
        tmp_smiles, tmp_pred_dcn, tmp_pred_mon, tmp_pred_ron = pred[0], pred[1], pred[2], pred[3]
        if pred_dict.get(tmp_smiles) is not None:
            pred_dict[tmp_smiles][0] = pred_dict[tmp_smiles][0] + float(tmp_pred_dcn)
            pred_dict[tmp_smiles][1] = pred_dict[tmp_smiles][1] + float(tmp_pred_mon)
            pred_dict[tmp_smiles][2] = pred_dict[tmp_smiles][2] + float(tmp_pred_ron)

        else:
            pred_dict[tmp_smiles] = [float(tmp_pred_dcn), float(tmp_pred_mon), float(tmp_pred_ron)]

for mol, results in pred_dict.items():
    print(mol + '  -  DCN: ' + str(results[0]/num_models) + ',  MON: ' + str(results[1]/num_models) + ',  RON: ' + str(results[2]/num_models))

