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


print('---- Multitask ensemble prediction for DCN, MON, RON, Pooling: ADD ----')

# Path for training dataset
dataset_folder_name = '/DCN_MON_RON_joined/'
path = osp.join(osp.dirname(osp.realpath(__file__)), '../Data' + dataset_folder_name)

# Path for external test dataset
dataset_folder_name = '/DCN_MON_RON_joined/'
ext_path = osp.join(osp.dirname(osp.realpath(__file__)), '../Data' + dataset_folder_name)

# Load training dataset (only necessary for preprocessing kgnn) 
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
dataset.data.iso_type_2 = torch.unique(dataset.data.iso_type_2, True, True)[1]
num_i_2 = int(dataset.data.iso_type_2.max().item() + 1)
ext_test_dataset.data.iso_type_2 = torch.unique(ext_test_dataset.data.iso_type_2, True, True)[1]
## make sure num_i_2 equals ext_num_i_2, otherwise add isomorphism types (see predict_DCN_MON_RON_single_mol.py)
#ext_num_i_2 = int(ext_test_dataset.data.iso_type_2.max().item() + 1)
ext_test_dataset.data.iso_type_2 = F.one_hot(ext_test_dataset.data.iso_type_2, num_classes=num_i_2).to(torch.float)

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
        real_value = data.y.tolist()
        data = data.to(device)
        pred = model(data).tolist()

        for c, k in enumerate(pred):
            if data_scaling is 'standard':
                pred_list.append([mol_names[c],(pred[c][0]*std[0]+mean[0]).item(),(pred[c][1]*std[1]+mean[1]).item(),(pred[c][2]*std[2]+mean[2]).item(), (real_value[c][0]*std[0]+mean[0]).item(), (real_value[c][1]*std[1]+mean[1]).item(), (real_value[c][2]*std[2]+mean[2]).item(), ((pred[c][0]-real_value[c][0])*std[0]).abs().item(), ((pred[c][1]-real_value[c][1])*std[1]).abs().item(), ((pred[c][2]-real_value[c][2])*std[2]).abs().item(), abs((pred[c][0]-real_value[c][0])/(real_value[c][0]+mean[0]/std[0])).item()])
        
    return pred_list

# Define save path and make directories
def preprocess_save(dataset_folder_name):
    model_type = 'base_models'
    save_dir_model_type = str(osp.dirname(osp.realpath(__file__))) + '/Predictions/' + model_type
    save_dir = save_dir_model_type + '/' + dataset_folder_name   # directory based on target
    save_path = save_dir + 'Test'

    try:
        os.mkdir(save_dir_model_type)
    except:
        print('Model type directory already exists.')
    try:
        os.mkdir(save_dir_target)
    except:
        print('Target directory already exists.')
    try:
        os.mkdir(save_dir)
    except:
        print('Base model for this dataset and target already exits.')
    try:
        os.mkdir(save_path)
    except:
        print('Prediction directory already exists - older model is replaced by model from this training run.')
    return save_path

save_path = preprocess_save(dataset_folder_name)

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

    ext_test_dataset.data.y = (ext_test_dataset.data.y - mean) / std
    ext_test_dataset = ext_test_dataset[:]
    dataset_size = len(ext_test_dataset)
    ext_test_loader = DataLoader(ext_test_dataset, batch_size=dataset_size)

    model.load_state_dict(torch.load('Model_Parameters/DCN_MON_RON_multi_task/base_model_' + str(i) + '.pt', map_location=torch.device(device)))
    tmp_pred = predict(ext_test_loader)

    for pred in tmp_pred:
        tmp_smiles, tmp_pred_dcn, tmp_pred_mon, tmp_pred_ron, tmp_real_dcn, tmp_real_mon, tmp_real_ron, tmp_error_dcn, tmp_error_mon, tmp_error_ron = pred[0], pred[1], pred[2], pred[3], pred[4], pred[5], pred[6], pred[7], pred[8], pred[9]
        if pred_dict.get(tmp_smiles) is not None:
            pred_dict[tmp_smiles][0] = pred_dict[tmp_smiles][0] + float(tmp_pred_dcn)
            pred_dict[tmp_smiles][1] = pred_dict[tmp_smiles][1] + float(tmp_pred_mon)
            pred_dict[tmp_smiles][2] = pred_dict[tmp_smiles][2] + float(tmp_pred_ron)
            pred_dict[tmp_smiles][3] = pred_dict[tmp_smiles][3] + float(tmp_real_dcn)
            pred_dict[tmp_smiles][4] = pred_dict[tmp_smiles][4] + float(tmp_real_mon)
            pred_dict[tmp_smiles][5] = pred_dict[tmp_smiles][5] + float(tmp_real_ron)
            pred_dict[tmp_smiles][6] = pred_dict[tmp_smiles][6] + float(tmp_error_dcn)
            pred_dict[tmp_smiles][7] = pred_dict[tmp_smiles][7] + float(tmp_error_mon)
            pred_dict[tmp_smiles][8] = pred_dict[tmp_smiles][8] + float(tmp_error_ron)

        else:
            pred_dict[tmp_smiles] = [float(tmp_pred_dcn), float(tmp_pred_mon), float(tmp_pred_ron),float(tmp_real_dcn), float(tmp_real_mon), float(tmp_real_ron),float(tmp_error_dcn), float(tmp_error_mon), float(tmp_error_ron)]

    ext_test_dataset.data.y = (ext_test_dataset.data.y * std) + mean

#for mol, results in pred_dict.items():
#    print([mol,results[0]/num_models,results[1]/num_models,results[2]/num_models])


with open(save_path + "/predictions.csv","a+") as pred_file:
    pred_file.write('\n ' + 'SMILES' + ',' + 'Predicted DCN' + ',' + 'Predicted MON' + ',' + 'Predicted RON' + ',' + 'Measured DCN' + ',' + 'Measured MON' + ',' + 'Measured RON' + ',' + 'Absolute Error DCN' + ',' + 'Absolute Error MON' + ',' + 'Absolute Error MON') 
    for mol, results in pred_dict.items():
        pred_file.write('\n ' + str(mol) + ',' + str(results[0]/num_models) + ',' + str(results[1]/num_models) + ',' + str(results[2]/num_models) + ',' + str(results[3]/num_models) + ',' + str(results[4]/num_models) + ',' + str(results[5]/num_models) + ',' + str(results[6]/num_models) + ',' + str(results[7]/num_models) + ',' + str(results[8]/num_models))

