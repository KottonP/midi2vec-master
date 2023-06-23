import time
import numpy as np
import pandas as pd
import winsound

import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from torch.utils.data import RandomSampler, SequentialSampler

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, SAGEConv, GATConv, Linear, models, to_hetero
from torch_geometric.loader import DataLoader, HGTLoader, NeighborLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader
from tqdm import tqdm

from HeteroDataFunctions import Encoder, add_types, complete_graph, flatten_lol, node_cat_dict, midi_type, plot_graph, plot_4graphs

# print(scipy.__version__)
# print(matplotlib.__version__)
# print(nx.__version__)
print(torch.__version__)
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
# # Complete Dataset
# G = complete_graph(".\slac\embeddings\\all")
# nx.write_edgelist(G, ".\slac\homograph.edgelist", data=False)
G = nx.read_edgelist(".\slac\homograph.edgelist")
nodes = pd.DataFrame((list(G.nodes)), columns=['name'])
edges = pd.DataFrame(np.array(list(G.edges)), columns=['source', 'target'])
node_categories = node_cat_dict(nodes)
node_categories.keys()


nodes_df_complete = pd.read_csv('.\slac\Contents of Slac\\nodes_complete.csv')
edges_df_complete = pd.read_csv('.\slac\Contents of Slac\edges_complete.csv')
print('Done')
node_types = set(nodes_df_complete['node_type'])
node_types

edge_types = ["MIDI__has__tempo",
                   "MIDI__in__time_sig",
                   "MIDI__has__program",
                   "MIDI__has__note_group",
                   "note_group__has__velocity",
                   "note_group__has__duration",
                   "note_group__contains__pitch"]
names_list = flatten_lol(node_categories.values())

encoder = Encoder(names_list, n_labels=10)

input_node_dict = {node_type: {'x': encoder.
                    encode_nodes(nodes_df_complete.
                    loc[nodes_df_complete['node_type'] == node_type, ['name']])}
                    for node_type in node_types}
node_enc_to_idx = {node_type: {encoder.decode_value(node_enc.item()): i for i, node_enc in enumerate(input_node_dict[node_type]['x'])} for node_type in node_types}
input_edge_dict = dict()
for edge_type in edge_types:
    node_type_s, node_type_t = edge_type.split('__')[0], edge_type.split('__')[2]

    edge_df = edges_df_complete.loc[edges_df_complete['edge_type'] == edge_type, ['source', 'target']].copy()

    edge_df['source'], edge_df['target'] = edge_df['source'].map(node_enc_to_idx[node_type_s]), edge_df['target'].map(node_enc_to_idx[node_type_t])

    input_edge_dict[edge_type] = {'edge_index': torch.tensor(edge_df.values).T}

# Extract the label of each Midi.
midi_val = nodes_df_complete.loc[nodes_df_complete['node_type'] == 'MIDI', ['name']].values
midi_class_5 = [midi_type(s[0], 5) for s in midi_val]

lb = LabelEncoder()
y_5 = torch.from_numpy(lb.fit_transform(midi_class_5)) # .type(torch.LongTensor)

lb.classes_
input_node_dict_5 = input_node_dict.copy()

input_node_dict_5['MIDI']['y'] = y_5
# H_5 = HeteroData(input_node_dict_5, **input_edge_dict).to(device)
# torch.save(H_5, '.\slac\H_5.pt')
H_5 = torch.load('.\slac\H_5.pt')
print(H_5)
H_5 = T.ToUndirected()(H_5)
#H_5 = T.RandomNodeSplit(num_val=0.1, num_test=0.1)(H_5)
print(H_5)
# GNN
model_5 = models.GraphSAGE(in_channels=-1, hidden_channels=64, num_layers=2, out_channels=len(set(lb.classes_)))
model_5 = to_hetero(model_5, H_5.metadata(), aggr='sum')
# model_5.to(device)
# model_5 = torch_geometric.compile(model_5)
def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None):

    val_losses, accs, durations = [], [], []
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        dataset['MIDI'].train_mask = train_idx
        dataset['MIDI'].val_mask = val_idx
        dataset['MIDI'].test_mask = test_idx

        train_dataset = dataset['MIDI'].x[dataset['MIDI'].train_mask]
        test_dataset = dataset['MIDI'].x[dataset['MIDI'].test_mask]
        val_dataset = dataset['MIDI'].x[dataset['MIDI'].val_mask]


        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_input_nodes = ('MIDI', train_dataset.to(torch.long))
        test_input_nodes = ('MIDI', test_dataset.to(torch.long))
        val_input_nodes = ('MIDI', val_dataset.to(torch.long))

        train_loader = HGTLoader(dataset, num_samples=[batch_size] * len(node_categories),
                                batch_size=batch_size, shuffle=True, input_nodes=train_input_nodes)
        val_loader = HGTLoader(dataset, num_samples=[batch_size] * len(node_categories),
                                batch_size=batch_size, shuffle=False, input_nodes=val_input_nodes)
        test_loader = HGTLoader(dataset, num_samples=[batch_size] * len(node_categories),
                                batch_size=batch_size, shuffle=False, input_nodes=test_input_nodes)

        model = model.to(device)
        reset_parameters(model)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = F.cross_entropy

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_start = time.perf_counter()


        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, loss_fn, train_loader)
            val_losses.append(eval_loss(model, loss_fn, val_loader))
            accs.append(eval_acc(model, test_loader))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}, Duration: {duration_mean:.3f}')

    return loss_mean, acc_mean, acc_std


def k_fold(dataset, folds):
    skf = KFold(folds, shuffle=True, random_state=42)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset['MIDI'].x)), dataset['MIDI'].y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]


    for i in range(folds):
        train_mask = torch.ones(len(dataset['MIDI'].x), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loss_fn, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x_dict, data.edge_index_dict)
        loss = loss_fn(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x_dict, data.edge_index_dict)['MIDI'].argmax(dim=-1)
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loss_fn, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
        loss += loss_fn(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


@torch.no_grad()
def inference_run(model, loader, bf16):
    model.eval()
    for data in loader:
        data = data.to(device)
        if bf16:
            data.x = data.x.to(torch.bfloat16)
        model(data)

cross_validation_with_val_set(
    H_5,
    model=model_5,
    folds=10,
    epochs=1000,
    batch_size=32,
    lr=0.01,
    lr_decay_factor=0.5,
    lr_decay_step_size=50,
    weight_decay=0.0005,
    logger=None
)