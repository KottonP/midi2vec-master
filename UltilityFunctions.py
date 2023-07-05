import os
from typing import Optional
import itertools
import time

import scipy
import numpy as np
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import NAdam

import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, Linear



def tictoc(func):
    """Decorator to measure the time of our functions"""

    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter() - t1
        if t2 < 60:
            print(f"{func.__name__} took {t2:.2f} secs to run")
        else:
            print(f"{func.__name__} took {t2 / 60:.2f} mins to run")

        return result

    return wrapper


def complete_graph(input_path) -> nx.Graph:
    """
    Compile all edgelists in input_path directory into a nx.Graph instance.

    :param input_path: Directory containing the edgelists to compile.
    :return: Complete graph specified by the edgelists.
    """
    edgelists = [qf for qf in os.listdir(input_path)
                 if qf.endswith('.edgelist') and not qf.startswith('_')]
    g = None

    print('loading edgelists...')
    for eg in edgelists:
        print('- ' + eg)
        h = nx.read_edgelist(os.path.join(input_path, eg), nodetype=str, create_using=nx.DiGraph(), delimiter=' ')
        for edge in h.edges():
            h[edge[0]][edge[1]]['weight'] = 1

        g = h if g is None else nx.compose(g, h)

    g = g.to_undirected()

    print('Nodes: %d' % nx.number_of_nodes(g))
    print('Edges: %d' % nx.number_of_edges(g))
    return g


@tictoc
def node_cat_dict(nodes: pd.DataFrame) -> dict:
    """Compile all nodes in the nodes Dataframe in a dictionary."""
    note_groups = [n for n in nodes['name'] if n[0] == 'g' and n[1] in [str(i) for i in range(10)] + ['-']]

    # not_group_nodes = [n for n in nodes['name'] if n not in note_groups]
    not_group_nodes = list(set(nodes['name']) - set(note_groups))

    url = [n for n in not_group_nodes if n[:4] == 'http']
    program_nodes = []
    note_nodes = []
    for u in url:
        if "programs" in u:
            program_nodes.append(u)
        elif "notes" in u:
            note_nodes.append(u)
        else:
            print(u)

    # name_nodes = [n for n in not_group_nodes if '_-_' in n]
    # dur_nodes = [n for n in not_group_nodes if n[:3] == 'dur']
    # vel_nodes = [n for n in not_group_nodes if n[:3] == 'vel']
    # time_nodes = [n for n in not_group_nodes if n[:4] == 'time']
    # tempo_nodes = list(set(not_group_nodes) - set(dur_nodes).union(vel_nodes, time_nodes, name_nodes, url))

    not_group_url_nodes = list(set(not_group_nodes) - set(url))
    name_nodes = []
    dur_nodes = []
    vel_nodes = []
    time_nodes = []
    tempo_nodes = []
    for n in not_group_url_nodes:
        if '_-_' in n:
            name_nodes.append(n)
        elif n[:3] == 'dur':
            dur_nodes.append(n)
        elif n[:3] == 'vel':
            vel_nodes.append(n)
        elif n[:4] == 'time':
            time_nodes.append(n)
        else:
            tempo_nodes.append(n)

    node_categories = {"note_group": note_groups,
                       "pitch": note_nodes,
                       "program": program_nodes,
                       "MIDI": name_nodes,
                       "duration": dur_nodes,
                       "velocity": vel_nodes,
                       "time_sig": time_nodes,
                       "tempo": tempo_nodes
                       }
    return node_categories

@tictoc
def node_cat_dict_giant(nodes: pd.DataFrame) -> dict:
    """Compile all nodes in the nodes Dataframe in a dictionary; for the Giant-MIDI dataset."""
    note_groups = [n for n in nodes['name'] if n[0] == 'g' and n[1] in [str(i) for i in range(10)] + ['-']]

    # not_group_nodes = [n for n in nodes['name'] if n not in note_groups]
    not_group_nodes = list(set(nodes['name']) - set(note_groups))

    url = [n for n in not_group_nodes if n[:4] == 'http']
    program_nodes = []
    note_nodes = []
    for u in url:
        if "programs" in u:
            program_nodes.append(u)
        elif "notes" in u:
            note_nodes.append(u)
        else:
            print(u)


    not_group_url_nodes = list(set(not_group_nodes) - set(url))
    name_nodes = []
    dur_nodes = []
    vel_nodes = []
    time_nodes = []
    tempo_nodes = []
    for n in not_group_url_nodes:
        if n[0] == '-' :
            name_nodes.append(n)
        elif n[:3] == 'dur':
            dur_nodes.append(n)
        elif n[:3] == 'vel':
            vel_nodes.append(n)
        elif n[:4] == 'time':
            time_nodes.append(n)
        else:
            tempo_nodes.append(n)

    node_categories = {"note_group": note_groups,
                       "pitch": note_nodes,
                       "program": program_nodes,
                       "MIDI": name_nodes,
                       "duration": dur_nodes,
                       "velocity": vel_nodes,
                       "time_sig": time_nodes,
                       "tempo": tempo_nodes
                       }
    return node_categories

def reverse_edge(df: pd.DataFrame, row: int, inplace: bool = False) -> Optional[pd.DataFrame]:
    """Reverse the source and target of a single edge(row) in the edge dataframe."""
    if inplace:
        df.iloc[row]['source'], df.iloc[row]['target'] = df.iloc[row]['target'], df.iloc[row]['source']
        return None
    elif not inplace:
        tmp = df.copy()
        tmp.iloc[row]['source'], tmp.iloc[row]['target'] = tmp.iloc[row]['target'], tmp.iloc[row]['source']
        return tmp


def format_edge_name(source: str, target: str) -> str:
    """Combine source and target names in the correct form."""
    edge_name = ""

    if source == "MIDI":
        if target == "tempo":
            edge_name = source + "__has__" + target
        elif target == "time_sig":
            edge_name = source + "__in__" + target
        elif target == "program":
            edge_name = source + "__has__" + target
        elif target == "note_group":
            edge_name = source + "__has__" + target
    elif source == "note_group":
        if target == "velocity":
            edge_name = source + "__has__" + target
        elif target == "duration":
            edge_name = source + "__has__" + target
        elif target == "pitch":
            edge_name = source + "__contains__" + target
    else:
        edge_name = source + "__?__" + target
        print("Not known edge detected: " + edge_name)
        return edge_name

    return edge_name


def add_node_type(nodes_df: pd.DataFrame, node_cat: dict) -> pd.DataFrame:
    """
    Return input node Dataframe with a new column named "node_type", which specifies the type of the node.

    :param nodes_df: Dataframe containing the original node Dataframe (without type column).
    :param node_cat: Dictionary with keys: node names, values: nodes of specified category.
    :return: Node Dataframe with the new "node_type" column.
    """
    node_type = []
    augmented_nodes_df = nodes_df.copy()
    for i in range(len(nodes_df.index)):
        for key in node_cat.keys():
            if nodes_df.iloc[i]['name'] in node_cat[key]:
                node_type.append(key)
    augmented_nodes_df['node_type'] = node_type

    return augmented_nodes_df


def add_edge_type(edges_df: pd.DataFrame, node_cat: dict) -> pd.DataFrame:
    """
    Return input edge Dataframe with a new column named "edge_type", which specifies the type of the edge.

    :param edges_df: Dataframe containing the original edge Dataframe (without type column).
    :param node_cat: Dictionary with keys: node names, values: nodes of specified category.
    :return: Edge Dataframe with the new "edge_type" column.
    """
    edge_type = []

    edge_name_source = ""
    edge_name_target = ""

    augmented_edges_df = edges_df.copy()

    for i in range(len(edges_df.index)):
        for name in node_cat.keys():
            if edges_df.iloc[i]['source'] in node_cat[name]:
                edge_name_source = name
                break
        for name in node_cat.keys():
            if edges_df.iloc[i]['target'] in node_cat[name]:
                edge_name_target = name
                break

        if (edge_name_source not in ("MIDI", "note_group")) or (
                edge_name_source == "note_group" and edge_name_target == "MIDI"):
            reverse_edge(augmented_edges_df, row=i, inplace=True)
            edge_name_source, edge_name_target = edge_name_target, edge_name_source

        edge_name = format_edge_name(edge_name_source, edge_name_target)
        edge_type.append(edge_name)

    augmented_edges_df['edge_type'] = edge_type
    return augmented_edges_df


@tictoc
def add_types(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, node_cat: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute add_node_type and add_edge_type, and return a tuple of the new Dataframes."""
     
    augmented_nodes_df = add_node_type(nodes_df, node_cat)
    augmented_edges_df = add_edge_type(edges_df, node_cat)
    return augmented_nodes_df, augmented_edges_df


def flatten_lol(lol: list) -> list:
    """Flatten list of lists (lol)."""
    return list(itertools.chain(*list(lol)))


def midi_type(midi_name: str, classes: int) -> str:
    if classes == 5:
        return midi_name.split('_-_')[0]
    elif classes == 10:
        return midi_name.split('_-_')[1].split('-')[0]


def plot_graph(acc_lists: dict, save_dir: str = None):
    epochs = range(1, len(acc_lists['train']) + 1)

    fig = plt.figure(figsize=(15, 5))

    plt.plot(epochs, acc_lists['train'], 'bo', label='Training accuracy')
    plt.plot(epochs, acc_lists['val'], 'b', label='Validation accuracy')
    plt.plot(epochs, acc_lists['test'], 'r', label='Test accuracy')
    plt.title('Training, Validation and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    if save_dir:
        fig.savefig(save_dir)
    plt.show()


def plot_4graphs(loss_list: list, acc_lists: dict, index: int = 0, save_dir: str = None):
    epochs = range(1, len(acc_lists['train']) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))

    axs[0, 0].plot(epochs, loss_list)
    axs[0, 0].set_ylim(0, max([v for v in loss_list[index:]]))
    axs[0, 0].set_title('Loss')

    axs[0, 1].plot(epochs, acc_lists['train'], 'bo')
    axs[0, 1].set_title('Training accuracy')

    axs[1, 0].plot(epochs, acc_lists['val'], 'b')
    axs[1, 0].set_title('Validation accuracy')

    axs[1, 1].plot(epochs, acc_lists['test'], 'r')
    axs[1, 1].set_title('Test accuracy')

    if save_dir:
        fig.savefig(save_dir)

def cross_val(dataset, model, folds, epochs, lr, device, lr_decay_factor, lr_decay_step_size=None, weight_decay=0., exp_factor=.75):
    
    val_losses, accs, durations, models, training_epochs = [], [], [], [], []
    print(f'Transfering dataset to device {device}')
    dataset = dataset.to(device)
    print('Done')
    print('_' * 80)
    print('Starting Cross Validation')
    print('_' * 80)

    scheduler = None

    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):
        dataset['MIDI'].train_mask = train_idx
        dataset['MIDI'].val_mask = val_idx
        dataset['MIDI'].test_mask = test_idx

        
        # early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, late_start=late_start)


        print(f'Transfering Model to device{device}')
        model = model.to(device)
        
        print('Resetting parameters')
        reset_parameters_(model, dataset)
        

        optimizer = NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = F.cross_entropy

        print('Synchronizing GPU')
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        print('Starting Training')
        for epoch in range(1, epochs + 1):
            train_loss = train_(model, optimizer, scheduler=scheduler, loss_fn=loss_fn, dataset=dataset, epoch=epoch, epochs=epochs, exp_factor=exp_factor)
            val_losses.append(eval_loss_(model, loss_fn, dataset))
            accs.append(eval_acc_(model, dataset))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
            }

            print_dict(eval_info)

            
            if lr_decay_factor:
                if epoch  >= (epochs * exp_factor) and epoch % lr_decay_step_size == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_decay_factor * param_group['lr']

            # if early_stopper.early_stop(val_losses[-1], epoch=epoch):
            #     val_losses = val_losses + [float('inf')] * (epochs - epoch)
            #     accs = accs + [float('NaN')] * (epochs - epoch)
            #     print('Early Stopping at epoch', epoch)
            #     break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
        models.append(model)
        training_epochs.append(epoch)

    loss, acc, duration, training_epochs = torch.tensor(val_losses), torch.tensor(accs), torch.tensor(durations), torch.tensor(training_epochs, dtype=torch.float32)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    training_epochs_mean = training_epochs.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}, Average Duration per fold: {duration_mean/60:.3f} mins')

    return loss_mean, acc_mean, models

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

def train_(model, optimizer, loss_fn, dataset, epoch, epochs, exp_factor, scheduler=None):
    model.train()
    
    optimizer.zero_grad()
    out = model(dataset.x_dict, dataset.edge_index_dict)

    mask = dataset['MIDI'].train_mask
    loss = loss_fn(out['MIDI'][mask], dataset['MIDI'].y[mask])
    loss.backward()
    optimizer.step()
    if scheduler and epoch <= epochs*exp_factor:
        scheduler.step()
    
    return float(loss)


def eval_acc_(model, dataset):
    model.eval()

    
    mask = dataset['MIDI'].test_mask

    with torch.no_grad():
        pred = model(dataset.x_dict, dataset.edge_index_dict)['MIDI'].argmax(dim=-1)

    y_true = dataset['MIDI'].y[mask].to('cpu').numpy()
    y_pred = pred[mask].to('cpu').numpy()
    
    acc = np.sum(np.equal(y_true, y_pred)) / len(y_true)

    return acc


def eval_loss_(model, loss_fn, dataset):
    model.eval()

    
    mask = dataset['MIDI'].val_mask
    with torch.no_grad():
        out = model(dataset.x_dict, dataset.edge_index_dict)
    loss = loss_fn(out['MIDI'][mask], dataset['MIDI'].y[mask])
    return (float(loss))

def reset_parameters_(model, dataset):
    print('Lazy Initialization of Model')
    with torch.no_grad():  # Initialize lazy modules.
        out = model(dataset.x_dict, dataset.edge_index_dict)

    for c in model.children():
        if isinstance(c, torch.nn.ModuleList):
            for d in c:
                for v in d.values():
                    if isinstance(v, SAGEConv):
                        print('Resetting SAGEConv')
                        nn.init.kaiming_normal_(v.lin_l.weight, nonlinearity='relu')
        
    # for c in model.children():
    #     for v in c.values():
    #         if isinstance(v, SAGEConv):
    #             print('Resetting SAGEConv')
    #             nn.init.kaiming_normal_(v.lin_l.weight, nonlinearity='relu')
    #         elif isinstance(c, Linear):
    #             print('Resetting Linear')
    #             nn.init.xavier_normal_(v.weight)
    
def print_dict(d):
    """ Prints the key:values of a dictionary, all in the same line. """
    for k, v in d.items():
        print(k, ':', v, end=' | ')
    print('\n' + '-' * 80)

def save_models(models, path):
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f'{path}\\model_{i}.pt')
    print('Models saved at', path)

def load_models(path, folds, model):
    models = []
    print('Loading Models from', path)
    for i in range(folds):
        model.load_state_dict(torch.load(f'{path}\\model_{i}.pt'))
        models.append(model)
    return models

def voting(models, dataset, device):
    dataset = T.RandomNodeSplit(num_val=0.1, num_test=0.2)(dataset)
    mask = dataset['MIDI'].test_mask
    test_predictions = get_predictions(models, dataset, mask, device)

    # Combine predictions using voting
    stacked_predictions = torch.stack(test_predictions)
    voted_predictions, _ = torch.mode(stacked_predictions, dim=0)

    # Convert to numpy array
    return voted_predictions.to('cpu').numpy(), mask
    

def get_predictions(models, dataset, mask, device):
    predictions = []
    for model in models:
        model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(dataset.x_dict, dataset.edge_index_dict)['MIDI'].argmax(dim=-1)[mask]
            predictions.append(outputs)
    return predictions


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, late_start=0):
        self.late_start = late_start
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, epoch):
        if epoch > self.late_start:
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False


class Encoder:
    def __init__(self, str_list: list, n_labels: int = 0):
        """n_labels is the number of the target categories.
        We need it so there is no mix-up between the LabelEncoder and the current one."""
        self.mapping = {string: i + n_labels for i, string in enumerate(str_list)}
        self.rev_mapping = {v: k for k, v in self.mapping.items()}

    def get_encoding(self, s: str) -> int:
        return self.mapping[s]

    def decode_value(self, value: int) -> str:
        # return list(self.mapping.keys())[list(self.mapping.values()).index(value)]
        return self.rev_mapping[value]

    @tictoc
    def encode_nodes(self, df: pd.DataFrame) -> torch.Tensor:
        out = torch.zeros([len(df.index), 1], dtype=torch.float)

        for i in range(len(df.index)):
            out[i, 0] = self.mapping[df.iloc[i]['name']]
        return out

    @tictoc
    def encode_edges(self, df: pd.DataFrame) -> torch.Tensor:
        out = torch.zeros([len(df.index), 2], dtype=torch.int32)

        for i in range(len(df.index)):
            out[i, 0], out[i, 1] = self.mapping[df.iloc[i]['source']], self.mapping[df.iloc[i]['target']]
        return out

    def decode_df(self, ten: torch.Tensor) -> pd.DataFrame:
        out = pd.DataFrame(index=range(ten.size(0)), columns=range(ten.size(1)))
        if ten.size(1) == 2:
            for i in range(len(out.index)):
                out.iloc[i][0], out.iloc[i][1] = self.decode_value(ten[i][0].item()), self.decode_value(
                    ten[i][1].item())
        elif ten.size(1) == 3:
            for i in range(len(out.index)):
                out.iloc[i][0], out.iloc[i][1], out.iloc[i][2] = \
                    self.decode_value(ten[i][0].item()), self.decode_value(ten[i][1].item()), self.decode_value(
                        ten[i][2].item())

        return out


