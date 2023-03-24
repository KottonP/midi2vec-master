import os
from typing import Optional
import itertools
import time

import scipy
import numpy as np
import pandas as pd

import networkx as nx
import matplotlib

import torch
from torch_geometric.data import HeteroData


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


@tictoc
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


@tictoc
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
def add_types(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, node_cat: dict) -> (pd.DataFrame, pd.DataFrame):
    """Execute add_node_type and add_edge_type, and return a tuple of the new Dataframes."""
    return add_node_type(nodes_df, node_cat), add_edge_type(edges_df, node_cat)


def flatten_lol(lol: list) -> list:
    """Flatten list of lists (lol)."""
    return list(itertools.chain(*list(lol)))


def midi_type(midi_name: str) -> str:
    return midi_name.split('_-_')[0]


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
