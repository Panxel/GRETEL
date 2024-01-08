import networkx as nx
import numpy as np

from os import listdir
from os.path import isfile, join

from src.n_dataset.generators.base import Generator
from src.n_dataset.instances.graph import GraphInstance

# Python
from collections import defaultdict


class TreeTwitterGCN(Generator):

    def init(self):
            base_path = self.local_config['parameters']['data_dir']
            self._adj_file_path = join(base_path, 'TWITTER-Real-Graph-Partial_A.txt')
            self._graph_ind_file_path = join(base_path, 'TWITTER-Real-Graph-Partial_graph_indicator.txt')
            self._graph_labels_file_path = join(base_path, 'TWITTER-Real-Graph-Partial_graph_labels.txt')
            self.generate_dataset()

    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.read_adjacency_matrices()
    
    
    def read_adjacency_matrices(self):
        """
        Reads the dataset from the adjacency matrices
        """

        instance_id = 0
        label = 0

        # Initialize a dictionary to hold the graph nodes
        graph_nodes = defaultdict(list)

        # Open the file and read the lines
        with open('TWITTER-Real-Graph-Partial_graph_indicator.txt', 'r') as file:
            lines = file.readlines()

        # Remove newline characters and convert to integers
        lines = [int(line.strip()) for line in lines]

        # Populate the dictionary
        node_index = 1
        for graph_index in lines:
            graph_nodes[graph_index].append(node_index)
            node_index += 1

        print(graph_nodes)
        adj_matrix = []
        for index in graph_nodes.keys():
            adj_matrix.append(np.zero((len(index), len(index))))
            
        # Open the file and read the lines
        with open('TWITTER-Real-Graph-Partial_A.txt', 'r') as file:
            lines = file.readlines()
        

        # Alternatively, you can use list comprehension
        numbers = [(int(a), int(b)) for line in lines for num in line for a, b in [num.strip().split(',')]]
        
        # Populate the adjacency matrix with the edges