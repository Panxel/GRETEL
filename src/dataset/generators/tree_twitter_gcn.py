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
        with open('graph_ind.txt', 'r') as file:
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
            adj_matrix.append(np.zeros((len(graph_nodes[index]), len(graph_nodes[index]))))
            print("HERE",index)
            print(adj_matrix[index-1])
            

        # Open the file and read the lines
        with open('a.txt', 'r') as file:
            lines = file.readlines()

        # Alternatively, you can use list comprehension
        # numbers = [(int(a), int(b)) for line in lines for num in line for a, b in [num.strip().split(',')]]

        # Remove newline characters and convert to integers
        lines = [line.strip() for line in lines]

        # Using a list comprehension to convert each string into a tuple
        tuple_list = [tuple(map(int, pair.split(','))) for pair in lines]

        # Using a list comprehension to find the key(s) for the given value

        for tuple in tuple_list:
            desired_value = tuple[0]
            keys_for_value = [key for key, value in graph_nodes.items() if desired_value in value]
            current_graph_id_list = graph_nodes[keys_for_value[0]]
            adj_matrix[keys_for_value[0]-1][current_graph_id_list.index(tuple[0])][current_graph_id_list.index(tuple[1])] = 1
            # print("NEXT")
            # print(adj_matrix[keys_for_value[0]-1])
            

        # Open the file and read the lines
        with open('graph_lab.txt', 'r') as file:
            lines = file.readlines()

        # Remove newline characters and convert to integers
        label_list = [line.strip() for line in lines]

        print(lines)

        for index in graph_nodes.keys():
            self.dataset.instances.append(GraphInstance(id =index , label = label_list[index-1], data = adj_matrix[index-1]))