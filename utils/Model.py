import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from integrate import RoadNetwork3D
from City import City
import numpy as np
from Generate import generate_random_CityWeight, tsp_bruteforce

class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphConvolutionalNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


path1 = '../data/accra_road.json'
path2 = '../data/elevation/n05_w001_1arc_v3.tif' #This is the data of the elevation of Ghana
accra_road = RoadNetwork3D(path1, path2)
accra_road.integrate()
accra_road.create_network()

bound_list = [6.2,5.3,-0.7,-0.3]
num_city = 10
cities_data = []
weights_data = []
labels_data = []
i=100



while i != 0:
    try:
        my_cities, weight = generate_random_CityWeight(accra_road, bound_list, num_city)
        best_path, min_distance = tsp_bruteforce(weight)
    except:
        print('failed')
        continue
    print('rest turn = ', i)
    cities_data.append(my_cities)
    weights_data.append(weight)
    labels_data.append(best_path)
    i -= 1
print(cities_data)
print(weights_data)
print(labels_data)
# np.save('cities_data_10city_1.npy', np.array(cities_data))
# np.save('weights_data_10city_1.npy', np.array(weights_data))
# np.save('labels_data_10city_1.npy', np.array(labels_data))