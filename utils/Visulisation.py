from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from integrate import RoadNetwork3D
from City import City

class Visulisation():
    def __init__(self) -> None:
        pass
    def show_route(self, G:nx.DiGraph, road2D:dict, nodes_list:List[int], offset_pos_marker:float=50) -> None:
        fig, ax = plt.subplots(figsize=[10, 7])
        for way in road2D:
            if way['type'] == 'way' and 'tags' in way and 'highway' in way['tags'] and way['tags']['highway'] in ['residential','service','unclassified','primary', 'trunk', 'secondary', 'tertiary']:
                node_ids = way['nodes']
                #Get the coordinates of each point in each road segment: (lon, lat)
                coordinates = [(way['geometry'][node_index]['lon'], way['geometry'][node_index]['lat']) for node_index, node_id in enumerate(node_ids)]
                
                x_values, y_values = zip(*coordinates)
                ax.plot(x_values, y_values, 'b-', linewidth = 0.3)

        x = []
        y = []
        deep_red_rgb = "#8B0000"
        for node_index, node in enumerate(nodes_list):
            coordinate = G.nodes[node]['coordinate']
            x.append(coordinate[0])
            y.append(coordinate[1])
            if node_index == 0:
                ax.plot(G.nodes[node]['coordinate'][0], G.nodes[node]['coordinate'][1],'ro', markersize=1)
                ax.text(G.nodes[node]['coordinate'][0]+offset_pos_marker, G.nodes[node]['coordinate'][1]-offset_pos_marker, 'start', color=deep_red_rgb)
            elif node_index == len(nodes_list)-1:
                ax.plot(G.nodes[node]['coordinate'][0], G.nodes[node]['coordinate'][1],'ro', markersize=1)
                ax.text(G.nodes[node]['coordinate'][0]+offset_pos_marker, G.nodes[node]['coordinate'][1]-offset_pos_marker, 'end', color=deep_red_rgb)

        ax.plot(x, y, 'r-', linewidth = 0.5)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Way Elements')
        ax.grid(False)
        plt.show()


if __name__ == '__main__':
    path1 = '../data/accra_road.json'
    path2 = '../data/elevation/n05_w001_1arc_v3.tif' #This is the data of the elevation of Ghana
    accra_road = RoadNetwork3D(path1, path2)
    accra_road.integrate()
    accra_road.create_network()
    
    accra_zoo = City(5.625279092167783, -0.20306731748089998, lat_lon=True)
    kotoka_airport = City(813329.05, 620518.36, None)
    shortest_path = accra_road.get_shortest_path(kotoka_airport, accra_zoo, weight='time')


    accra_network = accra_road.get_network()
    accra_road3d = accra_road.get_3Droad()
    visual = Visulisation()
    visual.show_route(accra_network, accra_road3d, shortest_path)