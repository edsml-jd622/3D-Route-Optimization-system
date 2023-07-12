import copy
import json
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import geopandas as gpd
import pyproj
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from City import City

class RoadNetwork3D():
    def __init__(self, path1:str, path2:str, ele_bound:list = [6,5,-1,0]) -> None:
        '''
        path1: str, the path of the 2D-road network data
        path2: str, the path of the elevation data
        '''
        with open(path1, 'r') as file:
            self.origin_2Ddata = json.load(file)
        image = Image.open(path2)
        #------------------------------------------------
        self.flag_3D = 0 #This flag is to show wether the Road data has been integrated into 3D road data
        self.flag_network = 0 # This flag is to show wether the Road data has been converted to network graph
        self.elevation = np.array(image)
        self.ele_bound = ele_bound
        self.road2D = copy.deepcopy(self.origin_2Ddata['elements'])
        self.road3D = copy.deepcopy(self.road2D)
        self.utm_converter = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')
        self.network = nx.Graph()

    def show_status(self) -> None:
        """
        Print the status of a RoadNetwork3D instande

        Parmeters
        ---------
        None
            This function does not have any parameters.

        Returns
        -------
        None
            This function does not return any value.
        """
        if self.flag_3D:
            print('The data has been integrated into 3D road data.')
            if self.flag_network:
                print('The data has been used to created a network graph, you can use it to find shortest path.')
            else:
                print('The network graph of the data has not been created, please use \'create_network()\' to create it.')
        else:
            print('The data is original 2D version.')


    def print_data(self, index:int=0) -> None:
        """
        Print the information of a specific road segment.

        Parameters
        ----------
        index : int, optional
            Index of the road segment to be displayed (default is 0).

        Returns
        -------
        None
            This function does not return any value.
        """
        formatted_data = json.dumps(self.road3D[index], indent=4)
        print(formatted_data)


    def draw_3Droad(self, way_list:list = ['primary']) -> None: 
        """
        Draw the 3D road network, user can choose which types of road to draw.
        Types include: 'primary','secondary', 'tertiary', 'residential', 'trunk', 'service', 'unclassified'

        Parameters
        ----------
        way_list : list, optional
            The list contains all the road types the user want to see(default is ['primary']).

        Returns
        -------
        None
            This function does not return any value.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for way in self.road3D:
            if (way['type'] == 'way' 
            and 'tags' in way 
            and 'highway' in way['tags'] 
            and way['tags']['highway'] in way_list):
                node_ids = way['nodes']
                coordinates = [(way['geometry'][node_index]['lon'], way['geometry'][node_index]['lat'], way['geometry'][node_index]['ele']) for node_index, node_id in enumerate(node_ids)]
                
                x_values, y_values, z_values = zip(*coordinates)
                
                cmap = plt.cm.coolwarm  # Choose a colormap (you can use any colormap you prefer)
                normalized_elevation = (z_values - min(z_values)) / (max(z_values) - min(z_values))  # Normalize the elevation values to [0, 1]
                colors = [cmap(value) for value in normalized_elevation]

                for i in range(len(x_values) - 1):
                    ax.plot([x_values[i], x_values[i+1]], [y_values[i], y_values[i+1]], [z_values[i], z_values[i+1]], color=colors[i], linewidth=0.5)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Elevation')
        ax.set_title('Way Elements (3D)')
        #ax.view_init(20, 45)
        plt.show()

    def draw_2Droad(self, way_list:List[str] = ['residential','service','unclassified','primary', 'trunk', 'secondary', 'tertiary']) -> None:
        """
        Draw the 2D road network, user can choose which types of road to draw.
        Types include: 'primary','secondary', 'tertiary', 'residential', 'trunk', 'service', 'unclassified'

        Parameters
        ----------
        way_list : list, optional
            The list contains all the road types the user want to see(default is ['residential','service','unclassified','primary', 'trunk', 'secondary', 'tertiary']).

        Returns
        -------
        None
            This function does not return any value.
        """
        for way in self.road2D:
            if way['type'] == 'way' and 'tags' in way and 'highway' in way['tags'] and way['tags']['highway'] in way_list:
                node_ids = way['nodes']
                #Get the coordinates of each point in each road segment: (lon, lat)
                coordinates = [(way['geometry'][node_index]['lon'], way['geometry'][node_index]['lat']) for node_index, node_id in enumerate(node_ids)]
                
                x_values, y_values = zip(*coordinates)
                if 'tags' in way and 'access' in way['tags'] and way['tags']['access']=='private':
                    plt.plot(x_values, y_values, 'r-', linewidth = 0.5)
                else:
                    plt.plot(x_values, y_values, 'b-', linewidth = 0.3)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Way Elements')
        plt.grid(True)
        plt.show()

    def integrate(self, target_list:List[str] = ['residential','service','unclassified','primary', 'trunk', 'secondary', 'tertiary']) -> None:
        """
        Integrate the 2D road network json data with elevation data into a 3D road network json data.
        It will automatically transfrom coordinates in lat/lon format into northing/easting format
        eg:
        {'lat': 5.5441366, 'lon': -0.2410432} --> {'lat': 613524.3283578429, 'lon': 805692.9580435926, 'ele': 29.284969791531932}

        Users can define which types of road segments to integrate, 
        Types in clude: ['residential','service','unclassified','primary', 'trunk', 'secondary', 'tertiary']

        Parameters
        ----------
        target_list : list, optional
            The list contains all road types the user want to integrate (default is ['residential','service','unclassified','primary', 'trunk', 'secondary', 'tertiary']).

        Returns
        -------
        None
            This function does not return any value.
        """
        if self.flag_3D:
            print('The data has been integrated, no need to integrate it again.')
        else:
            print('Integrating the 2D road network with elevation data...')
            min_i = 10000
            max_i = 0
            min_j = 10000
            max_j = 0
            pixel_number_lat = self.elevation.shape[0]
            pixel_number_lon = self.elevation.shape[1]

            utm_converter = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')
            left_bound, up_bound = utm_converter(self.ele_bound[2], self.ele_bound[0])
            right_bound, down_bound = utm_converter(self.ele_bound[3], self.ele_bound[1])

            pixel_length_lat = (up_bound - down_bound) / pixel_number_lat
            pixel_length_lon = (right_bound - left_bound) / pixel_number_lon

            for way in self.road3D:
                if way['type'] == 'way' and 'tags' in way and 'highway' in way['tags'] and way['tags']['highway'] in target_list:
                    node_ids = way['nodes']
                    #Get the coordinates of each point in each road segment: (lon, lat)
                    coordinates = [(way['geometry'][node_index]['lon'], way['geometry'][node_index]['lat']) for node_index, node_id in enumerate(node_ids)]
                    x_values, y_values = zip(*coordinates)
                    for node_index, node_id in enumerate(node_ids):
                        easting, northing = utm_converter(x_values[node_index], y_values[node_index])
                        #find the index of the elevation grid for this point
                        i = int((easting - left_bound) // pixel_length_lon)
                        j = int((up_bound - northing) // pixel_length_lat)
                        
                        min_i = min(min_i, i)
                        min_j = min(min_j, j)
                        max_i = max(max_i, i)
                        max_j = max(max_j, j)

                        #get the coordinate of the center of the grid
                        x_grid = i*pixel_length_lon + pixel_length_lon/2 + left_bound
                        y_grid = up_bound - (j*pixel_length_lat + pixel_length_lat/2)

                        #Judge which part of the grid does the point exist on
                        if easting > x_grid:
                            delta_i = 1
                        else:
                            delta_i = -1
                        if northing > y_grid:
                            delta_j = -1
                        else:
                            delta_j = 1

                        #calculate the distance of the point with the four grids surounding it.
                        distance_A = np.sqrt((easting - ((i+delta_i)*pixel_length_lon + pixel_length_lon/2 + left_bound))**2 + (northing - (up_bound - ((j+delta_j)*pixel_length_lat + pixel_length_lat/2)))**2)
                        distance_B = np.sqrt((easting - ((i)*pixel_length_lon + pixel_length_lon/2 + left_bound))**2 + (northing - (up_bound - ((j+delta_j)*pixel_length_lat + pixel_length_lat/2)))**2)
                        distance_C = np.sqrt((easting - ((i+delta_i)*pixel_length_lon + pixel_length_lon/2 + left_bound))**2 + (northing - (up_bound - ((j)*pixel_length_lat + pixel_length_lat/2)))**2)
                        distance_D = np.sqrt((easting-x_grid)**2 + (northing-y_grid)**2)
                        weight_total = 1/distance_A + 1/distance_B + 1/distance_C + 1/distance_D


                        elevation_cur = (
                                        1/distance_A/weight_total * self.elevation[j+delta_j][i+delta_i] +
                                        1/distance_B/weight_total * self.elevation[j+delta_j][i] +
                                        1/distance_C/weight_total * self.elevation[j][i+delta_i] +
                                        1/distance_D/weight_total * self.elevation[j][i]
                                        )
                        way['geometry'][node_index]['lon'] = easting
                        way['geometry'][node_index]['lat'] = northing
                        way['geometry'][node_index]['ele'] = elevation_cur
            self.flag_3D = 1
            print('Integration finished')

    def get_closest_point(self,city:City) -> int:
        """
        Get the closest node's id of the given city in the network.

        Parameters
        ----------
        city: City
            Each City is a location, the City class contain the coordinate(x,y,z) and unique id of this location.

        Returns
        -------
        target:int
            The node id in the self.network, the node is the closest node of the given city.
        """
        min_distance = float('inf')
        coordinate = city.get_coordinates()
        target = None
        for (p,c) in self.network.nodes(data=True):
            if ((coordinate[0]-c['coordinate'][0])**2 + (coordinate[1]-c['coordinate'][1])**2) < min_distance:
                min_distance = (coordinate[0]-c['coordinate'][0])**2 + (coordinate[1]-c['coordinate'][1])**2
                target = p
        return target

    def create_network(self) -> None:
        """
        Create the bidirectional graph for the road network. Each node in the graph is a point in the self.road3D, each edge in the road is a line between a pair of points in self.road3D.

        Parameters
        ----------
        None
            This function does not have any parameters.

        Returns
        -------
        None
            This function does not return any value.
        """
        if self.flag_network:
            print('The network has been created, no need to do it again.')
        else:
            for way in self.road3D:
                if way['type'] == 'way' and 'tags' in way and 'highway' in way['tags'] and way['tags']['highway'] in ['residential','service','unclassified','primary', 'trunk', 'secondary', 'tertiary']:
                    node_ids = way['nodes']
                    #Get the coordinates of each point in each road segment: (lon, lat)
                    coordinates = [(way['geometry'][node_index]['lon'], way['geometry'][node_index]['lat'], way['geometry'][node_index]['ele']) for node_index, node_id in enumerate(node_ids)]
                    x_values, y_values, z_values = zip(*coordinates)
                    for node_index, node_id in enumerate(node_ids):
                        if node_index == len(node_ids)-1:
                            nx.set_node_attributes(self.network, {node_id:(x_values[node_index], y_values[node_index], z_values[node_index])}, 'coordinate')
                            continue
                        else:
                            self.network.add_edges_from([(node_id, node_ids[node_index+1])], distance = np.sqrt((x_values[node_index]-x_values[node_index+1])**2 + (y_values[node_index]-y_values[node_index+1])**2 + (z_values[node_index]-z_values[node_index+1])**2)) 
                            nx.set_node_attributes(self.network, {node_id:(x_values[node_index], y_values[node_index], z_values[node_index])}, 'coordinate')
            self.flag_network = 1

    def get_shortest_path(self, city1:City, city2:City, weight:str = 'distance') -> List[int]:
        """
        Get the list of the nodes which form the shortest path between the two cities given.
        The algorithm to calculate the shortest path is Dijkstra.

        Parameters
        ----------
        city1: City
            The first location used to calculate the shortest path.
        city2: City
            The second location used to calculate the shortest path.
        weight: str
            The weight used to calculate the shortest path(default is 'distance').

        Returns
        -------
        List[int]:
            The list contain all the node's id on the shortest path.
        """
        city1_closest = self.get_closest_point(city1)
        city2_closest = self.get_closest_point(city2)
        path = nx.shortest_path(self.network, source=city1_closest, target=city2_closest, weight=weight)
        return path
    
    def get_shortest_path_length(self, city1, city2, weight:str = 'distance') -> float:
        """
        Get the shortest path's length between the two cities given.

        Parameters
        ----------
        city1: City
            The first location used to calculate the shortest path length.
        city2: City
            The second location used to calculate the shortest path length.
        weight: str
            The weight used to calculate the shortest path(default is distance).

        Returns
        -------
        float:
            The distance of the shortest path.
        """
        city1_closest = self.get_closest_point(city1)
        city2_closest = self.get_closest_point(city2)
        path_length = nx.shortest_path_length(self.network, source=city1_closest, target=city2_closest, weight=weight)
        return path_length
    
    def weight_matrix(self, city_list:List[City], weight:str = 'distance') -> np.ndarray:
        """
        Get the weight matrix of a list of Cities, the weight can be assgined by users.

        Parameters
        ----------
        city_list: List[City]
            The list of Cities used to calculate weight matrix.

        Returns
        -------
        np.ndarray:
            The matrix contain all weights of any pairs of Cities in the city_list.
        """       
        n = len(city_list)
        
        ans = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                else:
                    ans[i, j] = self.get_shortest_path_length(city_list[i], city_list[j], weight)
        return ans

if __name__ == '__main__':
    path1 = '../data/accra_road.json'
    path2 = '../data/elevation/n05_w001_1arc_v3.tif' #This is the data of the elevation of Ghana
    accra_road = RoadNetwork3D(path1, path2)
    accra_road.integrate()
    accra_road.create_network()
    #accra_road.print_data(1000)
    #accra_road.draw_2Droad()

    kotoka_airport = City(813329.05, 620518.36, None)
    uni_ghana = City(811795.639, 625324.503, None)
    random_position_1 = City(814795.639, 635324.503, None)
    random_position_2 = City(813929.05, 620018.36, None)

    print(accra_road.get_closest_point(kotoka_airport))
    print(accra_road.get_shortest_path(kotoka_airport, uni_ghana))
    print(accra_road.get_shortest_path_length(kotoka_airport, uni_ghana))
    print(accra_road.weight_matrix([kotoka_airport, uni_ghana, random_position_1, random_position_2]))