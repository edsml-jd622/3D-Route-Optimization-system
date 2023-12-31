'''
Name: Jinsong Dong
GitHub Username: edsml-jd622
'''
import copy
import json
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyproj
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from collections import Counter
from pprint import pprint
from tqdm import tqdm

from .City import City
from .CostFunctions import CostFunctions

class RoadNetwork3D():
    def __init__(self, path1:str, path2:str, ele_bound:List[int] = [6,5,-1,0], zone:int = 30) -> None:
        '''
        path1: str, the path of the 2D-road network data
        path2: str, the path of the elevation data
        ele_bound: List[int], the coordinate bound for the elevation data: [latitude(north), latitude(south), longitude(west), longitude(east)]
        zone: int, the projection zone for the coordinate transformation from WGS84 to utm. The default zone is 30.
        '''
        try:
            with open(path1, 'r') as file:
                self.origin_2Ddata = json.load(file)
                image = Image.open(path2)
        except FileNotFoundError:
            print("File not found. Please check if the file path is correct.")
        except IOError:
            print("IO error occurred while reading the file.")
        except Exception as e:
            print("An exception of another type occurred:", str(e))
        else:
            #------------------------------------------------
            self.flag_3D = 0 #This flag is to show wether the Road data has been integrated into 3D road data
            self.flag_network = 0 # This flag is to show wether the Road data has been converted to network graph
            self.elevation = np.array(image)
            self.ele_bound = ele_bound
            self.road2D = copy.deepcopy(self.origin_2Ddata['elements'])
            self.road3D = copy.deepcopy(self.road2D)
            self.utm_converter = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
            self.network = nx.DiGraph()
            self.weight_tool = CostFunctions()
            self.road_type = ['residential', 'unclassified', 'service', 'primary', 'primary_link', 'trunk','trunk_link', 'secondary', 'secondary_link', 'tertiary','tertiary_link']
            self.all_tags = {} #All possible tags and its frequency of appearance in the road dataset.

            #Add all possible tags as keys and their frequency as values into self.all_tags
            for feature in tqdm(self.road2D, desc='Reading data: ', unit='item'):
                if 'tags' in feature:
                    for key, value in feature['tags'].items():
                        if key in ['highway', 'oneway', 'maxspeed','junction','access', 'lanes']:
                            self.all_tags.setdefault(key, Counter())[value] += 1
    
    def add_road_type(self, names: Union[str, List[str]] = None) -> None:
        """
        Add the specified road type(s) from self.road_type.

        This function primarily affects the variable self.road_type, which will be used in the 'integrate' and 'create_network' functions.

        Parameters
        ----------
        names : str or List[str], optional
            The road type name(s) to be added. It can be a single name or a list of names.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        - If no names are provided, a message will be printed to specify that a name or a list of names should be provided.
        - If a single name (str) is provided, it will be checked against the valid road types. If it is valid, it will be added to self.road_type. Otherwise, an error message will be printed.
        - If a list of names is provided, each name will be checked against the valid road types. The valid names will be added to self.road_type, and the invalid names will be collected separately. Messages will be printed to indicate which names were added and which were not.
        - After the additions, a list of the valid road types will be printed for reference.
        """
        if not names:
            print('You should specify a name or a list of names to be added.')
            print('--------------------------------------------')
            print('The valid road types can be added are as follows:\n')
            pprint(list(self.all_tags['highway'].keys()))

        if isinstance(names, str):
            if names in self.all_tags['highway']:
                self.road_type.append(names)
            else:
                print('\''+names+'\''+ ' is not a valid road type.')
                print('--------------------------------------------')
                print('The valid road types can be added are as follows:\n')
                pprint(list(self.all_tags['highway'].keys()))

        if isinstance(names, List):
            unvalid_names = []
            valid_names = []
            for each_name in names:
                if each_name in self.all_tags['highway']:
                    valid_names.append(each_name)
                    self.road_type.append(each_name)
                else:
                    unvalid_names.append(each_name)
            
            if valid_names:
                print('The names below are valid and successfully added into self.road_type:')
                pprint(valid_names)
            if unvalid_names:
                print('The names below are unvalid and not added into self.road_type:')
                pprint(unvalid_names)
                print('--------------------------------------------')
                print('The valid road types can be added are as follows:\n')
                pprint(list(self.all_tags['highway'].keys()))

        self.road_type = list(set(self.road_type)) #remove duplicate elements

    def delete_road_type(self, names: Union[str, List[str]] = None) -> None:
        """
        Delete the specified road type(s) from self.road_type.

        Parameters
        ----------
        names : str or List[str], optional
            The road type name(s) to be deleted. It can be a single name or a list of names.

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        - If no names are provided, a message will be printed to specify that a name or a list of names should be provided. The current self.road_type will be printed for reference.
        - If a single name (str) is provided, it will be checked against the existing road types in self.road_type. If it is found, it will be removed from self.road_type. Otherwise, an error message will be printed.
        - If a list of names is provided, each name will be checked against the existing road types in self.road_type. The valid names will be removed from self.road_type, and the invalid names will be collected separately. Messages will be printed to indicate which names were removed and which were not.
        - After the deletions, a list of the remaining road types in self.road_type will be printed for reference.
        - The final step removes any duplicate elements from self.road_type.

        """
        if not names:
            print('You should specify a name or a list of names to be deleted.')
            print('--------------------------------------------')
            print('The valid road types can be added are as follows:\n')
            pprint(self.road_type)

        if isinstance(names, str):
            if names in self.road_type:
                self.road_type.remove(names)
            else:
                print('\''+names+'\''+ ' is not a valid road type.')
                print('--------------------------------------------')
                print('The valid road types can be deleted are as follows:\n')
                pprint(self.road_type)

        if isinstance(names, List):
            unvalid_names = []
            valid_names = []
            for each_name in names:
                if each_name in self.road_type:
                    valid_names.append(each_name)
                    self.road_type.remove(each_name)
                else:
                    unvalid_names.append(each_name)
            
            if valid_names:
                print('The names below are valid and successfully deleted from self.road_type:')
                pprint(valid_names)
            if unvalid_names:
                print('The names below are unvalid:')
                pprint(unvalid_names)
                print('--------------------------------------------')
                print('The valid road types can be deleted from self.road_type are as follows:\n')
                pprint(self.road_type)

        self.road_type = list(set(self.road_type)) #remove duplicate elements

    def print_road_type(self) -> None:
        """
        Print the road types to be integrated.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        - This function prints the road types stored in the `self.road_type` list.
        - The road types are printed in a well-formatted and readable manner using the `pprint` function from the `pprint` module.
        - The printed output includes a header to indicate that it displays the road types to be integrated.
        - The printed output is enclosed with dashed lines for visual separation.

        """
        print('\n---------------------------------------------')
        print('The road types to be integrated are as follows:')
        pprint(self.road_type)
        print('-----------------------------------------------\n')
    
    def show_tags(self) -> None:
        """
        Display the distribution of tags.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This function does not return any value.

        Notes
        -----
        - This function uses matplotlib to create a figure with subplots to display the distribution of different tag categories.
        - The figure has a 2x2 grid of subplots, with each subplot representing a tag category.
        - For each tag category, the function retrieves the corresponding tag data from `self.all_tags`, sorts it based on the count in descending order, and extracts the labels and values.
        - The data is plotted using a bar chart in each subplot, with the labels on the x-axis and the counts on the y-axis.
        - The title, x-label, y-label, and x-tick labels are set for each subplot based on the tag category.
        - The subplots are adjusted for better spacing using `tight_layout`.
        - The plot is displayed using `plt.show()`.
        """
        fig, axes = plt.subplots(2, 2, figsize=[5,5])
        for i, tag_catagory in enumerate(['highway', 'access', 'oneway', 'maxspeed']):
            tag = self.all_tags[tag_catagory]
            tag = sorted(tag.items(), key=lambda x: x[1], reverse = True)
            labels = [item[0] for item in tag]
            values = [item[1] for item in tag]

            axes[i // 2, i % 2].bar(labels, values)
            axes[i // 2, i % 2].set_title('Distribution of ' + tag_catagory)
            axes[i // 2, i % 2].set_xlabel(tag_catagory)
            axes[i // 2, i % 2].set_ylabel('Count')
            axes[i // 2, i % 2].set_xticks(range(len(labels)))
            axes[i // 2, i % 2].set_xticklabels(labels, rotation=90)
        plt.tight_layout()
        plt.show()

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

    def get_elevation(self) -> np.ndarray:
        """
        Get the elevation data from 'self.elevation'.

        Parameters
        ----------
        None
         
        Returns
        -------
        self.elevation: np.ndarray
            'self.elevation' is the raster elevation data, which is in 2D np.ndarray format.
        """
        return copy.deepcopy(self.elevation)

    def get_elebound(self) -> List[int]:
        """
        Get the boundary of elevation data from 'self.ele_bound'.

        Parameters
        ----------
        None
         
        Returns
        -------
        self.ele_bound: List[int]
            'self.ele_bound' is the list of boundary for elevation data, which is in List[int] format, [lat(N), lat(S), lon(W), lon(E)].
        """
        return copy.deepcopy(self.ele_bound)

    def get_3Droad(self) -> dict:
        """
        Get the 3D road json data 'self.road3D'.

        Parameters
        ----------
        None
         
        Returns
        -------
        self.road3D: dict
            'self.road3D' is the integrated json data, which is in dictionary format.
        """
        return copy.deepcopy(self.road3D)

    def get_roadtype(self) -> List[str]:
        """
        Get the roadtype that need to be manipulated from 'self.road_type.

        Parameters
        ----------
        None
         
        Returns
        -------
        self.road_type: List[str]
            'self.road_type' is the list of str represent road types need to be manipulated(integrate, create network).
        """
        return copy.deepcopy(self.road_type)

    def get_2Droad(self) -> dict:
        """
        Get the 2D road json data 'self.road2D'.

        Parameters
        ----------
        None
         
        Returns
        -------
        self.road2D: dict
            'self.road2D' is the json data that has not been integrated, which is in dictionary format.
        """
        return self.road2D 

    def integrate(self) -> None:
        """
        Integrate the 2D road network json data with elevation data into a 3D road network json data.
        It will automatically transfrom coordinates in lat/lon format into northing/easting format
        eg:
        {'lat': 5.5441366, 'lon': -0.2410432} --> {'lat': 613524.3283578429, 'lon': 805692.9580435926, 'ele': 29.284969791531932}

        Parameters
        ----------
        None
            This function does not have any parameters

        Returns
        -------
        None
            This function does not return any value.
        """
        try:
            if self.flag_3D:
                print('The data has been integrated, no need to integrate it again.')
            else:
                pixel_number_lat = self.elevation.shape[0]
                pixel_number_lon = self.elevation.shape[1]

                utm_converter = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')
                if self.ele_bound[0] <= 180:
                    left_bound, up_bound = utm_converter(self.ele_bound[2], self.ele_bound[0])
                    right_bound, down_bound = utm_converter(self.ele_bound[3], self.ele_bound[1])
                else:
                    left_bound, up_bound, right_bound, down_bound = self.ele_bound[2], self.ele_bound[0], self.ele_bound[3], self.ele_bound[1]

                pixel_length_lat = (up_bound - down_bound) / pixel_number_lat # The length of each pixel in self.elevation along the latitude direction
                pixel_length_lon = (right_bound - left_bound) / pixel_number_lon # The length of each pixel in self.elevation along the longitude direction

                for way in tqdm(self.road3D, desc='Integrating data', unit='item'):
                    if way['type'] == 'way' and 'tags' in way and 'access' in way['tags'] and way['tags']['access'] in ['private', 'customers', 'no', 'students', 'school', 'delivery']:
                        # Skip the ways that has no access
                        continue
                    if way['type'] == 'way' and 'tags' in way and 'highway' in way['tags'] and way['tags']['highway'] in self.road_type:
                        # Ensure the roads to be integrated is within the elevation data bound.
                        if self.ele_bound[0]<=180 and (way['bounds']['maxlat'] > self.ele_bound[0]
                            or way['bounds']['minlat'] < self.ele_bound[1]
                            or way['bounds']['maxlon'] > self.ele_bound[3]
                            or way['bounds']['minlon'] < self.ele_bound[2]):
                            raise ValueError("The road data beyond the bound of elevation data, please reduce the selected road types.")

                        # Add speed attribute in each road segment
                        if 'maxspeed' in way['tags']:
                            way['tags']['speed'] = float(''.join(filter(str.isdigit, way['tags']['maxspeed']))) #Extract the digit part of maxspeed, eg:"30 mph" will become 30(float)
                        else:
                            if way['tags']['highway'] in ['primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link']:
                                way['tags']['speed'] = 50
                            elif way['tags']['highway'] in ['motorway']:
                                way['tags']['speed'] = 100
                            elif way['tags']['highway'] in ['residential']:
                                way['tags']['speed'] = 30
                            else:
                                way['tags']['speed'] = 20

                        node_ids = way['nodes']
                        #Get the coordinates of each point in each road segment: (lon, lat)
                        coordinates = [(way['geometry'][node_index]['lon'], way['geometry'][node_index]['lat']) for node_index, node_id in enumerate(node_ids)]
                        x_values, y_values = zip(*coordinates)
                        
                        #processing each point in every road segment
                        for node_index, node_id in enumerate(node_ids):
                            if x_values[node_index] <= 180: #check what's the format of the coordinates
                                easting, northing = utm_converter(x_values[node_index], y_values[node_index])
                            else:
                                easting, northing = x_values[node_index], y_values[node_index]
                            #find the index of the elevation grid for this point
                            i = int((easting - left_bound) // pixel_length_lon)
                            j = int((up_bound - northing) // pixel_length_lat)
                            
                            if i > 0 and j > 0 and i < pixel_length_lon-1 and j < pixel_length_lat-1: # make sure i and j is not at the edge of the elevation data.
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
                            else:
                                elevation_cur = self.elevation[j][i]
                            way['geometry'][node_index]['lon'] = easting
                            way['geometry'][node_index]['lat'] = northing
                            way['geometry'][node_index]['ele'] = elevation_cur
                self.flag_3D = 1
        except ValueError:
            print('Something went wrong when integrating road data.')

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
        target_coor = None
        for (p,c) in self.network.nodes(data=True):
            if ((coordinate[0]-c['coordinate'][0])**2 + (coordinate[1]-c['coordinate'][1])**2) < min_distance:
                min_distance = (coordinate[0]-c['coordinate'][0])**2 + (coordinate[1]-c['coordinate'][1])**2
                target = p
                target_coor = c
        return target, target_coor

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
            for way in tqdm(self.road3D, desc='Creating network: ', unit='item'):
                if way['type'] == 'way' and 'tags' in way and 'access' in way['tags'] and way['tags']['access'] in ['private', 'customers', 'no', 'students', 'school', 'delivery']:
                    # Skip the ways that has no access
                    continue
                if way['type'] == 'way' and 'tags' in way and 'highway' in way['tags'] and way['tags']['highway'] in self.road_type:
                    node_ids = way['nodes']
                    #Get the coordinates of each point in each road segment: (lon, lat)
                    coordinates = [(way['geometry'][node_index]['lon'], way['geometry'][node_index]['lat'], way['geometry'][node_index]['ele']) for node_index, node_id in enumerate(node_ids)]
                    x_values, y_values, z_values = zip(*coordinates)

                    if 'tags' in way and (('oneway' in way['tags'] and way['tags']['oneway'] == 'yes') or ('lanes' in way['tags'] and way['tags']['lanes'] == '1')):
                        #create edge for one-way road
                        for node_index, node_id in enumerate(node_ids):
                            if node_index == len(node_ids)-1:
                                nx.set_node_attributes(self.network, {node_id:(x_values[node_index], y_values[node_index], z_values[node_index])}, 'coordinate')
                                continue
                            else:
                                self.network.add_edges_from([(node_id, node_ids[node_index+1])], 
                                                distance = self.weight_tool.distance(coordinates[node_index], coordinates[node_index+1]),
                                                slope = self.weight_tool.slope(coordinates[node_index], coordinates[node_index+1]),
                                                time = self.weight_tool.travel_time(coordinates[node_index], coordinates[node_index+1], way['tags']['speed'])
                                                )
                                nx.set_node_attributes(self.network, {node_id:(x_values[node_index], y_values[node_index], z_values[node_index])}, 'coordinate')
                    else:
                        #create edge for bidirectional road, by creating two edges in different direction for each road segment .
                        for node_index, node_id in enumerate(node_ids):
                            if node_index == len(node_ids)-1:
                                nx.set_node_attributes(self.network, {node_id:(x_values[node_index], y_values[node_index], z_values[node_index])}, 'coordinate')
                                continue
                            else:
                                self.network.add_edges_from([(node_id, node_ids[node_index+1])], 
                                                distance = self.weight_tool.distance(coordinates[node_index], coordinates[node_index+1]),
                                                slope = self.weight_tool.slope(coordinates[node_index], coordinates[node_index+1]),
                                                time = self.weight_tool.travel_time(coordinates[node_index], coordinates[node_index+1], way['tags']['speed'])
                                ) 
                                nx.set_node_attributes(self.network, {node_id:(x_values[node_index], y_values[node_index], z_values[node_index])}, 'coordinate')
                        
                        #The opposite direction of the same road segment. Reverse the points list of each road segment.
                        node_ids = list(reversed(node_ids))
                        coordinates = list(reversed(coordinates))
                        x_values, y_values, z_values = zip(*coordinates)
                        for node_index, node_id in enumerate(node_ids):
                            if node_index == len(node_ids)-1:
                                nx.set_node_attributes(self.network, {node_id:(x_values[node_index], y_values[node_index], z_values[node_index])}, 'coordinate')
                                continue
                            else:
                                self.network.add_edges_from([(node_id, node_ids[node_index+1])], 
                                                distance = self.weight_tool.distance(coordinates[node_index], coordinates[node_index+1]),
                                                slope = self.weight_tool.slope(coordinates[node_index], coordinates[node_index+1]),
                                                time = self.weight_tool.travel_time(coordinates[node_index], coordinates[node_index+1], way['tags']['speed'])
                                )
                                nx.set_node_attributes(self.network, {node_id:(x_values[node_index], y_values[node_index], z_values[node_index])}, 'coordinate')
            self.flag_network = 1

    def get_network(self) -> nx.DiGraph:
        '''
        Give the user access to self.network

        Parameters
        ----------
        None

        Return
        ------
        self.network: nx.Digraph
            The network of the area, in format of nx.Digraph.
        '''
        return copy.deepcopy(self.network)

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
        path: List[int]
            The list contain all the node's id on the shortest path.
        """
        city1_closest, _ = self.get_closest_point(city1)
        city2_closest, _ = self.get_closest_point(city2)
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
        path_length: float
            The distance of the shortest path.
        """
        city1_closest, _ = self.get_closest_point(city1)
        city2_closest, _ = self.get_closest_point(city2)
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
        ans: np.ndarray
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

    kotoka_airport = City(5.605522862563998, -0.17187326099346129, None, lon_lat=False)
    uni_ghana = City(5.650618146052781, -0.18703194651322047, None, lon_lat=False)
    accra_zoo = City(5.625370802046447, -0.20300362767245603, None, lon_lat=False)
    national_museum = City(5.560739525028722, -0.20650512945516059, None, lon_lat=False)

    print(accra_road.get_shortest_path_length(kotoka_airport, uni_ghana, weight='distance'))
    print(accra_road.get_shortest_path_length(kotoka_airport, accra_zoo, weight='distance'))
    print(accra_road.get_shortest_path_length(kotoka_airport, national_museum, weight='distance'))
    print(accra_road.get_shortest_path_length(accra_zoo, uni_ghana, weight='distance'))
    print(accra_road.get_shortest_path_length(accra_zoo, national_museum, weight='distance'))

    # accra_road.draw_2Droad()
    # print(accra_road.get_shortest_path_length(kotoka_airport, uni_ghana))
    # print(accra_road.weight_matrix([kotoka_airport, uni_ghana, random_position_1, random_position_2]))