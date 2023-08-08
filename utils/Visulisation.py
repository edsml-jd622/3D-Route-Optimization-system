from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from Road_Network import RoadNetwork3D
from matplotlib.gridspec import GridSpec
import numpy as np
import pyproj
from matplotlib.cm import ScalarMappable

from City import City

class Visulisation():
    def __init__(self, network:RoadNetwork3D) -> None:
        self.G = network.get_network()
        self.road2D = network.get_2Droad()
        self.road3D = network.get_3Droad()
        self.elevation = network.get_elevation()
        self.roadtype = network.get_roadtype()
        max_lat = -100
        max_lon = -100
        min_lat = 100
        min_lon = 100
        for way in self.road2D:
            if way['type'] == 'way' and 'tags' in way and 'highway' in way['tags'] and way['tags']['highway'] in self.roadtype:
                min_lat = min(way['bounds']['minlat'], min_lat)
                min_lon = min(way['bounds']['minlon'], min_lon)
                max_lat = max(way['bounds']['maxlat'], max_lat)
                max_lon = max(way['bounds']['maxlon'], max_lon)
        self.road_bound = [max_lat, min_lat, min_lon, max_lon]
        self.ele_bound = network.get_elebound()

    def show_route(self, nodes_list:List[int], ele_back:bool = True, line_width:float = 0.5, offset_pos_marker:float=50) -> None:
        #fig, ax = plt.subplots(figsize=[10, 7])
        plt.figure(figsize=(10.3, 7))
        gs = GridSpec(1, 2, width_ratios=[10, 0.3], height_ratios=[1])
        ax = plt.subplot(gs[0])
        cax = plt.subplot(gs[1])
        
        #Draw elevation data
        #Cut the elevation data into size of the road data
        num_lat = self.elevation.shape[1]
        num_lon = self.elevation.shape[0]
        utm_converter = pyproj.Proj(proj='utm', zone=30, ellps='WGS84')
        left_bound_road, up_bound_road = utm_converter(self.road_bound[2], self.road_bound[0])
        right_bound_road, down_bound_road = utm_converter(self.road_bound[3], self.road_bound[1])
        left_bound_ele, up_bound_ele = utm_converter(self.ele_bound[2], self.ele_bound[0])
        right_bound_ele, down_bound_ele = utm_converter(self.ele_bound[3], self.ele_bound[1])

        lon_index_down = int((left_bound_road - left_bound_ele)/(right_bound_ele-left_bound_ele) * num_lon)
        lon_index_up = int((right_bound_road - left_bound_ele)/(right_bound_ele-left_bound_ele) * num_lon)
        lat_index_down = num_lat-int((down_bound_road - down_bound_ele)/(up_bound_ele-down_bound_ele) * num_lat)
        lat_index_up = num_lat-int((up_bound_road - down_bound_ele)/(up_bound_ele-down_bound_ele) * num_lat)

        geo_extent = [left_bound_road, right_bound_road-200, down_bound_road, up_bound_road]

        elevation_resize = self.elevation[lat_index_up:lat_index_down,lon_index_down:lon_index_up]
        # Get min and max values from raster elevation data
        vmin = np.min(elevation_resize)
        vmax = np.max(elevation_resize)
        if ele_back:
            colormap = plt.get_cmap('terrain')
            norm = plt.Normalize(vmin, vmax)
        else:
            colormap = plt.get_cmap('terrain')
            norm = plt.Normalize(vmin, vmax)

        #Draw all roads in blue lines
        for way in self.road3D: 
            if way['type'] == 'way' and 'tags' in way and 'highway' in way['tags'] and way['tags']['highway'] in self.roadtype:
                node_ids = way['nodes']
                #Get the coordinates of each point in each road segment: (lon, lat)
                coordinates = [(way['geometry'][node_index]['lon'], way['geometry'][node_index]['lat'], way['geometry'][node_index]['ele']) for node_index, node_id in enumerate(node_ids)]
                x_values, y_values, z_values = zip(*coordinates)
                for i in range(len(x_values) - 1):
                    if ele_back:
                        ax.plot([x_values[i], x_values[i+1]], [y_values[i], y_values[i + 1]], color='k', linewidth=line_width)
                    else:
                        color = colormap(norm(z_values[i]))
                        ax.plot([x_values[i], x_values[i+1]], [y_values[i], y_values[i + 1]], color=color, linewidth=line_width)

        # Draw target path in red lines
        x = []
        y = []
        deep_red_rgb = "#8B0000"
        for node_index, node in enumerate(nodes_list): 
            coordinate = self.G.nodes[node]['coordinate']
            x.append(coordinate[0])
            y.append(coordinate[1])
            if node_index == 0:
                ax.plot(self.G.nodes[node]['coordinate'][0], self.G.nodes[node]['coordinate'][1],'ro', markersize=1)
                ax.text(self.G.nodes[node]['coordinate'][0]+offset_pos_marker, self.G.nodes[node]['coordinate'][1]-offset_pos_marker, 'start', color=deep_red_rgb)
            elif node_index == len(nodes_list)-1:
                ax.plot(self.G.nodes[node]['coordinate'][0], self.G.nodes[node]['coordinate'][1],'ro', markersize=1)
                ax.text(self.G.nodes[node]['coordinate'][0]+offset_pos_marker, self.G.nodes[node]['coordinate'][1]-offset_pos_marker, 'end', color=deep_red_rgb)
        ax.plot(x, y, 'r-', linewidth = line_width)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Way Elements')
        ax.grid(False)
        if ele_back:
            sm = ScalarMappable(cmap='terrain', norm=norm)
            sm.set_array([]) 
            plt.colorbar(sm, cax=cax)
            ax.imshow(elevation_resize, extent=geo_extent, cmap='terrain', vmin=vmin, vmax=vmax)
        else:
            sm = ScalarMappable(cmap='terrain', norm=norm)
            sm.set_array([]) 
            plt.colorbar(sm, cax=cax)
        plt.show()

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


if __name__ == '__main__':
    path1 = '../data/accra_road.json'
    path2 = '../data/elevation/n05_w001_1arc_v3.tif' #This is the data of the elevation of Ghana
    accra_road = RoadNetwork3D(path1, path2)
    accra_road.integrate()
    accra_road.create_network()
    
    accra_zoo = City(5.625279092167783, -0.20306731748089998, lon_lat=False)
    kotoka_airport = City(813329.05, 620518.36, None, lon_lat=True)
    uni_ghana = City(811795.639, 625324.503, None, lon_lat=True)
    shortest_path = accra_road.get_shortest_path(kotoka_airport, uni_ghana, weight='time')

    visual = Visulisation(accra_road)
    #visual.draw_2Droad()
    visual.show_route(shortest_path, ele_back=True)