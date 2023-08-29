'''
Name: Jinsong Dong
GitHub Username: edsml-jd622
'''
import random
from .Road_Network import RoadNetwork3D
import numpy as np
from typing import List
from .City import City
import itertools

def generate_random_CityWeight(G: RoadNetwork3D, bound_list: List[int]=[6,5,-1,0], num_city: int=5):
    '''
    Generate 5 random locations and return their coordinates and adjacent matrix.

    Parameters
    ----------
    G: RoadNetwork3D,
        The network graph.
    bound_list: List[int] (optional),
        The boundary of the graph.
        The default values is [6, 5, -1, 0] which is specially for Accra area
    num_city: int (optional),
        The number of locations to generate.
        The defalt value is 5.
    
    Returns
    -------
    coor_list: List[Tuple[float]],
        The coordinates list of the locations generated.
    weight: np.ndarray, 
        The 2D array of the adjacent matrix of the locations generated.
    '''
    coor_list = []
    city_list = []
    for i in range(num_city):
        lat = bound_list[1]+random.random()
        lon = bound_list[2]+random.random()
        city_id, coor = G.get_closest_point(City(lat,lon))
        coor = coor['coordinate']

        coor_list.append(coor)
        city_list.append(City(coor[0], coor[1], coor[2], lon_lat=True))

    weight = G.weight_matrix(city_list)
    
    return coor_list, weight


def tsp_bruteforce(distance_matrix: np.ndarray):
    '''
    Use brute-force method to solve Traveling Salesman Problem(TSP) given a certain adjacent matrix.

    Parameters
    ----------
    distance_matrix: np.ndarray,
        The adjacent matrix for a seires of locations.
    
    Returns
    -------
    best_path: List[int],
        The list of index for the optimal route (The optimal answer of TSP).
    min_distance: float,
        The distance of the optimal route.
    '''
    num_locations = len(distance_matrix)
    min_distance = float('inf')
    best_path = []

    for path in itertools.permutations(range(num_locations)):  # Enumerate all possible paths
        total_distance = 0
        for i in range(num_locations - 1):
            total_distance += distance_matrix[path[i]][path[i+1]]
        total_distance += distance_matrix[path[-1]][path[0]]  # Get back to the start point

        if total_distance < min_distance:
            min_distance = total_distance
            best_path = list(path)

    return best_path, min_distance


if __name__ == '__main__':
    path1 = '../data/accra_road.json'  #This is the data of the 2d road of Ghana
    path2 = '../data/elevation/n05_w001_1arc_v3.tif' #This is the data of the elevation of Ghana
    accra_road = RoadNetwork3D(path1, path2)
    accra_road.integrate()
    accra_road.create_network()

    # city = City(5.1, -0.2)
    # point, cor = accra_road.get_closest_point(city)
    # print(point, ':', cor['coordinate'])

    bound_list = [6.2,5.3,-0.7,-0.3]
    num_city = 5
    my_cities, weight = generate_random_CityWeight(accra_road, bound_list, num_city)
    best_path, min_distance = tsp_bruteforce(weight)
    print(my_cities)
    print("最短路径:", best_path)
    print("最短距离:", min_distance)





