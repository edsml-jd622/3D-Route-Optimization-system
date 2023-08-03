import random
from integrate import RoadNetwork3D
from City import City
import pprint
import itertools

def generate_random_CityWeight(G, bound_list=[6,5,-1,0], num_city=5):
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


def tsp_bruteforce(distance_matrix):
    num_locations = len(distance_matrix)
    min_distance = float('inf')
    best_path = []

    # 枚举所有可能的路径
    for path in itertools.permutations(range(num_locations)):
        total_distance = 0
        for i in range(num_locations - 1):
            total_distance += distance_matrix[path[i]][path[i+1]]
        total_distance += distance_matrix[path[-1]][path[0]]  # 回到起点

        if total_distance < min_distance:
            min_distance = total_distance
            best_path = list(path)

    return best_path, min_distance


if __name__ == '__main__':
    path1 = '../data/accra_road.json'
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





