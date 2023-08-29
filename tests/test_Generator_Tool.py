'''
Name: Jinsong Dong
GitHub Username: edsml-jd622
'''
import sys
import pytest

sys.path.append('../')
from utils.Road_Network import *
from utils.Generator_Tool import generate_random_CityWeight, tsp_bruteforce

@pytest.fixture(scope="module")
def accra_road():
    path1 = '../data/accra_road.json'
    path2 = '../data/elevation/n05_w001_1arc_v3.tif'
    accra_road = RoadNetwork3D(path1, path2)
    accra_road.integrate()
    accra_road.create_network()
    return accra_road

def test_generate_random_CityWeigh_5city(accra_road):
    i = 1
    while i>0:
        try:
            my_cities, weight = generate_random_CityWeight(accra_road, num_city=5)
            i -= 1
        except:
            continue
    list_length = len(my_cities)
    weight_shape = weight.shape
    expected_list_length = 5
    expected_weight_shape = (5,5)

    assert list_length == expected_list_length and weight_shape == expected_weight_shape

def test_generate_random_CityWeigh_10city(accra_road):
    i = 1
    while i>0:
        try:
            my_cities, weight = generate_random_CityWeight(accra_road, bound_list= [5.7, 5.5, -0.8, -0.5], num_city=10)
            i -= 1
        except:
            continue
    list_length = len(my_cities)
    weight_shape = weight.shape
    expected_list_length = 10
    expected_weight_shape = (10,10)

    assert list_length == expected_list_length and weight_shape == expected_weight_shape

def test_tsp_bruteforce_square():
    test_matrix = np.array([
        [0, 1, np.sqrt(2), 1],
        [1, 0, 1, np.sqrt(2)],
        [np.sqrt(2), 1, 0, 1],
        [1, np.sqrt(2), 1, 0]
    ])
    path, distance = tsp_bruteforce(test_matrix)
    expected_path = [0,1,2,3]
    expected_dist = 4
    assert path == expected_path and distance == expected_dist

def test_tsp_bruteforce_hexagon():
    test_matrix = np.array([
        [0, 1, np.sqrt(3), 2, np.sqrt(3), 1],
        [1, 0, 1, np.sqrt(3), 2, np.sqrt(3)],
        [np.sqrt(3), 1, 0, 1, np.sqrt(3), 2],
        [2, np.sqrt(3), 1, 0, 1, np.sqrt(3)],
        [np.sqrt(3), 2, np.sqrt(3), 1, 0, 1],
        [1, np.sqrt(3), 2, np.sqrt(3), 1, 0]
    ])
    path, distance = tsp_bruteforce(test_matrix)
    expected_path = [0,1,2,3,4,5]
    expected_dist = 6
    assert path == expected_path and distance == expected_dist

if __name__ == "__main__":
    pytest.main()