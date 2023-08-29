'''
Name: Jinsong Dong
GitHub Username: edsml-jd622
'''
import sys
import os
import pytest
import json
from unittest.mock import patch, Mock

sys.path.append('../')
from utils.Road_Network import *
from utils.City import City

@pytest.fixture(scope="module")
def accra_road():
    path1 = '../data/accra_road.json'
    path2 = '../data/elevation/n05_w001_1arc_v3.tif'
    accra_road = RoadNetwork3D(path1, path2)
    accra_road.integrate()
    accra_road.create_network()
    return accra_road

@pytest.fixture(autouse=True)
def temp_test_file():
    temp_road_path = "./test_file.json"
    temp_ele_path = "./test_ele.tif"
    # Create test files
    data = {
        "elements": [
            {
                "type": "way",
                "nodes": [0,1,2,3],
                "geometry": [
                    {"lat": 85000,
                    "lon": 61000,
                    "ele": 0},
                    {"lat": 85000,
                    "lon": 61001,
                    "ele": 1},
                    {"lat": 85001,
                    "lon": 61001,
                    "ele": 2},
                    {"lat": 85001,
                    "lon": 61000,
                    "ele": 3}
                ],
                "tags": {
                    "highway": "primary",
                    "speed": 50
                }
            }
        ]
    }
    with open(temp_road_path, 'w') as json_file:
        json.dump(data, json_file)
    matrix = np.array([[3, 2],
                   [0, 1]], dtype=np.float64)
    image = Image.fromarray(matrix)
    image.save(temp_ele_path)
    yield temp_road_path, temp_ele_path
    os.remove(temp_road_path)
    os.remove(temp_ele_path)

def test_add_road_type(accra_road):
    original_types = accra_road.get_roadtype()
    accra_road.add_road_type('primary')
    added_types = list(set(original_types + ['primary']))
    assert accra_road.get_roadtype() == added_types

def test_delete_road_type(accra_road):
    original_types = accra_road.get_roadtype()
    accra_road.delete_road_type('primary')
    original_types.remove('primary')
    deleted_types = list(set(original_types))
    assert accra_road.get_roadtype() == deleted_types
    
    #add back to the original types
    accra_road.add_road_type('primary')

def test_print_road_type(accra_road):
    with patch('builtins.print') as mock_print:
        accra_road.print_road_type()

    mock_print.assert_called()

def test_show_tags(accra_road):
    mock_show = Mock()

    with patch('matplotlib.pyplot.show', mock_show):
        accra_road.show_tags() 
    
    mock_show.assert_called_once()

def test_print_data(accra_road):
    with patch('builtins.print') as mock_print:
        accra_road.print_data()

    mock_print.assert_called()

def test_get_elevation(temp_test_file):
    test_road = RoadNetwork3D(temp_test_file[0], temp_test_file[1])
    expected_elevation = np.array([[3,2],
                                   [0,1]])

    assert np.array_equal(test_road.get_elevation(), expected_elevation)

def test_get_3Droad(temp_test_file):
    test_road = RoadNetwork3D(temp_test_file[0], temp_test_file[1])
    expected_road = [
            {
                "type": "way",
                "nodes": [0,1,2,3],
                "geometry": [
                    {"lat": 85000,
                    "lon": 61000,
                    "ele": 0},
                    {"lat": 85000,
                    "lon": 61001,
                    "ele": 1},
                    {"lat": 85001,
                    "lon": 61001,
                    "ele": 2},
                    {"lat": 85001,
                    "lon": 61000,
                    "ele": 3}
                ],
                "tags": {
                    "highway": "primary",
                    "speed": 50
                }
            }
        ]
    assert test_road.get_3Droad() == expected_road

def test_get_roadtype(accra_road):
    expected_road_type = ['residential', 'unclassified', 'service', 'primary', 'primary_link', 'trunk','trunk_link', 'secondary', 'secondary_link', 'tertiary','tertiary_link']
    
    assert set(accra_road.get_roadtype()) == set(expected_road_type)

def test_get_2Droad(temp_test_file):
    test_road = RoadNetwork3D(temp_test_file[0], temp_test_file[1])
    expected_road = [
            {
                "type": "way",
                "nodes": [0,1,2,3],
                "geometry": [
                    {"lat": 85000,
                    "lon": 61000,
                    "ele": 0},
                    {"lat": 85000,
                    "lon": 61001,
                    "ele": 1},
                    {"lat": 85001,
                    "lon": 61001,
                    "ele": 2},
                    {"lat": 85001,
                    "lon": 61000,
                    "ele": 3}
                ],
                "tags": {
                    "highway": "primary",
                    "speed": 50
                }
            }
        ]
    assert test_road.get_2Droad() == expected_road

def test_integrate(temp_test_file):
    test_road = RoadNetwork3D(temp_test_file[0], temp_test_file[1], ele_bound=[85001.5, 84999.5, 60999.5, 61001.5])
    test_road.integrate()
    expected_integrated_data = [
            {
                "type": "way",
                "nodes": [0,1,2,3],
                "geometry": [
                    {"lat": 85000,
                    "lon": 61000,
                    "ele": 0},
                    {"lat": 85000,
                    "lon": 61001,
                    "ele": 1},
                    {"lat": 85001,
                    "lon": 61001,
                    "ele": 2},
                    {"lat": 85001,
                    "lon": 61000,
                    "ele": 3}
                ],
                "tags": {
                    "highway": "primary",
                    "speed": 50
                }
            }
        ]

    assert test_road.get_3Droad() == expected_integrated_data

def test_get_closest_point(accra_road):
    my_point = City(5.6110919, -0.1856412)
    point_id = 1837599323

    assert accra_road.get_closest_point(my_point)[0] == point_id

def test_create_network(temp_test_file):
    test_road = RoadNetwork3D(temp_test_file[0], temp_test_file[1], ele_bound=[85001.5, 84999.5, 60999.5, 61001.5])
    test_road.integrate()
    test_road.create_network()

    expected_nodes_num = 4
    expected_edges_num = 6

    assert test_road.network.number_of_nodes() == expected_nodes_num and test_road.network.number_of_edges() == expected_edges_num

def test_get_network(temp_test_file):
    test_road = RoadNetwork3D(temp_test_file[0], temp_test_file[1], ele_bound=[85001.5, 84999.5, 60999.5, 61001.5])
    test_road.integrate()
    test_road.create_network()
    test_network = test_road.get_network()

    expected_nodes_num = 4
    expected_edges_num = 6

    assert test_network.number_of_nodes() == expected_nodes_num and test_network.number_of_edges() == expected_edges_num

def test_get_shortest_path(accra_road):
    path = [6007214717, 30730141, 30730132, 1837599323]
    p1 = City(5.6116855, -0.1849598)
    p2 = City(5.6115441, -0.1851643)
    p3 = City(5.611354, -0.1853967)
    p4 = City(5.6110919, -0.1856412)
    assert accra_road.get_shortest_path(p1, p4) == path

def test_get_shortest_path_length(temp_test_file):
    test_road = RoadNetwork3D(temp_test_file[0], temp_test_file[1])
    test_road.flag_3D=1
    test_road.create_network()
    
    city1 = City(85000, 61000)
    city2 = City(85000, 61001)
    city3 = City(85001, 61001)
    city4 = City(85001, 61000)
    length = test_road.get_shortest_path_length(city1, city4)
    expected_length = 3*np.sqrt(2)

    assert length == expected_length
     
def test_weight_matrix(temp_test_file):
    test_road = RoadNetwork3D(temp_test_file[0], temp_test_file[1])
    test_road.flag_3D=1
    test_road.create_network()

    city1 = City(85000, 61000)
    city2 = City(85000, 61001)
    city3 = City(85001, 61001)
    city4 = City(85001, 61000)
    city_list = [city1, city2, city3, city4]
    
    weight_matrix = test_road.weight_matrix(city_list)
    expected_matrix = [[0, np.sqrt(2), 2*np.sqrt(2), 3*np.sqrt(2)],
                       [np.sqrt(2), 0, np.sqrt(2), 2*np.sqrt(2)],
                       [2*np.sqrt(2), np.sqrt(2), 0, np.sqrt(2)],
                       [3*np.sqrt(2), 2*np.sqrt(2), np.sqrt(2), 0]]

    assert np.array_equal(weight_matrix, expected_matrix)

if __name__ == '__main__':
    pytest.main()