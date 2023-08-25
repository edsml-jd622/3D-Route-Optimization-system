import sys
import pytest
from unittest.mock import patch, Mock

sys.path.append('../')
from utils.City import City
from utils.Visulisation import Visulisation
from utils.Road_Network import RoadNetwork3D

@pytest.fixture(scope="module")
def accra_road():
    path1 = '../data/accra_road.json'
    path2 = '../data/elevation/n05_w001_1arc_v3.tif'
    accra_road = RoadNetwork3D(path1, path2)
    accra_road.integrate()
    accra_road.create_network()
    return accra_road

@pytest.fixture(scope='module')
def visual(accra_road):
    return Visulisation(accra_road)

def test_show_route(accra_road, visual):
    mock_show = Mock()

    with patch('matplotlib.pyplot.show', mock_show):
        kotoka_airport = City(5.605522862563998, -0.17187326099346129, None, lon_lat=False)
        uni_ghana = City(5.650618146052781, -0.18703194651322047, None, lon_lat=False)
        shortest_path = accra_road.get_shortest_path(kotoka_airport, uni_ghana, weight='distance')
        visual.show_route(shortest_path) 
    
    mock_show.assert_called_once()

def test_draw_3Droad(visual):
    mock_show = Mock()

    with patch('matplotlib.pyplot.show', mock_show):
        visual.draw_3Droad() 

def test_draw_2Droad(visual):
    mock_show = Mock()

    with patch('matplotlib.pyplot.show', mock_show):
        visual.draw_2Droad() 


if __name__ == '__main__':
    pytest.main()
