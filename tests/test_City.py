import sys
import pytest

sys.path.append('../')
from utils.City import City

def test_change_coordinate():
    city1 = City(1000, 2000, 3000, lon_lat=True)
    city1.change_coordinates(2000, 1000, 3000, lon_lat=True)
    expected_coord = [2000, 1000, 3000]

    assert city1.coordinates == expected_coord

def test_get_coordinate():
    city1 = City(1000, 2000, 3000)
    city1_coord = city1.get_coordinates()
    expected_coord = [2000, 1000, 3000]
    
    assert city1_coord == expected_coord

if __name__ == "__main__":
    pytest.main()

