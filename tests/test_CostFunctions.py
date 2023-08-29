'''
Name: Jinsong Dong
GitHub Username: edsml-jd622
'''
import sys
import pytest
import numpy as np

sys.path.append('../')
from utils.CostFunctions import CostFunctions

@pytest.fixture(scope='module')
def test_cost():
    return CostFunctions()

def test_distance(test_cost):
    start = (0,1,2)
    end = (1,2,3)
    distance = test_cost.distance(start, end)
    expected_distance = np.sqrt(3)

    assert distance == expected_distance

def test_distance_2D(test_cost):
    start = (0, 100, 100000000)
    end = (100, 0, 100)
    distance_2D = test_cost.distance_2D(start, end)
    expected_distance_2D = 100*np.sqrt(2)

    assert distance_2D == expected_distance_2D

def test_slope(test_cost):
    start = (0,10,100)
    end = (10, 0, 1000)
    slope = test_cost.slope(start, end)
    expected_slope = 90 / np.sqrt(2)
    
    assert slope == expected_slope

def test_travel_time(test_cost):
    start = (0, 100, 10)
    end = (10, 1000, 20)
    speed = 60
    travel_time = test_cost.travel_time(start, end, speed)
    expected_travel_time = np.sqrt(100 + 900**2) / (60 * 1000 / 3600) * (1+0.15*0.5**4) + np.sqrt(100 + 900**2 + 100) / (60 * 1000 / 3600) * 0.593

    assert travel_time == expected_travel_time

def test_travel_time2(test_cost):
    start = (0, 100, 10)
    end = (10, 1000, 20)
    speed = 100
    travel_time = test_cost.travel_time(start, end, speed)
    expected_travel_time = np.sqrt(100 + 900**2) / (speed * 1000 / 3600) * (1+0.15*0.5**4) + np.sqrt(100 + 900**2 + 100) / (speed * 1000 / 3600) * 0.754

    assert travel_time == expected_travel_time

def test_travel_time3(test_cost):
    start = (0, 100, 10)
    end = (10, 1000, 30)
    speed = 100
    travel_time = test_cost.travel_time(start, end, speed)
    expected_travel_time = np.sqrt(100 + 900**2) / (speed * 1000 / 3600) * (1+0.15*0.5**4) + np.sqrt(100 + 900**2 + 400) / (speed * 1000 / 3600) * 0.772

    assert travel_time == expected_travel_time

def test_travel_time4(test_cost):
    start = (0, 100, 30)
    end = (10, 1000, 10)
    speed = 100
    travel_time = test_cost.travel_time(start, end, speed)
    expected_travel_time = np.sqrt(100 + 900**2) / (speed * 1000 / 3600) * (1+0.15*0.5**4) + np.sqrt(100 + 900**2 + 400) / (speed * 1000 / 3600) * 0.723

    assert travel_time == expected_travel_time

if __name__ == "__main__":
    pytest.main()