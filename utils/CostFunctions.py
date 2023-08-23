import numpy as np
from typing import Tuple

class CostFunctions():
    def __init__(self):
        self.doc = 'This is the introduction of each method'

    def distance(self, start: Tuple[float], end: Tuple[float]) -> float:
        '''
        Calculate the 3D straight-line distance between two locations.

        Parameters
        ----------
        start: Tuple[float],
            The 3D coordinate of the start location.
        end: Tuple[float],
            The 3D coordinate of the end location.

        Returns
        -------
        distance: float,
            The distance value between start and end.
        '''
        distance = np.sqrt((start[0] - end[0])**2 
                         + (start[1] - end[1])**2 
                         + (start[2] - end[2])**2)
        return distance
    
    def distance_2D(self, start: Tuple[float], end: Tuple[float]) -> float:
        '''
        Calculate the 2D straight-line distance between two locations.

        Parameters
        ----------
        start: Tuple[float],
            The 2D coordinate of the start location.
        end: Tuple[float],
            The 2D coordinate of the end location.

        Returns
        -------
        distance: float,
            The 2D distance value between start and end.
        '''
        distance = np.sqrt((start[0] - end[0])**2 
                    + (start[1] - end[1])**2)
        return distance

    def slope(self, start: Tuple[float], end: Tuple[float]):
        '''
        Calculate the slope between two locations.

        Parameters
        ----------
        start: Tuple[float],
            The 3D coordinate of the start location.
        end: Tuple[float],
            The 3D coordinate of the end location.

        Returns
        -------
        distance: float,
            The slope value between start and end.
        '''
        slope = (start[2] - end[2])/ np.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2)
        return slope

    def travel_time(self, start: Tuple[float], end: Tuple[float], speed: float):
        '''
        Calculate the traveling time between two locations.

        Parameters
        ----------
        start: Tuple[float],
            The 3D coordinate of the start location.
        end: Tuple[float],
            The 3D coordinate of the end location.
        speed: float,
            The speed traveling on the road segment.

        Returns
        -------
        travel_time: float,
            The traveling time between start and end.
        '''
        speed = speed * 1000/3600 # unit of speed: km/h -> m/s
        travel_time = self.distance_2D(start, end) / speed * (1+0.15*0.5**4) + self.distance(start, end)/speed * self.slope(start, end)
        # travel_time = self.distance(start, end) / speed * (1+self.slope(start, end))
        return travel_time