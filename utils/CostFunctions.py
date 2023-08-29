'''
Name: Jinsong Dong
GitHub Username: edsml-jd622
'''
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
        slope = (end[2] - start[2])/ np.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2)
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
        slope_weight = {
            10: [0.012, 0.023, 0.003, 0.052, 0.066],
            20: [0.064, 0.083, 0.093, 0.152, 0.166], 
            30: [0.172, 0.193, 0.205, 0.269, 0.284], 
            40: [0.213, 0.239, 0.253, 0.333, 0.351], 
            50: [0.421, 0.446, 0.461, 0.538, 0.557], 
            60: [0.565, 0.593, 0.609, 0.694, 0.715], 
            70: [0.723, 0.754, 0.772, 0.865, 0.888]
        }
        speed_ref = speed //10 * 10 if speed <=70 else 70
        slope = self.slope(start, end)
        slope_ref = 0 if slope<0.005 else 4 if slope>0.04 else round(slope*100)
        weight = slope_weight[speed_ref][slope_ref] if slope_ref > -1 else 0
        speed = speed * 1000/3600 # unit of speed: km/h -> m/s
        travel_time = self.distance_2D(start, end) / speed * (1+0.15*0.5**4) + self.distance(start, end)/speed * weight
        # travel_time = self.distance(start, end) / speed * (1+self.slope(start, end))
        return travel_time