import numpy as np

class CostFunctions():
    def __init__(self):
        self.doc = 'This is the introduction of each method'

    def distance(self, start, end):
        distance = np.sqrt((start[0] - end[0])**2 
                         + (start[1] - end[1])**2 
                         + (start[2] - end[2])**2)
        return distance
    
    def distance_2D(self, start, end):
        distance = np.sqrt((start[0] - end[0])**2 
                    + (start[1] - end[1])**2)
        return distance

    def slope(self, start, end):
        slope = (start[2] - end[2])/ np.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2)
        return slope

    def travel_time(self, start, end, speed):
        speed = speed * 1000/3600 # unit of speed: km/h -> m/s
        travel_time = self.distance_2D(start, end) / speed * (1+0.15*0.5**4) + self.distance(start, end)/speed * self.slope(start, end)
        # travel_time = self.distance(start, end) / speed * (1+self.slope(start, end))
        return travel_time

    def battery_consumption(self):
        pass