import numpy as np

class WeightCalculation():
    def __init__(self):
        self.doc = 'This is the introduction of each method'

    def distance(self, start, end):
        distance = np.sqrt((start[0] - end[0])**2 
                         + (start[1] - end[1])**2 
                         + (start[2] - end[2])**2)
        return distance
    
    def slope(self, start, end):
        slope = (start[2] - end[2])/ np.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2)
        return slope

    def travel_time(self, start, end, speed):
        travel_time = self.distance(start, end) / speed*(1+self.slope(start, end))
        return travel_time

    def battery_consumption(self):
        pass