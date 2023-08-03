from typing import List
import pyproj

class City():
    def __init__(self, x:float, y:float, z:float=0, zone:int = 30, lon_lat=False):
        if not lon_lat:
            if -180<x<180 or -85<y<85:
                utm_converter = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
                y, x = utm_converter(y, x)
            self.x = y
            self.y = x
        else:
            if -180<x<180 or -85<y<85:
                utm_converter = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
                x, y = utm_converter(x, y)
            self.x = x
            self.y = y
        self.z = z
        self.coordinates = [self.x, self.y, self.z]
    
    def change_coordinates(self, x:float, y:float, z:float=0, zone:int=30, lon_lat=False):
        if not lon_lat:
            if -180<x<180 or -85<y<85:
                utm_converter = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
                y, x = utm_converter(y, x)
            self.x = y
            self.y = x
        else:
            if -180<x<180 or -85<y<85:
                utm_converter = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
                x, y = utm_converter(x, y)
            self.x = x
            self.y = y
        self.z = z
        self.coordinates = [self.x, self.y, self.z]
    
    def get_coordinates(self) -> List[float]:
        return self.coordinates

if __name__ == '__main__':
    accra_zoo = City(5.625279092167783, -0.20306731748089998)
    kotoka_airport = City(-0.1716731521848131,5.6053060689188, lon_lat=True)
    print(accra_zoo.get_coordinates())
    print(kotoka_airport.get_coordinates())