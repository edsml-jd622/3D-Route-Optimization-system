from typing import List
import pyproj

class City():
    '''
    A City contains the 3D coordinate of a location.
    It can automatically transform the coordinate from WGS84 format to utm format.
    '''
    def __init__(self, x:float, y:float, z:float=0, zone:int = 30, lon_lat=False) -> None:
        '''
        x: float,
            The x coordinate of the locatoin.
        y: float,
            The y coordinate of the location.
        z: float(optional),
            The z coordinate of the location.
            The default value is 0.
        zone: int(optional),
            The zone number of which zone the city belong to.
            The default value is 30.
        lon_lat: bool(optional),
            Wether the coordinate is in format of (longitude, latitude).
            If True, x should be longitude, y should be latitude.
            If False, x should be latitudem, y should be longitude.
            The default value is False.
        '''
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
    
    def change_coordinates(self, x:float, y:float, z:float=0, zone:int=30, lon_lat=False) -> None:
        '''
        Change the coordinate of the city.

        Parameters
        ----------
        x: float,
        The new x coordinate value of the location.
        y: float,
        The new y coordinate value of the location.
        z: float(optional),
        The new z coordinate value of the location.
        The default value is 0.
        zone: int(optional),
            The zone number of which zone the city belong to.
            The default value is 30.
        lon_lat: bool(optional),
            Wether the coordinate is in format of (longitude, latitude).
            If True, x should be longitude, y should be latitude.
            If False, x should be latitudem, y should be longitude.
            The default value is False.
        
        Returns
        -------
        None
            This function does not return any value.
        '''
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
        '''
        Get the coordinate of the city.

        Parameters
        ----------
        None

        Returns
        -------
        self.coordinates: List[float],
            The coordinate of the city.
        '''
        return self.coordinates

if __name__ == '__main__':
    accra_zoo = City(5.625279092167783, -0.20306731748089998)
    kotoka_airport = City(-0.1716731521848131,5.6053060689188, lon_lat=True)
    print(accra_zoo.get_coordinates())
    print(kotoka_airport.get_coordinates())