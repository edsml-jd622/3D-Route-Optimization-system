class City():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.coordinates = [x, y, z]
    
    def change_coordinates(self, x, y, z):
        self.x = x
        self.y = y
        self.z - z
        self.coordinates = [x, y, z]
    
    def get_coordinates(self):
        return self.coordinates

if __name__ == '__main__':
    city1 = City(1,2,3)
    print(city1.get_coordinates())
    city1.change_coordinate(4,5,6)
    print(city1.get_coordinats())