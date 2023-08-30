'''
Name: Jinsong Dong
GitHub Username: edsml-jd622
'''
from Road_Network import RoadNetwork3D
import numpy as np
from Generator_Tool import generate_random_CityWeight, tsp_bruteforce

if __name__ == '__main__':
    path1 = '../data/accra_road.json'
    path2 = '../data/elevation/n05_w001_1arc_v3.tif' #This is the data of the elevation of Ghana
    accra_road = RoadNetwork3D(path1, path2)
    accra_road.integrate()
    accra_road.create_network()

    bound_list = [6.2,5.3,-0.7,-0.3]
    num_city = 5
    cities_data = []
    weights_data = []
    labels_data = []
    i=10000


    while i != 0:
        try:
            my_cities, weight = generate_random_CityWeight(accra_road, bound_list, num_city)
            best_path, min_distance = tsp_bruteforce(weight)
        except:
            print('failed')
            continue
        print('rest turn = ', i)
        cities_data.append(my_cities)
        weights_data.append(weight)
        labels_data.append(best_path)
        i -= 1
    print(cities_data)
    print(weights_data)
    print(labels_data)
    np.save('cities_data_5city_29.npy', np.array(cities_data))
    np.save('weights_data_5city_29.npy', np.array(weights_data))
    np.save('labels_data_5city_29.npy', np.array(labels_data))