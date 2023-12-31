# Building a route optimisation system that takes elevation into consideration

Python tool for making route optimization system and building street network with elevation in consideration.  

## Installation Guide

### Ensure Python is installed:
-  Unix/MacOS: ``python --version`` 
-  Windows: ``py- -version`` 

### Ensure pip is installed:
-  Unix/MacOS: ``python -m pip --version`` 
-  Windows: ``py -m pip --version``

### Ensure git is installed: 
- Unix/MacOS:``git --version``  
- Windows:``git version``  

### Use git to clone this repository: 

``git clone https://github.com/ese-msc-2022/irp-jd622.git``    

This will create the ``irp-jd622`` directory.   

### Installing requirements:

Requirements can be installed using `pip` or `conda`. 

#### Using `pip`:

Naviate to the ``irp-jd622`` directory and run:

``pip install .``  

#### Using `conda`:

Naviate to the ``irp-jd622`` directory and run:

 ``conda env create -f environment.yml``

This will create a conda environment called `jd622`. To activate, run:

  ``conda activate jd622``

## Download Data
Download 2D road data JSON file and elevation raster data file of Accra area from the link below (Users can use their own data of other area, but the format of data should be same as this):

https://imperiallondon-my.sharepoint.com/:f:/g/personal/jd622_ic_ac_uk/EghwGhJTxGlIhjfsIkIcJxcBbV_fLLF_UPZ_rSChgTYNsg?e=3AbfEk

put the `data` folder in the `irp-jd622` folder


# User Guide
To use the route optimization system, navigate to the `irp-jd622` directory and run, `jupyter notebook`. Open the `User_Interface.ipynb` file in Jupyter Notebook. Follow the instructions of the user interface.

Here is some key function demo below:

## Integration

- `accra_road = RoadNetwork3D(road_path, ele_path)`, create the instance of integration class. `road_path` is the 2D road data JSON file path. `ele_path` is the elevation data file path.
- `accra_road.integrate()`, integrate 2D road data and elevation data into 3D road data that will be stored in the instance.
- `accra_road.create_network()`, create a network graph for the 3D road data. The graph will be stored in the instance too.
- `accra_road.get_shortest_path_length(city1, city2, weight)` get the length(travel time) of the route planning.
-`accra_road.get_closest_point(city)`, get the closest point in the network graph for a given coordinate.


## City
`kotoka_airport = City(5.605522862563998, -0.17187326099346129, None, lon_lat=False)`, create any locations in Accra you want. 

Each instance of `City` class stores information of one location. 

- The first parameter is latitude
- The second parameter is longitude
- The third parameter is elevation (optional)
- The last parameter is whether the coordinate format is (lon, lat). If `True`, then the first parameter should be longitude, and the second parameter should be latitude. Users can also use UTM coordinate system. Northing corresponds to latitude, easting corresponds to longitude.

## Visualisation
- `accra_road.get_shortest_path(city1, city2, weight)`, get the route planning which is a list of points on the target path.
- `visual = Visulisation(accra_road)`, reate an instance of `Visualisation` class for `accra_road`
- `visual.show_route(shortest_path)`, draw the shortest path on the map. You can specify the parameter `ele_back` for different figure style.


## Pointer Network Model
The pointer network model and its training Jupyter notebook can be found in folder `pointer_network`. In this folder, `PointerNet.py` is the model structure, 
**the Code of the Pointer Network Model is referenced by the following link: https://github.com/shirgur/PointerNet.** `train_scaleData_2048batch.ipynb` is the training Jupyter Notebook. The only revision on the model structure is adding a parameter named `num_city` which is the input size of the embedding layer. Users can change this parameter to train the model for solving traveling salesman problem(TSP) for different number of locations. For example, if `num_city`=5, the model can used for solving TSP of 5 locations.

In the `User_interface`:
- `model = PointerNet(5,256,512,2,0,True)` and 
`model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])`, load the trained Pointer Network Model.

- `_, p = model(torch.tensor(adjacent_matrix[index], dtype=torch.float32).unsqueeze(0))`, get the prediction of route optimization for a given adjacent matrix.

# Tests
Navigate the `jd622/tests/` folder, run each test with following code:
- `pytest test_Road_Network.py`
- `pytest test_City.py`
- `pytest test_CostFunctions.py`
- `pytest test_Generator_Tool.py`
- `pytest test_Visulisation.py`

# Other Files
`performance_test` folder: 

  - `Random_locations_testSystem.ipynb` is used to generate 100 random paths and their distance and travel time calculated by the system developed in this research. 
  - `random_locations.csv` is the data file produced by `Random_locations_testSystem.ipynb`.
  - `random_locations_withGoogle.csv` is based on `random_locations.csv`, only added distance and travel time from Google Maps manually for comparing.

`utils` folder:
  - `Data_Generator.py` a script for producing dataset for training Pointer Network.