import argparse
import math
import os
import pickle as pkl
import rasterio.features
import pandas as pd
import geopandas as gpd
import numpy as np

from scipy.spatial import KDTree
from scipy.sparse import lil_matrix

from shapely import affinity
from shapely.geometry import Point, MultiPolygon
from shapely.ops import unary_union

from tqdm import tqdm, trange
import h5py


from grid import Grid


class Preprocess(object):
    """
        This class implement common preprocessing for NYC and Singapore data
        The preprocessing includes the following parts
        - Turn the POI data and the building type into one-hot vector, Attach the POI data to the building data
        - Perform Poisson-disk sampling on the boundary map, remove overlapped points
        - Calculate density encoding and location encoding for each point
        - Group the buildings and the POIs according to the patterns
        - Group the patterns according to the regions
    """

    ### PUBLIC CONSTRUCTORS ###

    def __init__(self, city):

        # Setup the names of the input/output folders.
        in_path = f'./data/projected/{city}/'
        out_path = f'./data/processed/{city}/'
        self.in_path = in_path
        self.out_path = out_path

        # Create the output folder if it does not exist.
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Setup the names of several input subfolders.
        self.building_in_path = in_path + 'building/building.pkl'
        self.poi_in_path = in_path + 'poi/poi.pkl'
        self.segmentation_in_path = in_path + 'segmentation/segmentation.pkl'
        self.boundary_in_path = in_path + 'boundary/boundary.pkl'
        self.building_out_path = out_path + 'building.pkl'
        self.poi_out_path = out_path + 'poi.pkl'
        self.segmentation_out_path = out_path + 'segmentation'

        # Determine the boundaries of the considered geographical area.
        # NOTE in the case of Paris, we are going to consider the geometries of the IRIS cells.
        # NOTE 2: after unary_union, if we have a polygon convert it into a multipolygon, as it
        #         is required by the preprocessing's Grid class.
        print(f'Loading boundary from {self.boundary_in_path}')
        boundary_shapefile = gpd.GeoDataFrame(pd.read_pickle(self.boundary_in_path))
        self.boundary = unary_union(boundary_shapefile['geometry'])
        if ~isinstance(self.boundary, MultiPolygon) : self.boundary = MultiPolygon([self.boundary])

        #boundary = [boundary_row['geometry'] for index, boundary_row in boundary_shapefile.iterrows()]
        #if len(boundary) > 1:
        #    boundary = unary_union(boundary)
        #else:
        #    boundary = boundary[0]
        #self.boundary = boundary


    def get_building_and_poi(self, force=False):
        """
        This function will process the building and the pois.
        1. Load the building and poi from shapefile
        2. Turn the two shapefile into list of dict
            building: shape, type
            poi: x, y, code, fclass
        3. Turn the building type, poi code/fclass into one hot
        4. Attach the pois to buildings
        5. Save them to pickle, and return
        """

        # Check if the buildings and pois have already been processed.
        if not force and os.path.exists(self.building_out_path):
            print('Loading building from {}'.format(self.building_out_path))
            with open(self.building_out_path, 'rb') as f:
                building = pkl.load(f)
            print('Loading poi from {}'.format(self.poi_out_path))
            with open(self.poi_out_path, 'rb') as f:
                poi = pkl.load(f)
            return [{'shape': b['shape']} for b in building], poi

        
        print('Preprocessing building and poi data...')
        buildings_shapefile = gpd.GeoDataFrame(pd.read_pickle(self.building_in_path))
        pois_shapefile = gpd.GeoDataFrame(pd.read_pickle(self.poi_in_path))
        building = []
        poi = []
        for index, building_row in tqdm(buildings_shapefile.iterrows(), total=buildings_shapefile.shape[0]):
            output = {}
            # process polygon
            shape = building_row['geometry']
            output['shape'] = shape
            output['type'] = building_row['type']
            building.append(output)
        del buildings_shapefile
        
        for index, poi_row in tqdm(pois_shapefile.iterrows(), total=pois_shapefile.shape[0]):
            output = {}
            # process point
            output['x'] = poi_row['geometry'].x
            output['y'] = poi_row['geometry'].y
            output['code'] = poi_row['code']
            output['fclass'] = poi_row['fclass']
            poi.append(output)
        del pois_shapefile

        
        print('Turning building type and poi code/fclass into one-hot...')
        building_type = set([b['type'] for b in building])
        poi_code = set([p['code'] for p in poi])
        poi_fclass = set([p['fclass'] for p in poi])
        building_type_dict = {t: i for i, t in enumerate(building_type)}
        poi_code_dict = {c: i for i, c in enumerate(poi_code)}
        poi_fclass_dict = {f: i for i, f in enumerate(poi_fclass)}
        for b in building:
            # b['onehot'] = [0] * len(building_type) # This is brutally inefficient with memory!
            b['onehot'] = np.zeros(len(building_type), dtype=np.int8)
            b['onehot'][building_type_dict[b['type']]] = 1
            
        poi_dim = len(poi_code) + len(poi_fclass)
        for p in poi:
            # p['onehot'] = [0] * poi_dim # This is brutally inefficient with memory!
            p['onehot'] = np.zeros(poi_dim, dtype=np.int8)
            p['onehot'][poi_code_dict[p['code']]] = 1
            p['onehot'][len(poi_code) + poi_fclass_dict[p['fclass']]] = 1

        
        print('Attaching pois to buildings...')
        # build a kd-tree for poi
        poi_x = [p['x'] for p in poi]
        poi_y = [p['y'] for p in poi]
        poi_tree = KDTree(np.array([poi_x, poi_y]).T)
        attached_poi_indices = set()  # Set to store indices of attached pois
        attached_poi = []
        for b in tqdm(building):
            # sum up all the pois in the building
            b['poi'] = [0] * poi_dim
            bounds = b['shape'].bounds
            cx = (bounds[0] + bounds[2]) / 2
            cy = (bounds[1] + bounds[3]) / 2
            height = bounds[3] - bounds[1]
            width = bounds[2] - bounds[0]
            radius = np.sqrt(height ** 2 + width ** 2) / 2
            # find all the pois in the radius
            poi_index = poi_tree.query_ball_point([cx, cy], radius)
            for i in poi_index:
                if not b['shape'].contains(Point(poi[i]['x'], poi[i]['y'])):
                    continue
                b['poi'] = [b['poi'][j] + poi[i]['onehot'][j] for j in range(poi_dim)]
                attached_poi.append(poi[i])
                attached_poi_indices.add(i)  # Add the index to the set

        
        print('Taking note of the POIs that are not attached to any building...')
        # poi_not_attached = [p for p in poi if p not in attached_poi] # Incredibly slow, substituted with a set-based approach.
        unattached_poi_indices = set(range(len(poi))) - attached_poi_indices
        poi_not_attached = [poi[i] for i in unattached_poi_indices]

        
        print('Saving building and poi data...')
        with open(self.building_out_path, 'wb') as f:
            pkl.dump(building, f, protocol=5)
        with open(self.poi_out_path, 'wb') as f:
            pkl.dump(poi_not_attached, f, protocol=5)
        
        # NOTE: the subsequent methods will only need the key "shape" from every building.
        # return building, poi_not_attached
        return [{'shape': building['shape']} for building in building_list], poi_not_attached


    def poisson_disk_sampling(self, building_list, poi_list, radius, force=False):
        random_point_out_path = self.out_path + 'random_point_' + str(radius) + 'm.pkl'
        
        if not force and os.path.exists(random_point_out_path):
            print("Poisson disk sampling has already been executed for this combination of city and radius!")
            with open(random_point_out_path, 'rb') as f:
                result = pkl.load(f)
            return result
        
        grid = Grid(self.boundary, radius, building_list, poi_list)
        result = grid.poisson_disk_sampling()
        with open(random_point_out_path, 'wb') as f:
            pkl.dump(result, f, protocol=4)
        return result

    
    def partition(self, building_list, poi_list, random_point_list, radius, force=False):
        if not force and os.path.exists(self.segmentation_out_path):
            with open(self.segmentation_out_path, 'rb') as f:
                result = pkl.load(f)
            return result
        
        print('Partition city data by road network...')
        # segmentation_shapefile = gpd.read_file(self.segmentation_in_path)
        segmentation_shapefile = gpd.GeoDataFrame(pd.read_pickle(self.segmentation_in_path))
        
        segmentation_polygon_list = []
        for row in segmentation_shapefile.iterrows():
            it = row[1]
            segmentation_polygon_list.append(it.geometry)
        result = []
        building_loc = [[b['shape'].centroid.x, b['shape'].centroid.y] for b in building_list]
        poi_loc = [[p['x'], p['y']] for p in poi_list]
        random_point_loc = random_point_list
        building_tree = KDTree(building_loc)
        poi_tree = KDTree(poi_loc)
        random_point_tree = KDTree(random_point_loc)
        for i in trange(len(segmentation_polygon_list)):
            shape = segmentation_polygon_list[i]
            pattern = {
                'shape': shape,
                'building': [],
                'poi': [],
                'random_point': []
            }
            # calculate the diameter of the shape
            bounds = shape.bounds
            dx = bounds[2] - bounds[0]
            dy = bounds[3] - bounds[1]
            diameter = math.sqrt(dx * dx + dy * dy) / 2
            # find the buildings in the shape
            building_index = building_tree.query_ball_point([shape.centroid.x, shape.centroid.y], diameter)
            for j in building_index:
                if shape.intersects(building_list[j]['shape']):
                    pattern['building'].append(j)
            # find the poi in the shape
            poi_index = poi_tree.query_ball_point([shape.centroid.x, shape.centroid.y], diameter)
            for j in poi_index:
                if shape.contains(Point(poi_loc[j][0], poi_loc[j][1])):
                    pattern['poi'].append(j)
            # find the random points in the shape
            random_point_index = random_point_tree.query_ball_point([shape.centroid.x, shape.centroid.y], diameter)
            for j in random_point_index:
                if shape.contains(Point(random_point_loc[j][0], random_point_loc[j][1])):
                    pattern['random_point'].append(j)
            # ignore the pattern without any building & random point
            if len(pattern['building']) == 0:
                continue
            result.append(pattern)
        with open(self.segmentation_out_path + f'_{radius}.pkl', 'wb') as f:
            pkl.dump(result, f, protocol=4)
        return result

        
    def rasterize_buildings_OLD(self, building_list, rotation=True, force=False):
        
        image_out_path = self.out_path + 'building_raster.npz'
        rotation_out_path = self.out_path + 'building_rotation.npz'
        if not force and os.path.exists(image_out_path):
            return np.load(image_out_path)['arr_0']
            
            
        print('Rasterize buildings...')
        images = np.zeros((len(building_list), 224, 224), dtype=np.uint8)
        rotations = np.zeros((len(building_list), 2), dtype=float)
        for i in trange(len(building_list)):
            polygon = building_list[i]['shape']
            if rotation:
                # rotate the polygon to align with the x-axis
                rectangle = polygon.minimum_rotated_rectangle
                xc = polygon.centroid.x
                yc = polygon.centroid.y
                rec_x = []
                rec_y = []
                for point in rectangle.exterior.coords:
                    rec_x.append(point[0])
                    rec_y.append(point[1])
                top = np.argmax(rec_y)
                top_left = top - 1 if top > 0 else 3
                top_right = top + 1 if top < 3 else 0
                x0, y0 = rec_x[top], rec_y[top]
                x1, y1 = rec_x[top_left], rec_y[top_left]
                x2, y2 = rec_x[top_right], rec_y[top_right]
                d1 = np.linalg.norm([x0 - x1, y0 - y1])
                d2 = np.linalg.norm([x0 - x2, y0 - y2])
                if d1 > d2:
                    cosp = (x1 - x0) / d1
                    sinp = (y0 - y1) / d1
                else:
                    cosp = (x2 - x0) / d2
                    sinp = (y0 - y2) / d2
                rotations[i] = [cosp, sinp]
                matrix = (cosp, -sinp, 0.0,
                          sinp, cosp, 0.0,
                          0.0, 0.0, 1.0,
                          xc - xc * cosp + yc * sinp, yc - xc * sinp - yc * cosp, 0.0)
                polygon = affinity.affine_transform(polygon, matrix)
            # get the polygon bounding box
            min_x, min_y, max_x, max_y = polygon.bounds
            length_x = max_x - min_x
            length_y = max_y - min_y
            # ensure the bounding box is square
            if length_x > length_y:
                min_y -= (length_x - length_y) / 2
                max_y += (length_x - length_y) / 2
            else:
                min_x -= (length_y - length_x) / 2
                max_x += (length_y - length_x) / 2
            length = max(length_x, length_y)
            # enlarge the bounding box by 20%
            min_x -= length * 0.1
            min_y -= length * 0.1
            max_x += length * 0.1
            max_y += length * 0.1
            # get transform from the new bounding box to the image
            transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, 224, 224)
            image = rasterio.features.rasterize([polygon], out_shape=(224, 224), transform=transform)
            images[i] = image
        np.savez_compressed(image_out_path, images)
        np.savez_compressed(rotation_out_path, rotations)


    def rasterize_buildings(self, building_list, batch_size=50000, hdf5_chunk_size=128, rotation=True, force=False):
        
        image_out_path = self.out_path + 'building_raster.hdf5'
        rotation_out_path = self.out_path + 'building_rotation.hdf5'

        # Check if buildings have already been rasterized.
        if not force and os.path.exists(image_out_path):
            return # np.load(image_out_path)['arr_0']
            
            
        total_buildings = len(building_list)
        num_batches = (total_buildings + batch_size - 1) // batch_size  # Calculate the number of batches
    
        # Open two HDF5 files, one for images and one for rotations
        with h5py.File(raster_out_path, 'w') as f_raster, h5py.File(rotation_out_path, 'w') as f_rotation:
            
            # Pre-create datasets for images and rotations with compression
            dset_images = f_raster.create_dataset('images', 
                                                  (total_buildings, 224, 224), 
                                                  chunks=(hdf5_chunk_size, 224, 224), 
                                                  dtype='uint8', 
                                                  compression="gzip")
            dset_rotations = f_rotation.create_dataset('rotations', 
                                                       (total_buildings, 2), 
                                                       chunks=(hdf5_chunk_size, 2), 
                                                       dtype='float', 
                                                       compression="gzip")
    
            batch_loader = trange(num_batches, desc='Rasterizing batches of buildings')
            for batch_index in batch_loader:
                start_index = batch_index * batch_size
                end_index = min((batch_index + 1) * batch_size, total_buildings)
                batch_building_list = building_list[start_index:end_index]
        
                # Initialize arrays for this batch
                images = np.zeros((len(batch_building_list), 224, 224), dtype=np.uint8)
                rotations = np.zeros((len(batch_building_list), 2), dtype=float)
                
                for i, building in enumerate(batch_building_list):
                    polygon = building['shape']
                    if rotation:
                        # rotate the polygon to align with the x-axis
                        rectangle = polygon.minimum_rotated_rectangle
                        xc = polygon.centroid.x
                        yc = polygon.centroid.y
                        rec_x = []
                        rec_y = []
                        for point in rectangle.exterior.coords:
                            rec_x.append(point[0])
                            rec_y.append(point[1])
                        top = np.argmax(rec_y)
                        top_left = top - 1 if top > 0 else 3
                        top_right = top + 1 if top < 3 else 0
                        x0, y0 = rec_x[top], rec_y[top]
                        x1, y1 = rec_x[top_left], rec_y[top_left]
                        x2, y2 = rec_x[top_right], rec_y[top_right]
                        d1 = np.linalg.norm([x0 - x1, y0 - y1])
                        d2 = np.linalg.norm([x0 - x2, y0 - y2])
                        if d1 > d2:
                            cosp = (x1 - x0) / d1
                            sinp = (y0 - y1) / d1
                        else:
                            cosp = (x2 - x0) / d2
                            sinp = (y0 - y2) / d2
                        rotations[i] = [cosp, sinp]
                        matrix = (cosp, -sinp, 0.0,
                                  sinp, cosp, 0.0,
                                  0.0, 0.0, 1.0,
                                  xc - xc * cosp + yc * sinp, yc - xc * sinp - yc * cosp, 0.0)
                        polygon = affinity.affine_transform(polygon, matrix)
                    
                    # get the polygon bounding box
                    min_x, min_y, max_x, max_y = polygon.bounds
                    length_x = max_x - min_x
                    length_y = max_y - min_y
                    
                    # ensure the bounding box is square
                    if length_x > length_y:
                        min_y -= (length_x - length_y) / 2
                        max_y += (length_x - length_y) / 2
                    else:
                        min_x -= (length_y - length_x) / 2
                        max_x += (length_y - length_x) / 2
                    length = max(length_x, length_y)
                    
                    # enlarge the bounding box by 20%
                    min_x -= length * 0.1
                    min_y -= length * 0.1
                    max_x += length * 0.1
                    max_y += length * 0.1
                    
                    # get transform from the new bounding box to the image
                    transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, 224, 224)
                    image = rasterio.features.rasterize([polygon], out_shape=(224, 224), transform=transform)
                    images[i] = image
       
                # Write batch data to the HDF5 datasets
                dset_images[start_index:end_index] = images
                dset_rotations[start_index:end_index] = rotations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Singapore', help='city name, can be Singapore or NYC')
    parser.add_argument('--radius', type=float, default=100, help='radius of the Poisson Disk Sampling')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    city = args.city
    radius = args.radius
    assert radius > 50 # Too many sampling points will be too slow

    preprocessor = Preprocess(city)
    building, poi = preprocessor.get_building_and_poi()
    random_point = preprocessor.poisson_disk_sampling(building, poi, radius)
    preprocessor.rasterize_buildings(building)
    preprocessor.partition(building, poi, random_point, radius)
    print(f'Random Points: {len(random_point)}')

