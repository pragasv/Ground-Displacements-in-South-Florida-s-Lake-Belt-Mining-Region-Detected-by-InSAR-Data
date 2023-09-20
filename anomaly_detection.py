from mintpy.utils import readfile
from utils import run_test, create_sub_folders, find_cordinates_lat_lon, fetch_date_list
import tensorflow as tf
import argparse


# this code segment runs the model and save the best model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='anomaly detection on the model')
    parser.add_argument('--dataset_file', default="dataset/S1_IW23_048_0081_0083_20160412_20230611_N25850_N26000_W080420_W080220_PS.he5", help='the path to dataset file')
    parser.add_argument('--reference', default="average", help='reference is set to average alternate is coded for yet')
    parser.add_argument('--lat', default=25.928337, type=float, help='latitude of interest')
    parser.add_argument('--lon', default=-80.31182, type=float, help='longitude of interest')
    parser.add_argument('--max_location', default=500, type=int, help='maximum number of timeseries')
    parser.add_argument('--grid_size', default=30, type=int, help='length of the square grid (#pixels)')
    parser.add_argument('--method', default='Grid', help='method you want to do search. Available : Grid, Whole, Whole_grid')
    parser.add_argument('--output_filename', default='output.csv', help='file name of the output file')

    args = parser.parse_args()

    dataset_file = args.dataset_file
    reference = args.reference
    location_count = args.max_location
    base_lat = args.lat
    base_lon = args.lon
    test_method = args.method
    grid_size = args.grid_size
    output_filename = args.output_filename

    # dataset_file = 'dataset/S1_IW23_048_0081_0083_20160412_20230611_N25850_N26000_W080420_W080220_PS.he5'
    latitude, dict_lat = readfile.read(dataset_file, datasetName="/HDFEOS/GRIDS/timeseries/geometry/latitude")
    longitude, dict_lon = readfile.read(dataset_file, datasetName="/HDFEOS/GRIDS/timeseries/geometry/longitude")
    date_list = fetch_date_list(dataset_file, dataset_name='HDFEOS/GRIDS/timeseries/observation/date')

    x_cord_start, y_cord_start = find_cordinates_lat_lon(latitude, longitude, base_lat, base_lon)

    physical_devices = tf.config.list_physical_devices('GPU')
    print("setup GPU : ", physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    ###############################################
    ############Test the small grid################
    ###############################################
    if test_method=='Grid':

        test_name="test_smallgrid"
        create_sub_folders(test_name)
        run_test(dataset_file, date_list, reference, latitude, longitude, test_name, test_setting='Grid', x_cord_start=x_cord_start, y_cord_start=y_cord_start, location_count=location_count, grid_size_val=grid_size, output_file_name=output_filename)

    elif test_method=='Grid':
    # ###############################################
    # ############Test whole dataset ################
    # ###############################################
        test_name="test_wholedataset"
        create_sub_folders(test_name)
        run_test(dataset_file, date_list, reference, latitude, longitude, test_name, test_setting='Whole')

    elif test_method=='Whole_grid':
    ###############################################
    ############Test whole dataset ################
    ###############################################
        test_name="test_wholedataset_gridwise"
        create_sub_folders(test_name)
        run_test(dataset_file, date_list, reference, latitude, longitude, test_name, test_setting='Whole_grid')
    
    else:
        raise ValueError("Unknwon test method arg passed ! ")