#!/usr/bin/env python3
from mintpy.utils import readfile
from utils import run_test, create_sub_folders, find_cordinates_lat_lon, fetch_date_list
import tensorflow as tf
import argparse
import time

# this code segment runs the model and save the best model

NOTE = """
Todo:
- Let's make the calculation center a positional argument:  25.928337,80.31182 so that we can call:   anomaly_detection.py S1_*.he5   25.928337,-80.31182
- Currently it fails if location does not exist in latitude/longitude. It should just use pixel in the vicinity
- rename test_smallgrid ?
"""

EXAMPLE = """examples:

    anomaly_detection.py S1_IW23_048_0081_0083_20150921_20230915_N25850_N26000_W080420_W080220_PS.he5 --lat 25.928337 --lon -80.31182
         This will create output files with the default names:
             output_99.csv
             preproc_random_99.csv
    
    anomaly_detection.py --dataset_file S1_IW23_048_0081_0083_20150921_20230915_N25850_N26000_W080420_W080220_PS.he5  --lat 25.9695 --lon -80.3739 --suffix EvergladesHS --no-lalo
         Filenames created:
             output_99_EvergladesHS.csv
             preproc_random_99_EvergladesHS.csv

    anomaly_detection.py --dataset_file S1_IW23_048_0081_0083_20150921_20230915_N25850_N26000_W080420_W080220_PS.he5  --lat 25.9695 --lon -80.3739 --suffix EvergladesHS
         Filenames created:
             output_99_EvergladesHS_25.928_-80.312.csv
             preproc_random_90_EvergladesHS_25.928_-80.312.csv

"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Detection of anomalies in InSAR time series',
                                     formatter_class=argparse.RawTextHelpFormatter, epilog=NOTE + '\n' + EXAMPLE)
    parser.add_argument('dataset_file', nargs='?', default=None, help='timeseries file in hdf5eos format (e.g. S1_*.he5 file.\n')
    #parser.add_argument('--dataset_file', default="dataset/S1_IW23_048_0081_0083_20160412_20230611_N25850_N26000_W080420_W080220_PS.he5", help='the path to dataset file')
    parser.add_argument('--reference', default="average", help='reference is set to average alternate is coded for yet')
    parser.add_argument('--lat', default=25.928337, type=float, help='latitude of interest, (default: 25.928337)')
    parser.add_argument('--lon', default=-80.31182, type=float, help='longitude of interest, default: -80.31182)')
    parser.add_argument('--max_location', default=500, type=int, help='maximum number of timeseries')
    parser.add_argument('--grid_size', default=30, type=int, help='length of the square grid (#pixels)')
    parser.add_argument('--method', default='Grid', help='method you want to do search. Available : Grid, Whole, Whole_grid')
    parser.add_argument('--suffix', default='', dest='suffix', type=str, help='suffix for output file name (default='')')

    parser.add_argument('--lalo', dest='lalo_flag', action='store_true', default=True, help='add lat/long (lalo) string to outfile names (Default=True)')
    parser.add_argument('--no-lalo', dest='lalo_flag', action='store_false', help="Don't add lat/long (lalo) string to outfile name.")

    args = parser.parse_args()

    dataset_file = args.dataset_file
    reference = args.reference
    location_count = args.max_location
    base_lat = args.lat
    base_lon = args.lon
    test_method = args.method
    grid_size = args.grid_size
    lalo_flag = args.lalo_flag
    suffix = "_" + args.suffix

    str_latlon="_{:.3f}_{:.3f}".format(base_lat, base_lon)

    print('suffix: ', suffix)
    print('lalo_flag: ', args.lalo_flag)

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

        test_name="test_smallgrid" + str_latlon + suffix
        create_sub_folders(test_name)
        run_test(dataset_file, date_list, reference, latitude, longitude, test_name, test_setting='Grid', x_cord_start=x_cord_start, y_cord_start=y_cord_start, 
                 location_count=location_count, grid_size_val=grid_size, suffix=suffix, str_latlon=str_latlon)

    elif test_method=='Whole':
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
