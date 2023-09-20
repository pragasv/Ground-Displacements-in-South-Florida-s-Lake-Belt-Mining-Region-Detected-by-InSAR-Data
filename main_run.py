from mintpy.utils import readfile
from utils import run_test, create_sub_folders, find_cordinates_lat_lon
import tensorflow as tf
import argparse


# this code segment runs the model and save the best model

date_list = [
    b'20160412', b'20160705', b'20160927', b'20161009', b'20161015', b'20161021',
    b'20161102', b'20161114', b'20161126', b'20161208', b'20161220', b'20170101',
    b'20170113', b'20170125', b'20170206', b'20170302', b'20170314', b'20170326',
    b'20170407', b'20170513', b'20170525', b'20170618', b'20170724', b'20170817',
    b'20170829', b'20170910', b'20171004', b'20171016', b'20171028', b'20171109',
    b'20171121', b'20171203', b'20171215', b'20180108', b'20180120', b'20180201',
    b'20180213', b'20180225', b'20180309', b'20180321', b'20180402', b'20180414',
    b'20180426', b'20180508', b'20180520', b'20180613', b'20180625', b'20180707',
    b'20180719', b'20180731', b'20180824', b'20180905', b'20180917', b'20180929',
    b'20181011', b'20181023', b'20181104', b'20181116', b'20181128', b'20181210',
    b'20190103', b'20190115', b'20190127', b'20190208', b'20190220', b'20190304',
    b'20190316', b'20190328', b'20190409', b'20190421', b'20190503', b'20190515',
    b'20190527', b'20190608', b'20190620', b'20190702', b'20190714', b'20190726',
    b'20190807', b'20190819', b'20190831', b'20190906', b'20190912', b'20190924',
    b'20191006', b'20191018', b'20191030', b'20191111', b'20191123', b'20191205',
    b'20191217', b'20191229', b'20200110', b'20200122', b'20200203', b'20200215',
    b'20200227', b'20200310', b'20200322', b'20200403', b'20200415', b'20200427',
    b'20200509', b'20200521', b'20200602', b'20200614', b'20200626', b'20200708',
    b'20200720', b'20200801', b'20200813', b'20200825', b'20200906', b'20200930',
    b'20201012', b'20201024', b'20201105', b'20201129', b'20201211', b'20201223',
    b'20210104', b'20210128', b'20210209', b'20210221', b'20210305', b'20210317',
    b'20210329', b'20210410', b'20210422', b'20210504', b'20210516', b'20210528',
    b'20210609', b'20210621', b'20210703', b'20210715', b'20210727', b'20210808',
    b'20210820', b'20210901', b'20210913', b'20210925', b'20211007', b'20211019',
    b'20211112', b'20211124', b'20211206', b'20211218', b'20211230', b'20220111',
    b'20220123', b'20220204', b'20220216', b'20220228', b'20220312', b'20220324',
    b'20220405', b'20220417', b'20220429', b'20220511', b'20220604', b'20220616',
    b'20220628', b'20220710', b'20220722', b'20220803', b'20220815', b'20220827',
    b'20220920', b'20221002', b'20221026', b'20221107', b'20221119', b'20221201',
    b'20221213', b'20221225', b'20230106', b'20230118', b'20230130', b'20230211',
    b'20230223', b'20230307', b'20230319', b'20230331', b'20230412', b'20230424',
    b'20230506', b'20230611']

date_list = [element.decode() for element in date_list]

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