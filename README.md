# Mine-blasting

- install the packages from requirements.txt
- create a folder named 'dataset' & transfer the '.he5' file inside it / use the full path of the 'he.5' file

- the following arguments can be parsed : 

  '--dataset_file', default="dataset/S1_IW23_048_0081_0083_20160412_20230611_N25850_N26000_W080420_W080220_PS.he5", help='the path to dataset file')
  '--reference', default="average", help='reference is set to average alternate is coded for yet')
  '--lat', default=25.928337, type=float, help='latitude of interest')
  '--lon', default=-80.31182, type=float, help='longitude of interest')
  '--max_location', default=500, type=int, help='maximum number of timeseries')
  '--grid_size', default=30, type=int, help='length of the square grid (#pixels)')
  '--method', default='Grid', help='method you want to do search. Available : Grid, Whole, Whole_grid')
  '--output_filename', default='output.csv', help='file name of the output file')


    TODO
    - the time series selected based on a radius needs to be implemented 
