import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import LearningRateScheduler
from scipy import stats
from itertools import combinations
import h5py
import time

from mintpy.utils import readfile


# Generated training sequences for use in the model.
TIME_STEPS = 20

def fetch_date_list(dataset_file, dataset_name='HDFEOS/GRIDS/timeseries/observation/date'):
    
    # Open the HDF5 file
    with h5py.File(dataset_file, 'r') as hf:
        # Check if the dataset exists in the file
        if dataset_name in hf:
            # Access the dataset and retrieve its data as a NumPy array
            dataset = hf[dataset_name]
            dataset_data = dataset[()]

            # Now, dataset_data contains the data from the specified dataset
            print(dataset_data)
        else:
            print(f"Dataset '{dataset_name}' not found in the HDF5 file.")

    date_list = [element.decode() for element in list(dataset_data)]

    return date_list

def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def fetch_dataset_preprocess(dataset_file, date_list, reference='first_colomn', test_setting='Grid', x_cord_init=750, y_cord_init=3500, max_loction_count = 500, grid_size_val=30):
    RANDOM_PICK = False

    mask_data, meta_mask = readfile.read(dataset_file, datasetName="HDFEOS/GRIDS/timeseries/quality/mask")
    ts_data, meta = readfile.read(dataset_file)

    df_mine_blasting = pd.DataFrame()

    if test_setting == 'Grid':
        x_cord = x_cord_init
        y_cord = y_cord_init

        x_cord_max = x_cord + grid_size_val
        y_cord_max = y_cord + grid_size_val

        max_loction_count = max_loction_count
    elif test_setting == 'Whole':
        x_cord = int(973/2)
        y_cord = int(4307/2)

        x_cord_max = 973
        y_cord_max = 4307

        max_loction_count = 50000
    else:
        raise ValueError("unknown test setting")
    count = 0 

    chosen_combinations = set()
    while df_mine_blasting.shape[1]<max_loction_count:
        if RANDOM_PICK:
            import random 

            x_cord = random.randint(x_cord_init, x_cord_max)
            y_cord = random.randint(y_cord_init, y_cord_max)

            try:
                is_empty = mask_data[x_cord,y_cord]
            except IndexError:
                raise("index error over here-mask data")

            if not is_empty:
                try:
                    loc_timeseries = ts_data[:,x_cord,y_cord]
                except IndexError:
                    raise("index error over here- timeseries")

                if not np.all(loc_timeseries==0):
                    if (x_cord, y_cord) not in chosen_combinations:
                        chosen_combinations.add((x_cord, y_cord))
                        df_mine_blasting['values_%d_%d'%(x_cord, y_cord)] = loc_timeseries
                        # print(df_mine_blasting.shape[1])
                else:
                    count +=1
            
        else:
            try:
                is_empty = mask_data[x_cord,y_cord]
            except IndexError:
                raise("index error over here-mask data")

            if not is_empty:
                try:
                    loc_timeseries = ts_data[:,x_cord,y_cord]
                except IndexError:
                    raise("index error over here- timeseries")

                if not np.all(loc_timeseries==0):
                    df_mine_blasting['values_%d_%d'%(x_cord, y_cord)] = loc_timeseries
                    # print(df_mine_blasting.shape[1])
                else:
                    count +=1 
            
            if x_cord == x_cord_max - 1 and y_cord == y_cord_max - 1:
                # stop of both x_cord and y_cord have reached cap 
                print("grid cap reached")
                break
            elif y_cord != y_cord_max - 1:
                y_cord += 1 # reset x_cord to zero
            elif y_cord == y_cord_max - 1:
                x_cord += 1 # increase y_cord by 1
                y_cord = y_cord_init
            else:
                raise ValueError("coming into the else case")
        
    # normalize the data 
    df_mine_blasting_values=(df_mine_blasting-df_mine_blasting.mean())/df_mine_blasting.std()

    print("Number of training samples:", len(df_mine_blasting_values))

    if reference=='first_colomn':
        reference_column = df_mine_blasting_values.columns[0]
        df_mine_blasting_values_reference_adjusted = df_mine_blasting_values.subtract(df_mine_blasting_values[reference_column], axis=0)
    elif reference=='average':
        df_average = df_mine_blasting_values.mean(axis=1)
        df_mine_blasting_values_reference_adjusted = df_mine_blasting_values.subtract(df_average, axis=0)
    else:
        raise ValueError("unknown method")

    x_train = create_sequences(df_mine_blasting_values_reference_adjusted.values)
    print("Training input shape: ", x_train.shape)

    return x_train, df_mine_blasting_values_reference_adjusted


def create_model(x_train, activation="relu", learning_rate=0.001, dropout_rate=0.2, layer_1_size=256, layer_2_size=128, final_filter_size=500, bottleneck_layer_size=128 ):
    model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=layer_1_size, kernel_size=7, padding="same", strides=2, activation=activation
        ),
        layers.Dropout(rate=dropout_rate),
        layers.Conv1D(
            filters=layer_2_size, kernel_size=7, padding="same", strides=2, activation=activation
        ),
        ### boottle neck layer
        layers.Conv1D(
            filters=bottleneck_layer_size, kernel_size=7, padding="same", strides=1, activation=activation
        ),
        layers.Conv1DTranspose(
            filters=layer_2_size, kernel_size=7, padding="same", strides=2, activation=activation
        ),
        layers.Dropout(rate=dropout_rate),
        layers.Conv1DTranspose(
            filters=layer_1_size, kernel_size=7, padding="same", strides=2, activation=activation
        ),
        layers.Conv1DTranspose(filters=final_filter_size, kernel_size=7, padding="same"),
    ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.summary()

    return model

def lr_schedule(epoch, lr):
    if epoch % 50 == 0 and epoch > 0:
        # every 50 epoch drop the LR to 0.1 
        lr = lr * 0.1
    return lr

def train_model(model, x_train, patience, test_name):
    history = model.fit(
        x_train,
        x_train,
        epochs=250,
        batch_size=128,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min")
        ],
    )

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("epoch count")
    plt.ylabel("mean squared error")
    plt.legend()
    figure_path = '%s/mean_squared_error_vs_epoch.png' % test_name
    plt.savefig(figure_path)

    # Get train MAE loss.
    x_train_pred = model.predict(x_train)
    all_train_loss = x_train_pred - x_train
    train_mae_loss = np.mean(np.abs(all_train_loss), axis=1)
    

    # train_mae_loss_reshaped = train_mae_loss.reshape((-1))

    plt.hist(train_mae_loss[:,1], bins=50)
    plt.xlabel("Train MAE loss")
    plt.ylabel("No of samples")
    figure_path = '%s/train_MAE_loss_count.png' % test_name
    plt.savefig(figure_path)

    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    print("Reconstruction error threshold: ", threshold)


    n, bins, patches = plt.hist(train_mae_loss.reshape((-1)), bins=50)
    plt.xlabel("test MAE loss")
    plt.ylabel("No of samples")
    figure_path = '%s/test_MAE_loss_count.png' % test_name
    plt.savefig(figure_path)

    # cumulative_sum = np.cumsum(n) / np.sum(n)
    # # Find the bin corresponding to the top 5% value
    # top_5_percent_bin = np.argmax(cumulative_sum >= 0.99)

    # # Get the bin boundaries
    # bin_lower = bins[top_5_percent_bin]
    # bin_upper = bins[top_5_percent_bin + 1]

    # print("threshold : ", bin_upper)

    # # Detect all the samples which are anomalies.
    # threshold = bin_upper  # pick top 10% instead of this
    # anomalies = train_mae_loss > threshold
    # print("Number of anomaly samples: ", np.sum(anomalies))
    # print("Indices of anomaly samples: ", np.where(anomalies))

    return train_mae_loss

def generate_anomalies(train_mae_loss, cum_sum_threshold=0.99):
    n, bins, patches = plt.hist(train_mae_loss.reshape((-1)), bins=50)

    cumulative_sum = np.cumsum(n) / np.sum(n)
    # Find the bin corresponding to the top 5% value
    top_5_percent_bin = np.argmax(cumulative_sum >= cum_sum_threshold)

    # Get the bin boundaries
    bin_lower = bins[top_5_percent_bin]
    bin_upper = bins[top_5_percent_bin + 1]

    print("threshold : ", bin_upper)

    # Detect all the samples which are anomalies.
    threshold = bin_upper  # pick top 10% instead of this
    anomalies = train_mae_loss > threshold
    print("Number of anomaly samples: ", np.sum(anomalies))
    print("Indices of anomaly samples: ", np.where(anomalies))

    return anomalies


def create_model_and_train(x_train, patience=15, test_name='sample'):

    model = create_model(x_train)

    # history = model.fit(
    #     x_train,
    #     x_train,
    #     epochs=250,
    #     batch_size=128,
    #     validation_split=0.2,
    #     callbacks=[
    #         keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    #     ],
    # )

    # plt.plot(history.history["loss"], label="Training Loss")
    # plt.plot(history.history["val_loss"], label="Validation Loss")
    # plt.xlabel("epoch count")
    # plt.ylabel("mean squared error")
    # plt.legend()
    # figure_path = '%s/mean_squared_error_vs_epoch.png' % test_name
    # plt.savefig(figure_path)

    # # Get train MAE loss.
    # x_train_pred = model.predict(x_train)
    # train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

    # # train_mae_loss_reshaped = train_mae_loss.reshape((-1))

    # plt.hist(train_mae_loss[:,1], bins=50)
    # plt.xlabel("Train MAE loss")
    # plt.ylabel("No of samples")
    # figure_path = '%s/train_MAE_loss_count.png' % test_name
    # plt.savefig(figure_path)

    # # Get reconstruction loss threshold.
    # threshold = np.max(train_mae_loss)
    # print("Reconstruction error threshold: ", threshold)


    # n, bins, patches = plt.hist(train_mae_loss.reshape((-1)), bins=50)
    # plt.xlabel("test MAE loss")
    # plt.ylabel("No of samples")
    # figure_path = '%s/test_MAE_loss_count.png' % test_name
    # plt.savefig(figure_path)

    # cumulative_sum = np.cumsum(n) / np.sum(n)
    # # Find the bin corresponding to the top 5% value
    # top_5_percent_bin = np.argmax(cumulative_sum >= 0.99)

    # # Get the bin boundaries
    # bin_lower = bins[top_5_percent_bin]
    # bin_upper = bins[top_5_percent_bin + 1]

    # print("threshold : ", bin_upper)

    # # Detect all the samples which are anomalies.
    # threshold = bin_upper  # pick top 10% instead of this
    # anomalies = train_mae_loss > threshold
    # print("Number of anomaly samples: ", np.sum(anomalies))
    # print("Indices of anomaly samples: ", np.where(anomalies))
    anomalies = train_model(model, x_train, patience, test_name)

    return anomalies


def run_grid_search(X_train, location_count=500):
    # Create a Keras classifier wrapper for Scikit-learn
    model = KerasRegressor(build_fn=create_model, x_train=X_train, verbose=0)

    # Define the hyperparameter grid
    param_grid = {
        
        'activation': ['relu', 'sigmoid'], 
        'dropout_rate': [0.1,0.2,0.5],
        'layer_1_size': [128,256,512], 
        'layer_2_size': [64,128,256], 
        'final_filter_size': [location_count] 
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X_train, X_train, callbacks=[LearningRateScheduler(lr_schedule)])
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


    return create_model(X_train, 
                        dropout_rate=grid_result.best_params_['dropout_rate'], 
                        layer_1_size=grid_result.best_params_['layer_1_size'], 
                        layer_2_size=grid_result.best_params_['layer_2_size'],
                        )


def scatter_plot_anomalies(df_mine_blasting_values, anomalies, latitude, longitude, date_list, test_name, suffix="", str_latlon="", anomaly_thresh=0.99):
    from datetime import datetime

    df_output = []

    count_list = []

    for column_idx in range(len(df_mine_blasting_values.columns)):
        column_name =  df_mine_blasting_values.columns[column_idx]

        # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
        anomalous_data_indices = []
        for data_idx in range(TIME_STEPS - 1, len(df_mine_blasting_values) - TIME_STEPS + 1):
            anomaly_count = np.sum(anomalies[data_idx - TIME_STEPS + 1 : data_idx, column_idx])
            if anomaly_count != 0:
                count_list.append(anomaly_count)
            if anomaly_count > 10:
                anomalous_data_indices.append(data_idx)

        dates = [datetime.strptime(date, '%Y%m%d') for date in date_list]
        df_mine_blasting_values["date"] = dates
        df_mine_blasting_values.set_index('date', inplace=True)

        df_subset = df_mine_blasting_values[column_name].iloc[anomalous_data_indices]
        df_subset_base = df_mine_blasting_values[column_name].iloc[anomalous_data_indices]
        split_string = column_name.split('_')
        last_two_numbers = split_string[-2:]
        last_two_numbers = [int(num) for num in last_two_numbers]

        lat_val = latitude[last_two_numbers[0],last_two_numbers[1]]
        lon_val = longitude[last_two_numbers[0],last_two_numbers[1]]

        if len(df_subset)>0:
       
            plt.figure(figsize=(18, 8))
            plt.scatter(df_mine_blasting_values.index, df_mine_blasting_values[column_name])
            plt.scatter(df_subset.index, df_subset.values, color="r")
            plt.xlabel("datetime")
            plt.ylabel("standardized LOS")
            plt.title("anomally detected in : %s & lat:%f long:%f" % (column_name,lat_val, lon_val))
            plt.xticks(rotation=45)
            plt.legend(["time series", "anomaly region"])
            figure_path = '%s/anomaly/standardized LOS lat:%f long:%f.png' % (test_name,lat_val, lon_val)
            plt.savefig(figure_path)

            df_output.append({
                'TS': [df_mine_blasting_values[column_name].values],
                'start_anomaly': df_subset.index[0],
                'anomaly sample length': len(df_subset),
                'prob_anomaly': anomaly_thresh,  # this value is hard coded for now 
                'lat': lat_val,
                'lon': lon_val,
            })
        else:
            # no anomaly in this time series 
            df_output.append({
                'TS': df_mine_blasting_values[column_name],
                'start_anomaly': 0,
                'window length': 0,
                'prob_anomaly': 0, 
                'lat': lat_val,
                'lon': lon_val,
            })

    amended_output_file_name1 = "output_%d"%(anomaly_thresh*100) + suffix + str_latlon + '.csv' 
    print('output_file:', amended_output_file_name1)
    
    df_output = pd.DataFrame(df_output)
    df_output.to_csv(amended_output_file_name1)

    ## preprocess output for plotting 
    list_needed_columns = ['start_anomaly', 'anomaly sample length', 'prob_anomaly', 'lat', 'lon']

    df_preprocessed = df_output[list_needed_columns]
    amended_output_file_name2 = "preproc_random_%d"%(anomaly_thresh*100) + suffix + str_latlon + '.csv' 
    print('preprocessed output_file1:', amended_output_file_name2)

    df_preprocessed.to_csv(amended_output_file_name2)

    return count_list

def find_columns_with_true(matrix):
    # Use np.all() along axis=0 to check if all values in each column are True
    column_mask = np.any(matrix, axis=0)
    
    # Find the column indexes where the mask is True
    column_indexes = np.where(column_mask)[0]
    
    return column_indexes

def fetch_anomaly_cordinates(df_mine_blasting_values,anomalies, latitude, longitude):
    column_names = df_mine_blasting_values.columns
    column_indexes = find_columns_with_true(anomalies)

    anomally_locations_names = column_names[column_indexes]

    cordinates_result = []
    lat_long_result = []

    for string in anomally_locations_names:
        split_string = string.split('_')
        last_two_numbers = split_string[-2:]
        last_two_numbers = [int(num) for num in last_two_numbers]
        cordinates_result.append(last_two_numbers)
    
        long_result = longitude[last_two_numbers[0],last_two_numbers[1]]
        lat_result = latitude[last_two_numbers[0],last_two_numbers[1]]

        lat_long_result.append([lat_result, long_result])

    return np.array(lat_long_result)


def create_sub_folders(test_folder_name):

    if not os.path.exists(test_folder_name):
        os.makedirs(test_folder_name)
        os.makedirs(test_folder_name+'/anomaly')
        print(f"Folder '{test_folder_name}' created.")
    else:
        print(f"Folder '{test_folder_name}' already exists.")


def create_non_overlapping_grids(x_range, y_range, grid_size):
    grid_width, grid_height = grid_size
    x_min, x_max = x_range
    y_min, y_max = y_range

    grids = []
    x = x_min
    while x + grid_width <= x_max:
        y = y_min
        while y + grid_height <= y_max:
            grid = (x, y)  # Save only the starting x and y coordinates
            grids.append(grid)
            y += grid_height
        x += grid_width

    return grids


def find_cordinates_lat_lon(latitude, longitude, given_lat, given_lon):
    """
    This function is used to fetch the cordinate of a given lat or long value
    """
    indices = np.where((latitude == given_lat) & (longitude == given_lon))
    if indices[0].size > 0:
        x, y = indices[0][0], indices[1][0]
        print(f"Latitude: {given_lat}, Longitude: {given_lon} corresponds to x={x}, y={y}")
    else:
        raise ValueError("Coordinates not found in the arrays. Please provide a cordinate we have the data to. eg latitude == 25.928337 longitude == -80.31182")
    return x,y


def pairwise_t_tests(model_accuracies_array):
    num_models = model_accuracies_array.shape[0]
    p_values = np.zeros((num_models, num_models))

    for i, j in combinations(range(num_models), 2):
        _, p_value = stats.ttest_ind(model_accuracies_array[i], model_accuracies_array[j])
        p_values[i, j] = p_value

    return p_values

def run_test(dataset_file, date_list, reference, latitude, longitude, test_name, test_setting, x_cord_start=750, y_cord_start=3500,location_count=500, grid_size_val=30, suffix='', str_latlon=''):
    # first model was trained on : x_cord=750, y_cord=3500
    if test_setting == "Whole_grid":
        x_range = (0, 973)
        y_range = (0, 4307)
        grid_size = (grid_size_val, grid_size_val)

        grids = create_non_overlapping_grids(x_range, y_range, grid_size)
        model_AE_array = []
        model_MAE_array = []

        anomaly_thresh = 99
        count_list = scatter_plot_anomalies(df_mine_blasting_values_reference_adjusted, anomalies, latitude, longitude,
                                            date_list, test_name, suffix=suffix, str_latlon=str_latlon, anomaly_thresh=anomaly_thresh)

        for grid_cords in grids:
            x_train, df_mine_blasting_values_reference_adjusted = fetch_dataset_preprocess(dataset_file, date_list, reference=reference, 
                                            test_setting='Grid', x_cord_init=grid_cords[0], y_cord_init=grid_cords[1], max_loction_count=location_count, grid_size_val=grid_size_val)
            # best_model = run_grid_search(x_train)
            best_model = create_model(x_train, 
                            activation="relu", 
                            learning_rate=0.001, 
                            dropout_rate=0.1, 
                            layer_1_size=512, 
                            layer_2_size=1024, 
                            final_filter_size=location_count )
            all_train_loss = train_model(best_model, x_train, patience=50, test_name=test_name)
            anomalies = generate_anomalies(all_train_loss, 0.95)
            model_AE_array.append(all_train_loss)
            model_MAE_array.append(np.mean(np.abs(all_train_loss), axis=1))
            # anomalies = create_model_and_train(x_train, patience=50, test_name)
            count_list = scatter_plot_anomalies(df_mine_blasting_values_reference_adjusted, anomalies, latitude, longitude,date_list, test_name)

        f_statistic, p_value = stats.f_oneway(*model_MAE_array)
        pairwise_p_values = pairwise_t_tests(np.array(model_MAE_array))

        
        plt.hist(model_MAE_array, bins=50, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
        # Add labels and title
        plt.xlabel('MSE')
        plt.ylabel('Frequency of MAE')
        plt.title('Histogram of MAE error on fit')
        # Save the histogram as an image (replace 'histogram.png' with your desired file name)
        plt.savefig('histogram.png')

        alpha = 0.05
        if p_value < alpha:
            print("There are significant differences in model performance.")
        else:
            print("No significant differences in model performance.")


    else:
        x_train, df_mine_blasting_values_reference_adjusted = fetch_dataset_preprocess(dataset_file, date_list, reference=reference, test_setting=test_setting, 
                                       x_cord_init=x_cord_start, y_cord_init=y_cord_start, max_loction_count=location_count, grid_size_val=grid_size_val)
        
        best_model = run_grid_search(x_train, location_count=location_count)
        train_mae_loss = train_model(best_model, x_train, patience=50, test_name=test_name)
        
        for anomaly_thresh in [0.90, 0.95, 0.99]:
    
            anomalies = generate_anomalies(train_mae_loss, anomaly_thresh)
            
            # anomalies = create_model_and_train(x_train, patience=50, test_name)
            count_list = scatter_plot_anomalies(df_mine_blasting_values_reference_adjusted, anomalies, latitude, longitude,
                               date_list, test_name, suffix=suffix, str_latlon=str_latlon, anomaly_thresh=anomaly_thresh)

