import os
import warnings
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from time import sleep
from datetime import datetime
import logging
from tensorflow.keras.models import load_model
from preprocessing.data_representation import DataRepresentation
from preprocessing.data_preprocessing import load, new_non_detected_value, data_reshape_sample_timestep_feature

warnings.filterwarnings('ignore')

# Descending
def sort_df(df):
    return pd.DataFrame(
        data=df.columns.values[np.argsort(-df.values, axis=1)],
        columns=[i for i in range(df.shape[1])]
    )

def test_nn(X_test, y_test, path_saved_models, path_results, ap_selection, nn_config):

    y_test_original = y_test
    
    # Load general models

    floor_label_encoder = joblib.load(os.path.join(path_saved_models, 'floor_label_encoder.save'))
    floor_ohe_encoder = joblib.load(os.path.join(path_saved_models, 'floor_ohe_encoder.save'))

    building_label_encoder = joblib.load(os.path.join(path_saved_models, 'building_label_encoder.save'))
    building_ohe_encoder = joblib.load(os.path.join(path_saved_models, 'building_ohe_encoder.save'))

    X_test = pd.DataFrame(X_test)

    df_prediction_results = pd.DataFrame(columns=['LONGITUDE', 'LATITUDE', 'ALTITUDE', 'FLOOR', 'BUILDINGID', 'AP', 'ERROR_3D', 'ERROR_2D'])

    for i in range(0, np.shape(X_test)[0]):
        sorted_fp = np.where(np.sort(X_test.iloc[i,:])[::-1] > 0, 1, -1)
        fp_sorted_ap = np.argsort(X_test.iloc[i,:])
        fp_sorted_ap = fp_sorted_ap[::-1][:]
        ap_valid = np.where(np.multiply(fp_sorted_ap,sorted_fp)>=0, np.multiply(fp_sorted_ap,sorted_fp),10000)

        cont = 0
        model_exist = False
        while model_exist == False:
            path_model_ap = os.path.join(path_saved_models, 'AP'+str(ap_valid[cont]))
            
            if ap_valid[cont] == 10000:
                prediction = {'LONGITUDE':0, 'LATITUDE':0, 'ALTITUDE': 0, 'FLOOR':0, 'BUILDINGID':0,  
                              'AP':0, 'ERROR_3D':0, 'ERROR_2D':0}
                df_prediction_results = df_prediction_results.append(prediction, ignore_index=True)
                break

            if os.path.exists(path_model_ap):
                # Cleaning the tensorflow session
                # tf.keras.backend.clear_session()

                list_features = joblib.load(os.path.join(path_model_ap, 'list_features.save'))
                fingerprint = pd.DataFrame(columns = list_features)
                
                measurements = {}
                aps = np.where(X_test.iloc[i,:]>0)

                for j in aps[0]:
                    if j in list_features:
                        measurements[j] = X_test.iloc[i,j]
                
                fingerprint = fingerprint.append( measurements, ignore_index=True).fillna(0)
                
                latitude_model = joblib.load(os.path.join(path_model_ap, 'latitude_model_MinMaxScaler.save'))
                longitude_model = joblib.load(os.path.join(path_model_ap, 'longitude_model_MinMaxScaler.save'))
                altitude_model = joblib.load(os.path.join(path_model_ap, 'altitude_model_MinMaxScaler.save'))

                building_model =load_model(os.path.join(path_model_ap,'building.h5'))
                floor_model = load_model(os.path.join(path_model_ap,'floor.h5'))
                position_model = load_model(os.path.join(path_model_ap,'positioning.h5'))

                fp_reshaped = fingerprint.values

                # Predict Longitude, Latitude and Altitude
                position_prediction = position_model.predict(fp_reshaped)

                longitude = longitude_model.inverse_transform(position_prediction[:,0].reshape(-1,1))
                latitude  = latitude_model.inverse_transform(position_prediction[:,1].reshape(-1,1))
                altitude  = altitude_model.inverse_transform(position_prediction[:,2].reshape(-1,1))

                # Predict floor
                predict_floor = floor_model.predict(fp_reshaped)
                predict_floor = np.argmax(predict_floor, axis=-1)
                floor = floor_label_encoder.inverse_transform(predict_floor)

                # Predict building
                predict_building = building_model.predict(fp_reshaped)
                predict_building = np.argmax(predict_building, axis=-1)
                building = building_label_encoder.inverse_transform(predict_building)

                error_3d = np.linalg.norm([longitude[0][0], latitude[0][0],altitude[0][0]] - y_test_original.iloc[i,0:3].values)
                error_2d = np.linalg.norm([longitude[0][0], latitude[0][0]] - y_test_original.iloc[i,0:2].values)

                # Save the results
                prediction = {'LONGITUDE':longitude[0][0], 'LATITUDE':latitude[0][0], 'ALTITUDE': altitude[0][0], 
                                'FLOOR':floor[0], 'BUILDINGID':building[0], 'AP': ap_valid[cont], 
                                'ERROR_3D': error_3d, 'ERROR_2D':error_2d}

                df_prediction_results = df_prediction_results.append(prediction, ignore_index=True)
                model_exist = True
            else:
                cont += 1

    if not os.path.exists(path_results):
        os.makedirs(path_results)
        
    df_prediction_results.to_csv(os.path.join(path_results, 'results_'+ datetime.today().strftime('%Y_%m_%d_%H_%M_%S') +'.csv') , index=False)
    mean_3d_error = np.mean(np.linalg.norm(df_prediction_results.iloc[:,0:3].values - y_test_original.iloc[:,0:3].values, axis=1))
    mean_2d_error = np.mean(np.linalg.norm(df_prediction_results.iloc[:,0:2].values - y_test_original.iloc[:,0:2].values, axis=1))

    diff_pred_test_floor = np.subtract(df_prediction_results['FLOOR'], y_test_original['FLOOR'])
    floor_hit_rate = (sum(map(lambda x:x == 0, diff_pred_test_floor))/np.shape(df_prediction_results['FLOOR'])[0])*100

    diff_pred_test_building = np.subtract(df_prediction_results['BUILDINGID'], y_test_original['BUILDINGID'])
    building_hit_rate = (sum(map(lambda x:x == 0, diff_pred_test_building))/np.shape(df_prediction_results['BUILDINGID'])[0])*100

    datestr = "%m/%d/%Y %I:%M:%S %p"
    
    logging.basicConfig(
        filename=path_results + '/' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '.log',
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        # force=True
    )

    logging.info("---------------------------- NN ---------------------------")
    logging.info(' Mean 2D positioning error : {:.3f}m'.format(mean_2d_error))
    logging.info(' Mean 3D positioning error : {:.3f}m'.format(mean_3d_error))
    logging.info(' Floor hit rate : {:.3f}%'.format(floor_hit_rate))
    logging.info(' Building hit rate : {:.3f}%'.format(building_hit_rate))
    logging.info(' -----------------------------------------------------------')
    logging.info(' Num. max APs:')
    logging.info(str(ap_selection))
    logging.info(' -----------------------------------------------------------')
    logging.info(' Neural network hyperparameters:')
    logging.info(str(nn_config))

    print( "Mean 3D error: {:.3f}".format(mean_3d_error))
    print( "Mean 2D error: {:.3f}".format(mean_2d_error))
    print( "Building hit rate: {:.3f}".format(building_hit_rate))
    print( "Floor hit rate: {:.3f}".format(floor_hit_rate))