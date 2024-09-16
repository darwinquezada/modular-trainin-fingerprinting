import os
import warnings
import pandas as pd
import numpy as np
import joblib
from time import sleep
from preprocessing import data_helper
from preprocessing.data_representation import DataRepresentation
from preprocessing.data_preprocessing import load, new_non_detected_value, data_reshape_sample_timestep_feature
from datetime import datetime
import logging
from tensorflow.keras.models import load_model


warnings.filterwarnings('ignore')

# Descending
def sort_df(df):
    return pd.DataFrame(
        data=df.columns.values[np.argsort(-df.values, axis=1)],
        columns=[i for i in range(df.shape[1])]
    )

def test_cnnloc(y_train, X_test, y_test, path_saved_models, path_results, ap_selection, cnnloc_config):
    
    y_test_original = y_test
            
    # Data transformation
    X_test = data_helper.normalizeX(X_test, cnnloc_config['b'])
    
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
            path_model_ap = os.path.join(path_saved_models, 'AP'+ str(ap_valid[cont]))
            
            if ap_valid[cont] == 10000:
                prediction = {'LONGITUDE':0, 'LATITUDE':0, 'ALTITUDE': 0, 'FLOOR':0, 'BUILDINGID':0, 'AP':0, 'ERROR_3D':0, 'ERROR_2D':0}
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
                
                fingerprint = fingerprint.append(measurements, ignore_index=True).fillna(0)
                
                # building_model = joblib.load(os.path.join(path_model_ap,'building.save'))
                # floor_model = joblib.load(os.path.join(path_model_ap,'floor.save'))
                # position_model = joblib.load(os.path.join(path_model_ap,'positioning.save'))

                building_model = load_model(os.path.join(path_model_ap,'building.tf'))
                floor_model = load_model(os.path.join(path_model_ap,'floor.tf'))
                position_model = load_model(os.path.join(path_model_ap,'positioning.tf'))
                
                mask = joblib.load(os.path.join(path_model_ap,'mask_samples.save'))
                
                normY = data_helper.NormY()
                normY.fit(y_train[mask, 0], y_train[mask, 1], y_train[mask, 2])

                # Predict longitude, latitude and altitude
                predict_Location= position_model.predict(fingerprint)
                longitude, latitude, altitude = normY.reverse_normalizeY(predict_Location[:, 0],
                                                                         predict_Location[:, 1],
                                                                         predict_Location[:, 2])
                if np.isnan(longitude[0]) or np.isnan(latitude[0]) or np.isnan(altitude[0]):
                    print("-------------- Null result with the AP" + str(ap_valid[cont]) + "-----------------")
                    cont += 1
                else:
                    
                    #Loading floor and building label encoding models
                    floor_label_enc= joblib.load(os.path.join(path_model_ap, 'floor_label_encoder.save'))
                    building_label_enc= joblib.load(os.path.join(path_model_ap, 'building_label_encoder.save'))
                    
                    
                    # Floor hit rate
                    floor = floor_model.predict(fingerprint)
                    floor = data_helper.oneHotDecode(floor)
                    floor = floor_label_enc.inverse_transform(floor)

                    # Building hit rate
                    building = building_model.predict(fingerprint)
                    building = data_helper.oneHotDecode(building)
                    building = building_label_enc.inverse_transform(building)
                    
                    error_3d = np.linalg.norm([longitude[0][0], latitude[0][0], altitude[0][0]] - y_test_original[i,0:3])
                    error_2d = np.linalg.norm([longitude[0][0], latitude[0][0]] - y_test_original[i,0:2])

                    # Save the results
                    prediction = {'LONGITUDE':longitude[0][0], 'LATITUDE':latitude[0][0], 'ALTITUDE': altitude[0][0], 
                                    'FLOOR':floor[0], 'BUILDINGID':building[0], 'AP':ap_valid[cont], 'ERROR_3D':error_3d, 
                                    'ERROR_2D':error_2d}

                    df_prediction_results = df_prediction_results.append(prediction, ignore_index=True)

                    model_exist = True
            else:
                cont += 1


    if not os.path.exists(path_results):
        os.makedirs(path_results)
        
    df_prediction_results.to_csv(os.path.join(path_results, 'results_'+ datetime.today().strftime('%Y_%m_%d_%H_%M_%S') +'.csv') , index=False)
    mean_3d_error = np.mean(np.linalg.norm(df_prediction_results.iloc[:,0:3].values - y_test_original[:,0:3], axis=1))
    mean_2d_error = np.mean(np.linalg.norm(df_prediction_results.iloc[:,0:2].values - y_test_original[:,0:2], axis=1))

    diff_pred_test_floor = np.subtract(df_prediction_results['FLOOR'], y_test_original[:,3])
    floor_hit_rate = (sum(map(lambda x:x == 0, diff_pred_test_floor))/np.shape(df_prediction_results['FLOOR'])[0])*100

    diff_pred_test_building = np.subtract(df_prediction_results['BUILDINGID'], y_test_original[:,4])
    building_hit_rate = (sum(map(lambda x:x == 0, diff_pred_test_building))/np.shape(df_prediction_results['BUILDINGID'])[0])*100

    datestr = "%m/%d/%Y %I:%M:%S %p"
    
    logging.basicConfig(
        filename=path_results + '/' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '.log',
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        # force=True
    )

    logging.info("---------------------------- CNNLoc ---------------------------")
    logging.info(' Mean 2D positioning error : {:.3f}m'.format(mean_2d_error))
    logging.info(' Mean 3D positioning error : {:.3f}m'.format(mean_3d_error))
    logging.info(' Floor hit rate : {:.3f}%'.format(floor_hit_rate))
    logging.info(' Building hit rate : {:.3f}%'.format(building_hit_rate))
    logging.info(' -----------------------------------------------------------')
    logging.info(' Num. max APs:')
    logging.info(str(ap_selection))
    logging.info(' -----------------------------------------------------------')
    logging.info(' Neural network hyperparameters:')
    logging.info(str(cnnloc_config))
    
    
    print( "Mean 3D error: {:.3f}".format(mean_3d_error))
    print( "Mean 2D error: {:.3f}".format(mean_2d_error))
    print( "Building hit rate: {:.3f}".format(building_hit_rate))
    print( "Floor hit rate: {:.3f}".format(floor_hit_rate))
    
   