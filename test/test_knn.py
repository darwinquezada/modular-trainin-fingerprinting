import os
import warnings
import pandas as pd
import numpy as np
import joblib
from time import sleep
import logging
from positioning.position import Position_KNN
from preprocessing.data_representation import DataRepresentation
from preprocessing.data_preprocessing import load, new_non_detected_value, data_reshape_sample_timestep_feature
from datetime import datetime

warnings.filterwarnings('ignore')

# Descending
def sort_df(df):
    return pd.DataFrame(
        data=df.columns.values[np.argsort(-df.values, axis=1)],
        columns=[i for i in range(df.shape[1])]
    )



def test_knn(X_train, y_train, X_test, y_test, path_saved_models, results_path):
    
    floor_label_encoder = joblib.load(os.path.join(path_saved_models, 'floor_label_encoder.save'))

    building_label_encoder = joblib.load(os.path.join(path_saved_models, 'building_label_encoder.save'))

    X_test = pd.DataFrame(X_test)

    df_prediction_results = pd.DataFrame(columns=['LONGITUDE', 'LATITUDE', 'ALTITUDE', 'FLOOR', 'BUILDINGID', 'AP', 'ERROR_3D', 'ERROR_2D'])

    position = Position_KNN(k=1, metric='manhattan', weight='distance')

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
                prediction = {'LONGITUDE':0, 'LATITUDE':0, 'ALTITUDE': 0, 'FLOOR':0, 'BUILDINGID':0, 'AP':0, 'ERROR_3D':0, 'ERROR_2D':0}
                df_prediction_results = df_prediction_results.append(prediction, ignore_index=True)
                break

            if os.path.exists(path_model_ap):

                data = joblib.load(os.path.join(path_model_ap, 'data.save'))
                list_features = data['aps']
                mask = data['indexes']
                
                fingerprint = pd.DataFrame(columns = list_features)
                
                measurements = {}
                aps = np.where(X_test.iloc[i,:]>0)

                for j in aps[0]:
                    if j in list_features:
                        measurements[j] = X_test.iloc[i,j]
                
                fingerprint = fingerprint.append( measurements, ignore_index=True).fillna(0)
                
                # Training data
                X_train_selected = pd.DataFrame(X_train).iloc[mask,list_features].reset_index(drop=True).values
                y_train_selected = y_train.iloc[mask,:].reset_index(drop=True).values
                
                position.fit(X_train_selected, y_train_selected)
                prediction_2d, error_2d = position.predict_position_2D(X_test=fingerprint.values, y_test=y_test.iloc[i,:].values)
                prediction_3d, error_3d = position.predict_position_3D(X_test=fingerprint.values, y_test=y_test.iloc[i,:].values)
                prediction_floor = position.floor_hit_rate(X_test=fingerprint.values, y_test=y_test.iloc[i,:].values)
                prediction_building = position.building_hit_rate(X_test=fingerprint.values, y_test=y_test.iloc[i,:].values)

                prediction = {'LONGITUDE':prediction_3d[0][0], 'LATITUDE': prediction_3d[0][1], 'ALTITUDE': prediction_3d[0][2], 
                              'FLOOR':prediction_floor[0], 'BUILDINGID':prediction_building[0], 'AP':ap_valid[cont], 
                              'ERROR_3D': error_3d, 'ERROR_2D':error_2d}
                df_prediction_results = df_prediction_results.append(prediction,  ignore_index=True)
                model_exist = True
            else:
                cont += 1

    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    df_prediction_results.to_csv(os.path.join(results_path, 'results.csv') , index=False)
    mean_3d_error = np.mean(np.linalg.norm(df_prediction_results.iloc[:,0:3].values - y_test.iloc[:,0:3].values, axis=1))
    mean_2d_error = np.mean(np.linalg.norm(df_prediction_results.iloc[:,0:2].values - y_test.iloc[:,0:2].values, axis=1))

    diff_pred_test_floor = np.subtract(df_prediction_results['FLOOR'], y_test['FLOOR'])
    floor_hit_rate = (sum(map(lambda x:x == 0, diff_pred_test_floor))/np.shape(df_prediction_results['FLOOR'])[0])*100

    diff_pred_test_building = np.subtract(df_prediction_results['BUILDINGID'], y_test['BUILDINGID'])
    building_hit_rate = (sum(map(lambda x:x == 0, diff_pred_test_building))/np.shape(df_prediction_results['BUILDINGID'])[0])*100

    datestr = "%m/%d/%Y %I:%M:%S %p"
    
    logging.basicConfig(
        filename=results_path + '/' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '.log',
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        # force=True
    )


    logging.info("---------------------------- KNN ---------------------------")
    logging.info(' Mean 2D positioning error : {:.3f}m'.format(mean_2d_error))
    logging.info(' Mean 3D positioning error : {:.3f}m'.format(mean_3d_error))
    logging.info(' Floor hit rate : {:.3f}%'.format(floor_hit_rate))
    logging.info(' Building hit rate : {:.3f}%'.format(building_hit_rate))
    
    print( "Mean 3D error: {:.3f}".format(mean_3d_error))
    print( "Mean 2D error: {:.3f}".format(mean_2d_error))
    print( "Building hit rate: {:.3f}".format(building_hit_rate))
    print( "Floor hit rate: {:.3f}".format(floor_hit_rate))