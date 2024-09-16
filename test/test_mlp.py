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

def test_mlp(X_test, y_test, path_saved_models, path_results, ap_selection, mlp_config):
    
    X_test = pd.DataFrame(X_test)
    y_test_original = y_test

    df_prediction_results = pd.DataFrame(columns=[ 'FLOOR', 'AP',])

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
                prediction = {'FLOOR':0, 'AP':0}
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
                
                if mlp_config['x_normalization'] != "":
                    norm_model = joblib.load(os.path.join(path_model_ap,'x_normalization_'+mlp_config['x_normalization']+'.save'))
                    fingerprint = norm_model.transform(fingerprint)
                
                floor_model = load_model(os.path.join(path_model_ap,'floor.h5'))
                
                #Loading floor and building label encoding models
                floor_label_enc= joblib.load(os.path.join(path_model_ap, 'floor_label_encoder.save'))
                
                # Floor hit rate
                floor = floor_model.predict(fingerprint)
                floor = data_helper.oneHotDecode(floor)
                floor = floor_label_enc.inverse_transform(floor)
                
                if np.isnan(floor[0]):
                    print("-------------- Null result with the AP" + str(ap_valid[cont]) + "-----------------")
                    cont += 1
                else:
                    # Save the results
                    prediction = { 'FLOOR':floor[0],  'AP':ap_valid[cont]}
                    df_prediction_results = df_prediction_results.append(prediction, ignore_index=True)
                    model_exist = True
            else:
                cont += 1


    if not os.path.exists(path_results):
        os.makedirs(path_results)
    
    df_prediction_results.to_csv(os.path.join(path_results, 'results_'+ datetime.today().strftime('%Y_%m_%d_%H_%M_%S') +'.csv') , index=False)   
    diff_pred_test_floor = np.subtract(df_prediction_results['FLOOR'], y_test_original['FLOOR'])
    floor_hit_rate = (sum(map(lambda x:x == 0, diff_pred_test_floor))/np.shape(df_prediction_results['FLOOR'])[0])*100

    datestr = "%m/%d/%Y %I:%M:%S %p"
    
    logging.basicConfig(
        filename=path_results + '/' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '.log',
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        # force=True
    )

    logging.info("---------------------------- CNNLoc ---------------------------")
    logging.info(' Floor hit rate : {:.3f}%'.format(floor_hit_rate))
    logging.info(' -----------------------------------------------------------')
    logging.info(' Num. max APs:')
    logging.info(str(ap_selection))
    logging.info(' -----------------------------------------------------------')
    logging.info(' Neural network hyperparameters:')
    logging.info(str(mlp_config))
    

    print( "Floor hit rate: {:.3f}".format(floor_hit_rate))
    
   