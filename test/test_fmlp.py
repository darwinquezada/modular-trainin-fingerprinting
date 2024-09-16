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


def test_fmlp(X_test, y_test, path_saved_models, path_results, ap_selection, mlp_config):
    
    X_test = pd.DataFrame(X_test)
    y_test_original = y_test

    df_prediction_results = pd.DataFrame(columns=[ 'FLOOR' ])
    
    if mlp_config['x_normalization'] != "":
        norm_model = joblib.load(os.path.join(path_saved_models,'x_normalization_'+mlp_config['x_normalization']+'.save'))
        X_test = norm_model.transform(X_test)
                
    floor_model = load_model(os.path.join(path_saved_models,'floor.h5'))
                
    #Loading floor and building label encoding models
    floor_label_enc= joblib.load(os.path.join(path_saved_models, 'floor_label_encoder.save'))
                
    # Floor hit rate
    floor = floor_model.predict(X_test)
    floor = np.argmax(floor, axis=-1)
    floor = floor_label_enc.inverse_transform(floor)
    
    df_prediction_results = pd.DataFrame(floor, columns = ['FLOOR']) 
    
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

    logging.info("---------------------------- FMLP ---------------------------")
    logging.info(' Floor hit rate : {:.3f}%'.format(floor_hit_rate))
    logging.info(' -----------------------------------------------------------')
    logging.info(' Num. max APs:')
    logging.info(str(ap_selection))
    logging.info(' -----------------------------------------------------------')
    logging.info(' Neural network hyperparameters:')
    logging.info(str(mlp_config))
    

    print( "Floor hit rate: {:.3f}".format(floor_hit_rate))
    
   