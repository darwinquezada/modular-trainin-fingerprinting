from time import sleep
from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from models.mlp import MLP
from miscellaneous.misc import Misc
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from statistics import mode
from test.test_fmlp import test_fmlp
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
### Warning ###
import warnings
warnings.filterwarnings('ignore')


def run_fmlp(dataset_name=None, ap_selection=None, path_config=None, dataset_config=None, mlp_config=None):
    
    '''
    This function run all the machine learning models CNN-LSTM for building classification, floor classification,
    position prediction, and GAN
    :param algorithm:
    :param path_config:
    :param dataset_name:
    :param dataset:
    :param dataset_config:
    :param building_config:
    :param floor_config:
    :param positioning_config:
    :param gan_full_config:
    :param data_augmentation:
    :return:
    '''

    misc = Misc()
    
    # Path initializing
    dataset_path = os.path.join(path_config['data_source'], dataset_name)
    
    if bool(dataset_config['train_dataset']):
        X_train, y_train = load(os.path.join(dataset_path, dataset_config['train_dataset']))

    if bool(dataset_config['test_dataset']):
        X_test, y_test = load(os.path.join(dataset_path, dataset_config['test_dataset']))

    if bool(dataset_config['validation_dataset']):
        X_valid, y_valid = load(os.path.join(dataset_path, dataset_config['validation_dataset']))
    else:
        X_valid = []
        y_valid = []
        
    # Changing the data representation
    new_non_det_val = new_non_detected_value(X_train, X_test, X_valid)
    dr = DataRepresentation(x_train=X_train, x_test=X_test, x_valid=X_valid,
                            type_rep='positive',
                            def_no_val=dataset_config['default_null_value'],
                            new_no_val=new_non_det_val)
    X_train, X_test, X_valid = dr.data_rep()

    # Path to save the models
    path_data_model = os.path.join(path_config['saved_model'], dataset_name, 'FMLP')
    if not os.path.exists(path_data_model):
        os.makedirs(path_data_model)
    
        y_train_temp = y_train.copy()
            
        # Full y_data
        y_data = [y_train, y_test]
        y_data = pd.concat(y_data)
        
        df_x_train = pd.DataFrame(X_train)
        
        # Data normalization (X train)
        if mlp_config['x_normalization'] != "":
            norm_model, x_train = misc.normalization(mlp_config['x_normalization'], df_x_train)
                
            joblib.dump(norm_model, os.path.join(path_data_model,
                                                'x_normalization_'+mlp_config['x_normalization']+'.save'))
                
        # Label encoding floor
        encoding_floor = LabelEncoder()
        floor_train_selected = encoding_floor.fit_transform(y_data['FLOOR'])
        y_train_floor = encoding_floor.transform(y_train['FLOOR'])
        y_train_floor = y_train_floor.reshape(-1, 1)

        joblib.dump(encoding_floor, os.path.join(path_data_model, 'floor_label_encoder.save'))
                
        # One hot encoder floor
        floor_ohe_model = OneHotEncoder(sparse=False)
        floor_one_hot_encoder = floor_ohe_model.fit_transform(y_train_floor)
        joblib.dump(floor_ohe_model, os.path.join(path_data_model, 'floor_ohe_encoder.save'))
                
        # Train model
        mlp_model = MLP(X_data=x_train, y_data=floor_one_hot_encoder,floor_config=mlp_config)
        floor_model = mlp_model.train()
        # Save model
        floor_model.save(os.path.join(path_data_model, 'floor.h5'))
    
    # Results
    path_results = os.path.join(path_config['results'], dataset_name, 'FMLP')
    test_fmlp(X_test, y_test, path_data_model, path_results, ap_selection, mlp_config)
    
    print("END")