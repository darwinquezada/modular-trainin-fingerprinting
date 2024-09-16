from time import sleep
from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from models.mlp import MLP
from miscellaneous.misc import Misc
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from statistics import mode
from test.test_mlp import test_mlp
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


# Descending
def sort_df(df):
    return pd.DataFrame(
        data=df.columns.values[np.argsort(-df.values, axis=1)],
        columns=[i for i in range(df.shape[1])]
    )


def match_function(row, position, dataframe, num_max_rss, m_percent):
    intersect = dataframe.apply(lambda x: list(set(x).intersection(set(row))),axis=1)
    # intersect[position] = []
    intersect = list(map(lambda x: list(filter(lambda y: y > 0,x)), intersect))
    match_percent = list(map(lambda x: 0 if (len(x)*100)/num_max_rss < m_percent else (len(x)*100)/num_max_rss, intersect))
    return match_percent

def run_mlp(dataset_name=None, ap_selection=None, path_config=None, dataset_config=None, mlp_config=None):
    
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
    path_data_model = os.path.join(path_config['saved_model'], dataset_name, 'MLP', str(ap_selection['num_aps']))
    if not os.path.exists(path_data_model):
        os.makedirs(path_data_model)
    
    y_train_temp = y_train.copy()
         
    # Full y_data
    y_data = [y_train, y_test]
    y_data = pd.concat(y_data)
    
    df_x_train = pd.DataFrame(X_train)

    # Sorting the dataset (Descending)
    sorted_df = np.sort(df_x_train)[:, ::-1]
    sorted_df = pd.DataFrame(np.where(sorted_df > 0, 1, -1))
    temp = sort_df(df_x_train).replace(0, df_x_train.shape[1] + 1)
    temp2 = temp.mul(sorted_df).values
    temp2 = pd.DataFrame(np.where(temp2 < 0, 0, temp2)).iloc[:, 0:int(ap_selection['num_aps'])]

    if np.shape(temp2)[1] > 1:
        unique_ap_list = pd.DataFrame(temp2.iloc[:,0].drop_duplicates().reset_index(drop=True))
    else:
        unique_ap_list = temp2.drop_duplicates().reset_index(drop=True)

    # Create groups
    df_match = pd.DataFrame()
    for i in range(0, np.shape(unique_ap_list)[0]):
        ap = unique_ap_list[0]
        path_save_info_ap = os.path.join(path_data_model, 'AP' + str(ap[i]))
        
        if not os.path.exists(path_save_info_ap):
            os.makedirs(path_save_info_ap)
            
            df_match = match_function(unique_ap_list.iloc[i, :], i, temp2, ap_selection['num_aps'], 1)
            mask = [j for j in range(len(df_match)) if df_match[j] != 0]
    
            # Selecting data to train
            selected_y_train = y_train_temp.iloc[mask,:].copy()
            selected_x_train = df_x_train.iloc[mask,:].copy()
            selected_x_train = selected_x_train.loc[:, (selected_x_train != 0).any(axis=0)]
            
            # Save APs in view
            aps_used = selected_x_train.columns.values
            
            # Save list of features of the selected fingerprints
            joblib.dump(aps_used, os.path.join(path_save_info_ap, 'list_features.save'))
            
            # Data normalization (X train)
            if mlp_config['x_normalization'] != "":
                norm_model, selected_x_train = misc.normalization(mlp_config['x_normalization'], selected_x_train)
               
                joblib.dump(norm_model, os.path.join(path_save_info_ap,
                                                    'x_normalization_'+mlp_config['x_normalization']+'.save'))
            
            # Label encoding floor
            encoding_floor = LabelEncoder()
            floor_train_selected = encoding_floor.fit_transform(selected_y_train['FLOOR'])
            floor_train_selected = floor_train_selected.reshape(-1, 1)

            joblib.dump(encoding_floor, os.path.join(path_save_info_ap, 'floor_label_encoder.save'))
            
            # One hot encoder floor
            floor_ohe_model = OneHotEncoder(sparse=False)
            floor_one_hot_encoder = floor_ohe_model.fit_transform(floor_train_selected)
            joblib.dump(floor_ohe_model, os.path.join(path_save_info_ap, 'floor_ohe_encoder.save'))
            
            print("----------------------"+ str(ap[i]) +"---------------------------")
            print(str(np.shape(selected_x_train)))
            
            # Train model
            mlp_model = MLP(X_data=selected_x_train, y_data=floor_one_hot_encoder,floor_config=mlp_config)
            floor_model = mlp_model.train()
            # Save model
            floor_model.save(os.path.join(path_save_info_ap, 'floor.h5'))
        else:
            pass

    path_results = os.path.join(path_config['results'], dataset_name, 'MLP', str(ap_selection['num_aps']))
    test_mlp(X_test, y_test, path_data_model, path_results, ap_selection, mlp_config)
    print("END")
    # return True
