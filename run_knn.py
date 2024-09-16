from time import sleep
from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from miscellaneous.misc import Misc
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from test.test_knn import test_knn
from statistics import mode
import os
import numpy as np
import pandas as pd
import joblib
import numpy as np
### Warning ###
import warnings
warnings.filterwarnings('ignore')


# Descending
def sort_df(df):
    return pd.DataFrame(
        data=df.columns.values[np.argsort(-df.values, axis=1)],
        columns=[i for i in range(df.shape[1])]
    )
    
def most_common(List):
    return(mode(List))


def match_function(row, position, dataframe, num_max_rss, m_percent):
    intersect = dataframe.apply(lambda x: list(set(x).intersection(set(row))),axis=1)
    # intersect[position] = []
    intersect = list(map(lambda x: list(filter(lambda y: y > 0,x)), intersect))
    match_percent = list(map(lambda x: 0 if (len(x)*100)/num_max_rss < m_percent else (len(x)*100)/num_max_rss, intersect))
    return match_percent

def run_knn(dataset_name=None, ap_selection=None, path_config=None, dataset_config=None):

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
                            type_rep=dataset_config['data_representation'],
                            def_no_val=dataset_config['default_null_value'],
                            new_no_val=new_non_det_val)
    X_train, X_test, X_valid = dr.data_rep()

    # Path to save the models
    path_data_model = os.path.join(path_config['saved_model'], dataset_name, 'KNN')
    if not os.path.exists(path_data_model):
        os.makedirs(path_data_model)
    
    y_train_temp = y_train.copy()
         
    # Data normalization (X train)
    if dataset_config['x_normalization'] != "":
        norm_model, X_train = misc.normalization(dataset_config['x_normalization'],X_train)
        X_test = norm_model.transform(X_test)
        if X_valid:
            X_valid = norm_model.transform(X_valid)
        
        joblib.dump(norm_model, os.path.join(path_data_model,
                                             'x_normalization_'+dataset_config['x_normalization']+'.save'))
    
    # saving norm model
    if dataset_config['y_normalization'] != "":
        longitude_model, long_train = misc.normalization(dataset_config['y_normalization'],
                                                              y_train['LONGITUDE'].values.reshape(-1, 1))
        latitude_model, lat_train = misc.normalization(dataset_config['y_normalization'],
                                                              y_train['LATITUDE'].values.reshape(-1, 1))
        altitude_model, alt_train = misc.normalization(dataset_config['y_normalization'],
                                                              y_train['ALTITUDE'].values.reshape(-1, 1))
        # Saving models
        joblib.dump(longitude_model, os.path.join(path_data_model,
                                             'longitude_model_'+dataset_config['y_normalization']+'.save'))
        joblib.dump(latitude_model, os.path.join(path_data_model,
                                             'latitude_model_'+dataset_config['y_normalization']+'.save'))
        joblib.dump(altitude_model, os.path.join(path_data_model,
                                             'altitude_model_'+dataset_config['y_normalization']+'.save'))
        y_train_temp['LONGITUDE'] = long_train
        y_train_temp['LATITUDE'] = lat_train
        y_train_temp['ALTITUDE'] = alt_train
        

    # Full y_data
    y_data = [y_train, y_test]
    y_data = pd.concat(y_data)

    # Floor - Label Encoding
    encoding_floor = LabelEncoder()
    lab_encoded_floor = encoding_floor.fit_transform(y_data['FLOOR'])
    y_encoded_floor = encoding_floor.transform(y_train_temp['FLOOR'])
    y_train_temp['FLOOR'] = y_encoded_floor.reshape(-1, 1)
    joblib.dump(encoding_floor, os.path.join(path_data_model, 'floor_label_encoder.save'))
    
    # Label encoding building
    encoding_building = LabelEncoder()
    lab_encoded_building = encoding_building.fit_transform(y_data['BUILDINGID'])
    y_encoded_building = encoding_building.transform(y_train_temp['BUILDINGID'])
    y_train_temp['BUILDINGID'] = y_encoded_building.reshape(-1, 1)
    joblib.dump(encoding_building, os.path.join(path_data_model, 'building_label_encoder.save'))
    
    # 
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
        df_match = match_function(unique_ap_list.iloc[i, :], i, temp2, ap_selection['num_aps'], 30)
        mask = [j for j in range(len(df_match)) if df_match[j] != 0]

        
        # Selecting data to train
        selected_y_train = y_train_temp.iloc[mask,:].copy()
        selected_x_train = df_x_train.iloc[mask,:].copy()
        selected_x_train = selected_x_train.loc[:, (selected_x_train != 0).any(axis=0)]
        # Save APs in view
        aps_used = selected_x_train.columns.values

        # Unique APs
        ap = unique_ap_list[0]

        path_save_info_ap = os.path.join(path_data_model,'AP'+str(ap[i]))
        
        if not os.path.exists(path_save_info_ap):
            os.makedirs(path_save_info_ap)

            data = {}

            data['aps'] = aps_used
            data['indexes'] = mask

            joblib.dump(data, os.path.join(path_save_info_ap, 'data.save'))
        else:
            pass
    
    results_path = os.path.join(path_config['results'], dataset_name, 'KNN') 
    test_knn(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
             path_saved_models=path_data_model,results_path=results_path)
    
    print("END")
    return True
