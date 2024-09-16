from collections import Counter
from time import sleep
import preprocessing.data_helper as data_helper
from sklearn.preprocessing import LabelEncoder
from models.cnnloc import EncoderDNN
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Adagrad, Ftrl, Nadam, RMSprop, SGD
from test.test_cnnloc_prom import test_cnnloc_prom
import tensorflow as tf
import time
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


def match_function(row, position, dataframe, num_max_rss, m_percent):
    intersect = dataframe.apply(lambda x: list(
        set(x).intersection(set(row))), axis=1)
    # intersect[position] = []
    intersect = list(
        map(lambda x: list(filter(lambda y: y >= 0, x)), intersect))
    match_percent = list(map(lambda x: 0 if (
        len(x)*100)/num_max_rss < m_percent else (len(x)*100)/num_max_rss, intersect))
    return match_percent


def run_cnnloc_prom(dataset_name=None, ap_selection=None, path_config=None, dataset_config=None, cnnloc_config=None):

    path_data_model = os.path.join(
        path_config['saved_model'], dataset_name, 'CNNLoc', str(ap_selection['num_aps']))

    dataset_path = os.path.join(path_config['data_source'], dataset_name)

    test_csv_path = os.path.join(dataset_path, dataset_config['test_dataset'])

    if bool(dataset_config['validation_dataset']):
        valid_csv_path = os.path.join(
            dataset_path, dataset_config['validation_dataset'])
    else:
        X_valid = []
        y_valid = []

    train_csv_path = os.path.join(
        dataset_path, dataset_config['train_dataset'])

    (train_x, train_y), (test_x, test_y) = data_helper.load_data_all(
        train_csv_path, test_csv_path)

    
    if cnnloc_config['opt'] == 'Adamax':
        lr = cnnloc_config['LR_23']
    else:
        lr = cnnloc_config['LR_34']

    df_x_train_transformed = pd.DataFrame(
        data_helper.normalizeX(train_x, cnnloc_config['b']))

    df_x_train = pd.DataFrame(train_x)
    y_train_temp = pd.DataFrame(train_y)

    # Sorting the dataset (Descending)
    sorted_df = np.sort(df_x_train_transformed)[:, ::-1]
    sorted_df = pd.DataFrame(np.where(sorted_df > 0, 1, -1))
    temp = sort_df(df_x_train_transformed).replace(
        0, df_x_train_transformed.shape[1] + 1)
    temp2 = temp.mul(sorted_df).values
    temp2 = pd.DataFrame(np.where(temp2 < 0, 0, temp2)
                         ).iloc[:, 0:int(ap_selection['num_aps'])]

    if np.shape(temp2)[1] > 1:
        unique_ap_list = pd.DataFrame(
            temp2.iloc[:, :].drop_duplicates().reset_index(drop=True))
    else:
        unique_ap_list = temp2.drop_duplicates().reset_index(drop=True)

    # Create groups
    df_match = pd.DataFrame()
    for i in range(0, np.shape(unique_ap_list)[0]):
        ap = unique_ap_list[0]
        path_save_info_ap = os.path.join(path_data_model, 'AP' + str(ap[i]))
            
        if not os.path.exists(path_save_info_ap):
            os.makedirs(path_save_info_ap)
            
            df_match = match_function(
                unique_ap_list.iloc[i, :], i, temp2, ap_selection['num_aps'], 1)

            mask = [j for j in range(len(df_match)) if df_match[j] != 0]
            # Selecting data to train
            selected_y_train = y_train_temp.iloc[mask, :].copy()
            selected_x_train = df_x_train.iloc[mask, :].copy()
            selected_x_train = selected_x_train.loc[:,
                                                    (selected_x_train != 100).any(axis=0)]
        
            # Label encoding
            lab_enc = LabelEncoder()
            y_train_floor = lab_enc.fit_transform(selected_y_train.iloc[:, 3].values)
            joblib.dump(lab_enc, os.path.join(path_save_info_ap, 'floor_label_encoder.save'))

            # Label encoding Building
            lab_enc_bld = LabelEncoder()
            y_train_bld = lab_enc_bld.fit_transform(selected_y_train.iloc[:, 4].values)
            joblib.dump(lab_enc_bld, os.path.join(path_save_info_ap, 'building_label_encoder.save'))

            train_y_sel = np.delete(selected_y_train.values, [3, 4], 1)

            train_y_sel = np.concatenate((train_y_sel, np.transpose(np.array(y_train_floor, ndmin=2)),
                                  np.transpose(np.array(y_train_bld, ndmin=2))), axis=1)


            # Save APs in view
            aps_used = selected_x_train.columns.values

            num_classes_flr = len(np.unique(train_y_sel[:,3]))
            num_classes_bld = len(np.unique(train_y_sel[:,4]))

            # Save list of features of the selected fingerprints
            joblib.dump(aps_used, os.path.join(
                path_save_info_ap, 'list_features.save'))

            # Save mask
            joblib.dump(mask, os.path.join(
                path_save_info_ap, 'mask_samples.save'))

            # Training
            encode_dnn_model = EncoderDNN(num_floor_classes=num_classes_flr, num_bld_classes=num_classes_bld,
                                          num_features=np.shape(selected_x_train)[1], path_saved_model=path_save_info_ap)

            encode_dnn_model.patience = cnnloc_config['p']
            encode_dnn_model.b = cnnloc_config['b']
            encode_dnn_model.epoch_AE = cnnloc_config['epoch_sae']
            encode_dnn_model.epoch_floor = cnnloc_config['epoch_floor']
            encode_dnn_model.epoch_position = cnnloc_config['epoch_position']
            encode_dnn_model.epoch_building = cnnloc_config['epoch_building']
            encode_dnn_model.dropout = cnnloc_config['dp']
            encode_dnn_model.loss = cnnloc_config['loss']
            encode_dnn_model.opt = Adam(lr=lr)

            encoder_model, bottleneck_model, position_model, floor_model, building_model = encode_dnn_model.fit(
                selected_x_train.values, selected_y_train.values)

            encoder = os.path.join(path_save_info_ap, 'encoder.tf')
            encoder_model.save(encoder, overwrite=True)
            # joblib.dump(encoder_model,encoder)

            bottleneck = os.path.join(path_save_info_ap, 'bottleneck.tf')
            bottleneck_model.save(bottleneck, overwrite=True)
            # joblib.dump(bottleneck_model, bottleneck)

            position = os.path.join(path_save_info_ap, 'positioning.tf')
            position_model.save(position, overwrite=True)
            # joblib.dump(position_model, position)

            floor = os.path.join(path_save_info_ap, 'floor.tf')
            floor_model.save(floor, overwrite=True)
            # joblib.dump(floor_model, floor)

            building = os.path.join(path_save_info_ap, 'building.tf')
            building_model.save(building, overwrite=True)
            # joblib.dump(building_model, building)
    # Test
    path_results = os.path.join(path_config['results'], dataset_name, 'CNNLocProm', str(ap_selection['num_aps']))
    test_cnnloc_prom(y_train=train_y, X_test=test_x, y_test=test_y, path_saved_models=path_data_model,
                path_results=path_results, ap_selection=ap_selection, cnnloc_config=cnnloc_config)

