from time import sleep
import preprocessing.data_helper as data_helper
from sklearn.preprocessing import LabelEncoder
from models.cnnloc import EncoderDNN
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Adagrad, Ftrl, Nadam, RMSprop, SGD
from test.test_full_cnnloc import test_full_cnnloc
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


def run_full_cnnloc(dataset_name=None, ap_selection=None, path_config=None, dataset_config=None, cnnloc_config=None):

    path_data_model = os.path.join(
        path_config['saved_model'], dataset_name, 'FullCNNLoc')

    dataset_path = os.path.join(path_config['data_source'], dataset_name)

    test_csv_path = os.path.join(
        dataset_path, dataset_config['test_dataset'])

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

    if not os.path.exists(path_data_model):
        os.makedirs(path_data_model)

        if cnnloc_config['opt'] == 'Adamax':
            lr = cnnloc_config['LR_23']
        else:
            lr = cnnloc_config['LR_34']

        # Label encoding
        lab_enc = LabelEncoder()
        y_train_floor = lab_enc.fit_transform(train_y[:, 3])
        joblib.dump(lab_enc, os.path.join(
            path_data_model, 'floor_label_encoder.save'))

        # Label encoding Building
        lab_enc_bld = LabelEncoder()
        y_train_bld = lab_enc_bld.fit_transform(train_y[:, 4])
        joblib.dump(lab_enc_bld, os.path.join(
            path_data_model, 'building_label_encoder.save'))

        train_y = np.delete(train_y, [3, 4], 1)

        train_y = np.concatenate((train_y, np.transpose(np.array(y_train_floor, ndmin=2)),
                                  np.transpose(np.array(y_train_bld, ndmin=2))), axis=1)

        num_classes_flr = len(np.unique(train_y[:, 3]))
        num_classes_bld = len(np.unique(train_y[:, 4]))

        # Training
        encode_dnn_model = EncoderDNN(num_floor_classes=num_classes_flr, num_bld_classes=num_classes_bld,
                                      num_features=np.shape(train_x)[1], path_saved_model=path_data_model)

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
            train_x, train_y)

        encoder = os.path.join(path_data_model, 'encoder.tf')
        encoder_model.save(encoder, overwrite=True)
        # joblib.dump(encoder_model,encoder)

        bottleneck = os.path.join(path_data_model, 'bottleneck.tf')
        bottleneck_model.save(bottleneck, overwrite=True)
        # joblib.dump(bottleneck_model, bottleneck)

        position = os.path.join(path_data_model, 'positioning.tf')
        position_model.save(position, overwrite=True)
        # joblib.dump(position_model, position)

        floor = os.path.join(path_data_model, 'floor.tf')
        floor_model.save(floor, overwrite=True)
        # joblib.dump(floor_model, floor)

        building = os.path.join(path_data_model, 'building.tf')
        building_model.save(building, overwrite=True)
        # joblib.dump(building_model, building)

    path_results = os.path.join(
        path_config['results'], dataset_name, 'FullCNNLoc')
    test_full_cnnloc(y_train=train_y, X_test=test_x, y_test=test_y, path_saved_models=path_data_model,
                path_results=path_results, ap_selection=ap_selection, cnnloc_config=cnnloc_config)
