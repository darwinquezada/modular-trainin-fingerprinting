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


def test_full_cnnloc(y_train, X_test, y_test, path_saved_models, path_results, ap_selection, cnnloc_config):

    y_test_original = y_test

    # Data transformation
    X_test = data_helper.normalizeX(X_test, cnnloc_config['b'])

    df_prediction_results = pd.DataFrame(columns=[
                                         'LONGITUDE', 'LATITUDE', 'ALTITUDE', 'FLOOR', 'BUILDINGID', 'ERROR_3D', 'ERROR_2D'])

    floor_label_enc = joblib.load(os.path.join(
        path_saved_models, 'floor_label_encoder.save'))
    building_label_enc = joblib.load(os.path.join(
        path_saved_models, 'building_label_encoder.save'))
    building_model = load_model(os.path.join(path_saved_models, 'building.tf'))
    floor_model = load_model(os.path.join(path_saved_models, 'floor.tf'))
    position_model = load_model(os.path.join(
        path_saved_models, 'positioning.tf'))

    # Floor hit rate
    floor = floor_model.predict(X_test)
    floor = data_helper.oneHotDecode(floor)
    floor = floor_label_enc.inverse_transform(floor)
    # floor = floor_label_enc.inverse_transform(floor)

    # Building hit rate
    building = building_model.predict(X_test)
    building = data_helper.oneHotDecode(building)
    # building = building_label_enc.inverse_transform(building)

    normY = data_helper.NormY()
    normY.fit(y_train[:, 0], y_train[:, 1], y_train[:, 2])

    # Predict longitude, latitude and altitude
    predict_Location = position_model.predict(X_test)
    longitude, latitude, altitude = normY.reverse_normalizeY(predict_Location[:, 0],
                                                             predict_Location[:, 1],
                                                             predict_Location[:, 2])
    
    error_3d = np.linalg.norm(
        np.transpose([longitude[:,0], latitude[:,0], altitude[:,0]]) - y_test_original[:, 0:3], axis=1)
    error_2d = np.linalg.norm(
        np.transpose([longitude[:,0], latitude[:,0]]) - y_test_original[:, 0:2], axis=1)

    prediction = pd.DataFrame(list(zip(longitude[:,0], latitude[:,0], altitude[:,0], floor,
                                              building, error_3d, error_2d)),
                                     columns=['LONGITUDE', 'LATITUDE', 'ALTITUDE', 'FLOOR', 'BUILDINGID', 'ERROR_3D', 'ERROR_2D'])
    

    df_prediction_results = df_prediction_results.append(
        prediction, ignore_index=True)

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    df_prediction_results.to_csv(os.path.join(
        path_results, 'results_' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '.csv'), index=False)
    mean_3d_error = np.mean(np.linalg.norm(
        df_prediction_results.iloc[:, 0:3].values - y_test_original[:, 0:3], axis=1))
    mean_2d_error = np.mean(np.linalg.norm(
        df_prediction_results.iloc[:, 0:2].values - y_test_original[:, 0:2], axis=1))

    diff_pred_test_floor = np.subtract(
        df_prediction_results['FLOOR'], y_test_original[:, 3])
    floor_hit_rate = (sum(map(lambda x: x == 0, diff_pred_test_floor)) /
                      np.shape(df_prediction_results['FLOOR'])[0])*100

    diff_pred_test_building = np.subtract(
        df_prediction_results['BUILDINGID'], y_test_original[:, 4])
    building_hit_rate = (sum(map(lambda x: x == 0, diff_pred_test_building)) /
                         np.shape(df_prediction_results['BUILDINGID'])[0])*100

    datestr = "%m/%d/%Y %I:%M:%S %p"

    logging.basicConfig(
        filename=path_results + '/' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '.log',
        level=logging.INFO,
        filemode="w",
        datefmt=datestr,
        # force=True
    )

    logging.info(
        "---------------------------- FullCNNLoc ---------------------------")
    logging.info(
        ' Mean 2D positioning error : {:.3f}m'.format(mean_2d_error))
    logging.info(
        ' Mean 3D positioning error : {:.3f}m'.format(mean_3d_error))
    logging.info(' Floor hit rate : {:.3f}%'.format(floor_hit_rate))
    logging.info(' Building hit rate : {:.3f}%'.format(building_hit_rate))
    logging.info(
        ' -----------------------------------------------------------')
    logging.info(' Neural network hyperparameters:')
    logging.info(str(cnnloc_config))

    print("Mean 3D error: {:.3f}".format(mean_3d_error))
    print("Mean 2D error: {:.3f}".format(mean_2d_error))
    print("Building hit rate: {:.3f}".format(building_hit_rate))
    print("Floor hit rate: {:.3f}".format(floor_hit_rate))
