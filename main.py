import argparse
import os
from miscellaneous.misc import Misc
from run_nn import run_nn
from run_cnn_lstm import run_cnn_lstm
from run_cnnloc import run_cnnloc
from run_cnnloc_prom import run_cnnloc_prom
from run_knn import run_knn
from run_mlp import run_mlp
from run_fmlp import run_fmlp
from plots import run_plot
from run_full_cnnloc import run_full_cnnloc
from numpy.random import seed, default_rng
import numpy as np
import tensorflow as tf
### Warning ###
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
rnd_seed = 11
default_rng(rnd_seed)
tf.random.set_seed(
    rnd_seed
)

if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--config-file', dest='config_file', action='store', default='', help='Config file')
    p.add_argument('--dataset', dest='dataset', action='store', default='', help='Dataset')
    p.add_argument('--algorithm', dest='algorithm', action='store', default='', help='Algorithm')

    args = p.parse_args()

    # Check if the the config file exist
    config_file = str(args.config_file)
    datasets = str(args.dataset)
    algorithm = str(args.algorithm)

    misc = Misc()

    if config_file == '':
        print(misc.log_msg("ERROR", "Please specify the config file \n"
                                    " e.g., python main.py --config-file config.json \n"
                                    "or \n python main.py --config-file config.json --dataset DSI1,DSI2"))
        exit(-1)

    if os.path.exists(config_file):
        pass
    else:
        print(misc.log_msg("ERROR", "Oops... Configuration file not found. Please check the name and/or path."))
        exit(-1)

    # Config file from .json to dict
    config = misc.json_to_dict(config_file)

    # Check if all the parameters are present in the configuration file
    misc.conf_parse(config)

    # Get all the datasets availables in the config file
    list_datasets = misc.get_datasets_availables(config)

    if datasets != '':
        datasets = datasets.split(',')
        for i, dataset in enumerate(datasets):
            for j in range(0, len(config['dataset'])):
                if dataset == config['dataset'][j]['name']:
                    main_path = config['path']['data_source']
                    if os.path.exists(os.path.join(main_path, dataset,
                                                   config['dataset'][j]['train_dataset'])) and \
                            config['dataset'][j]['train_dataset'] != "":
                        pass
                    else:
                        print(misc.log_msg("ERROR",
                                           "Train Dataset not found."))
                        exit(-1)

                    if os.path.exists(os.path.join(main_path, dataset,
                                                   config['dataset'][j]['test_dataset'])) and \
                            config['dataset'][j]['test_dataset'] != "":
                        pass
                    else:
                        print(misc.log_msg("ERROR",
                                           "Test Dataset not found."))
                        exit(-1)

                    if os.path.exists(os.path.join(main_path, dataset,
                                                   config['dataset'][j]['validation_dataset'])):
                        pass
                    else:
                        print(misc.log_msg("ERROR",
                                           "Validation Dataset not found."))
                        exit(-1)

                    if algorithm == 'CNN-LSTM':
                        run_cnn_lstm(dataset_name=dataset, ap_selection=config['ap_selection'], 
                                     path_config=config['path'], dataset_config=config['dataset'][j], 
                                     cnn_lstm_config=config['CNN-LSTM'])
                    elif algorithm == 'NN':
                        run_nn(dataset_name=dataset, ap_selection=config['ap_selection'], 
                               path_config=config['path'], dataset_config=config['dataset'][j], 
                               nn_config=config['NN'])
                    elif algorithm == 'MLP':
                        run_mlp(dataset_name=dataset, ap_selection=config['ap_selection'], 
                               path_config=config['path'], dataset_config=config['dataset'][j], 
                               mlp_config=config['MLP'])
                    elif algorithm == 'FMLP':
                        run_fmlp(dataset_name=dataset, ap_selection=config['ap_selection'], 
                               path_config=config['path'], dataset_config=config['dataset'][j], 
                               mlp_config=config['MLP'])
                    elif algorithm == 'CNNLoc':
                        run_cnnloc(dataset_name=dataset, ap_selection=config['ap_selection'], 
                                     path_config=config['path'], dataset_config=config['dataset'][j], 
                                     cnnloc_config=config['CNNLoc'])
                    elif algorithm == 'FullCNNLoc':
                        run_full_cnnloc(dataset_name=dataset, ap_selection=config['ap_selection'], 
                                     path_config=config['path'], dataset_config=config['dataset'][j], 
                                     cnnloc_config=config['CNNLoc'])
                    elif algorithm == 'CNNLocProm':
                        run_cnnloc_prom(dataset_name=dataset, ap_selection=config['ap_selection'], 
                                     path_config=config['path'], dataset_config=config['dataset'][j], 
                                     cnnloc_config=config['CNNLoc'])
                    elif algorithm == 'PLOT':
                        run_plot(dataset_name=dataset, ap_selection=config['ap_selection'], 
                                     path_config=config['path'], dataset_config=config['dataset'][j])
                    elif algorithm == 'KNN':
                        run_knn(dataset_name=dataset, ap_selection=config['ap_selection'], 
                                     path_config=config['path'], dataset_config=config['dataset'][j])
                    
                    else:
                        print(misc.log_msg("ERROR",
                                           "Method not available. Accepted training methods "
                                           "{CNN-LSTM|CNNLoc}"))
                        exit(-1)
                        
    else:
        for dataset in config['dataset']:
            dataset_name = dataset['name']
            if algorithm == 'CNN-LSTM':
                run_cnn_lstm(dataset_name=dataset_name, path_config=config['path'], dataset_config=dataset,
                             building_config=config['model_config'][0], floor_config=config['model_config'][1],
                             positioning_config=config['model_config'][2], algorithm=algorithm)
            elif algorithm == 'NN':
                        run_nn(dataset_name=dataset, ap_selection=config['ap_selection'], 
                               path_config=config['path'], dataset_config=config['model_config'][1], 
                               nn_config=config['NN'])
            elif algorithm == 'MLP':
                        run_mlp(dataset_name=dataset, ap_selection=config['ap_selection'], 
                               path_config=config['path'], dataset_config=config['model_config'][1], 
                               nn_config=config['MLP'])
            elif algorithm == 'CNNLoc':
                run_cnnloc(dataset_name=dataset_name, path_config=config['path'], dataset_config=dataset,
                             building_config=config['model_config'][0], floor_config=config['model_config'][1],
                             positioning_config=config['model_config'][2], algorithm=algorithm)
            elif algorithm == 'KNN':
                run_cnnloc(dataset_name=dataset_name, path_config=config['path'], dataset_config=dataset,
                             building_config=config['model_config'][0], floor_config=config['model_config'][1],
                             positioning_config=config['model_config'][2], algorithm=algorithm)
            else:
                print(misc.log_msg("ERROR",
                                   "Method not available. Accepted training methods "
                                   "{CNN-LSTM|CNNLoc}"))
                exit(-1)