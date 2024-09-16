import json
# from colorama import init, Fore, Back, Style
from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Adagrad, Ftrl, Nadam, RMSprop, SGD
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

class Misc:

    def json_to_dict(self, config_file):
        # Opening JSON file
        with open(config_file) as json_file:
            dictionary = json.load(json_file)
        return dictionary

    def check_key(self, dict, list_parameters):
        for param in range(0, len(list_parameters)):
            if list_parameters[param] in dict.keys():
                pass
            else:
                print(self.log_msg("ERROR", " The following parameter is not found in the configuration file: " +
                                   list_parameters[param]))
                exit(-1)
        return True

    def log_msg(self, level, message):
        # init(autoreset=True)
        if level == 'WARNING':
            return message # Fore.YELLOW + message
        elif level == 'ERROR':
            return message # Fore.RED + message
        elif level == 'INFO':
            return message # Style.RESET_ALL + message

    def conf_parse(self, dict):
        # These parameters are compulsory in the config file
        conf_main_param = ['path', 'dataset', 'CNNLoc', 'CNN-LSTM']
        dataset_param = ['name', 'data_representation', 'x_normalization', 'y_normalization',
                         'default_null_value', 'distance_metric', 'k','train_dataset', 
                         'test_dataset','validation_dataset']
        model_param = ['model', 'train', 'lr', 'epochs', 'batch_size', 'optimizer']

        # Check if all the main parameters are in the config file
        if self.check_key(dict, conf_main_param):
            pass

        # Datasets parameters
        for data in dict['dataset']:
            if self.check_key(data, dataset_param):
                pass


    def get_datasets_availables(self, dict):
        list_datasets = []
        for data in dict['dataset']:
            list_datasets.append(data['name'])
        return list_datasets

    def optimizer(self, opt, lr):
        if opt == 'Adam':
            return Adam(lr)
        elif opt == 'Adamax':
            return Adamax(lr)
        elif opt == 'Adadelta':
            return Adadelta(lr)
        elif opt == 'Adagrad':
            return Adagrad(lr)
        elif opt == 'Ftrl':
            return Ftrl(lr)
        elif opt == 'Nadam':
            return Nadam(lr)
        elif opt == 'RMSprop':
            return RMSprop(lr)
        elif opt == 'SGD':
            return SGD(lr)
        else:
            return Adam(lr)
    
    def normalization(self, opt, data):
        if opt == 'Normalizer':
            norm_model = Normalizer()
            norm_data = norm_model.fit_transform(data)
        elif opt == 'MinMaxScaler':
            norm_model = MinMaxScaler()
            norm_data = norm_model.fit_transform(data)
        elif opt == 'StandardScaler':
            norm_model = StandardScaler()
            norm_data = norm_model.fit_transform(data)
        elif opt == 'RobustScaler':
            norm_model = RobustScaler()
            norm_data = norm_model.fit_transform(data)
        elif opt == 'MaxAbsScaler':
            norm_model = MaxAbsScaler()
            norm_data = norm_model.fit_transform(data)
        else:
            print(self.log_msg("ERROR", " Method not allowed. Please, choose one of the following methods: " +
                               "{Normalizer|MinMaxScaler|StandardScaler|RobustScaler|MaxAbsScaler}"))
            exit(-1)
        return norm_model, norm_data


