import os
from mpl_toolkits.mplot3d import axes3d
from preprocessing.data_preprocessing import load, new_non_detected_value
from preprocessing.data_representation import DataRepresentation
from miscellaneous.misc import Misc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def run_plot(dataset_name=None, ap_selection=None, path_config=None, dataset_config=None):

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
        
    y_train_temp = y_train.copy()
     
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
        df_match = match_function(unique_ap_list.iloc[i, :], i, temp2, ap_selection['num_aps'], 1)
        mask = [j for j in range(len(df_match)) if df_match[j] != 0]
        # Selecting data to train
        selected_y_train = y_train_temp.iloc[mask,:].copy()
        selected_x_train = df_x_train.iloc[mask,:].copy()
        selected_x_train = selected_x_train.loc[:, (selected_x_train != 0).any(axis=0)]

        fig = plt.figure(figsize=(16,16), dpi=80)
        ax = fig.add_subplot(projection='3d')

        ax.scatter(y_train_temp.iloc[:,0], y_train_temp.iloc[:,1], y_train_temp.iloc[:,2], facecolors='none', edgecolors='black')
        # Unique APs
        ap = unique_ap_list[0]

        print( "---------------------------------- Max. AP: " + str(ap[i]) + ", Samples: " + str(np.shape(mask)[0]) + ", Valid APs: " + str(np.shape(selected_x_train)[1]) + "-------------------------------")
        plt.title("Max. AP: " + str(ap[i]) + ", Samples: " + str(np.shape(mask)[0]) + ", Valid APs: " + str(np.shape(selected_x_train)[1]))
        ax.scatter(selected_y_train.iloc[:,0], selected_y_train.iloc[:,1], selected_y_train.iloc[:,2], marker='o', color='red')
        ax.view_init(12,-104)
        plt.savefig(os.path.join('plots', dataset_name, 'AP'+str(ap[i]) + '.pdf'))
        # plt.show()