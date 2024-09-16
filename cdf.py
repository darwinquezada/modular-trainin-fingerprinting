from cProfile import label
from collections import Counter
import os
from turtle import color
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt

# plt.style.use('seaborn-v0_8-paper')

dataset = 'UJI1'

df_test = pd.read_csv('/Users/darwinquezada/Documents/Development/Python/IDE/DNNAPWhioutSavingModels/datasets/UJI1/Test.csv')
df_results = pd.read_csv('/Users/darwinquezada/Documents/Development/Python/IDE/DNNAPWhioutSavingModels/results/UJI1/CNNLoc/3/results_2023_05_07_16_08_56.csv')

df_results_cnnloc = pd.read_csv('/Users/darwinquezada/Documents/Development/Python/IDE/CNNLoc/POS_ERRORUJI1.csv')

y_test = df_test.iloc[:, -5:]

two_error = np.linalg.norm(df_results_cnnloc.iloc[:,0:2].values - y_test.iloc[:,0:2].values, axis=1)
two_error_modular = np.linalg.norm(df_results.iloc[:,0:2].values - y_test.iloc[:,0:2].values, axis=1)

count, bins_count = np.histogram(two_error_modular, bins=30)
pdf = count / sum(count)
cdf = np.cumsum(pdf)

count_fdb, bins_count_fdb = np.histogram(two_error, bins=30)
pdf_fdb = count_fdb / sum(count_fdb)
cdf_fdb = np.cumsum(pdf_fdb)

N = 50
x = np.linspace(0, 60, N , endpoint=False)
y = np.zeros(N)

with plt.style.context(("seaborn-pastel",)):
    
    plt.rcParams["figure.figsize"] = [4, 3]
    plt.rcParams["figure.autolayout"] = True

    plt.plot(bins_count_fdb[1:], cdf_fdb, label="CNNLoc", color='#92c6ff')
    plt.plot(bins_count[1:], cdf, label="Modular NN", color='#ff9e9a')
    plt.plot(x, y+0.8, ":", color='#c6cdd7');  # dotted red
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    # plt.text(30, 0.81, '80%', fontsize=10, rotation_mode='anchor')
    plt.title(dataset + " - CDF 2D positioning error", fontsize=12)
    plt.xlabel("2D positioning error [m]", fontsize=11)
    plt.ylabel("CDF", fontsize=11)
    plt.legend()
    
    plt.xlim([0, 60])

    main_path = os.path.join('report', 'PLOTS')

    if not os.path.exists(main_path):
        os.makedirs(main_path)
        
    plt.savefig(os.path.join(main_path,'fig_'+dataset+'_cdf.pdf'))
    plt.show()