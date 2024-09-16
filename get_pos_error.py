import pandas as pd
import numpy as np

df_test = pd.read_csv('/Users/darwinquezada/Documents/Development/Python/IDE/DNNAPWhioutSavingModels/datasets/UJI1/Test.csv')
df_results = pd.read_csv('/Users/darwinquezada/Documents/Development/Python/IDE/DNNAPWhioutSavingModels/results/UJI1/CNNLoc/3/results_2023_05_07_16_08_56.csv')

y_test = df_test.iloc[:, -5:]

three_error = np.linalg.norm(df_results.iloc[:,0:3].values - y_test.iloc[:,0:3].values, axis=1)
two_error = np.linalg.norm(df_results.iloc[:,0:2].values - y_test.iloc[:,0:2].values, axis=1)

mean_3d_error = np.mean(three_error)
mean_2d_error = np.mean(two_error)

fifty = np.percentile(two_error, 50)
seventy_five = np.percentile(two_error, 75)
eighty = np.percentile(two_error, 80)
ninety = np.percentile(two_error, 90)
ninety_five = np.percentile(two_error, 95)

diff_pred_test_floor = np.subtract(df_results['FLOOR'], y_test['FLOOR'])
floor_hit_rate = (sum(map(lambda x:x == 0, diff_pred_test_floor))/np.shape(df_results['FLOOR'])[0])*100

diff_pred_test_building = np.subtract(df_results['BUILDINGID'], y_test['BUILDINGID'])
building_hit_rate = (sum(map(lambda x:x == 0, diff_pred_test_building))/np.shape(df_results['BUILDINGID'])[0])*100


print( "Mean 3D error: {:.3f}".format(mean_3d_error))
print( "Mean 2D error: {:.3f}".format(mean_2d_error))
print( "50 percentil: {:.3f}".format(fifty))
print( "75 percentil: {:.3f}".format(seventy_five))
print( "80 percentil: {:.3f}".format(eighty))
print( "90 percentil: {:.3f}".format(ninety))
print( "95 percentil: {:.3f}".format(mean_2d_error))
print( "Building hit rate: {:.3f}".format(building_hit_rate))
print( "Floor hit rate: {:.3f}".format(floor_hit_rate))