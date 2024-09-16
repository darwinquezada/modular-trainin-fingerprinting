import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


class Position_KNN():
    def __init__(self, k=1, metric='euclidean', weight='distance'):
        self.classifier_building = None
        self.classifier_floor = None
        self.regressor = None
        self.y_train = None
        self.k = k
        self.metric = metric
        self.weight = weight

    def fit(self, X_train=None, y_train=None):
        self.y_train = y_train
        self.regressor = KNeighborsRegressor(n_neighbors=self.k, metric=self.metric, weights=self.weight)
        self.classifier_building = KNeighborsClassifier(n_neighbors=self.k, metric=self.metric, weights=self.weight)
        self.classifier_floor = KNeighborsClassifier(n_neighbors=self.k, metric=self.metric, weights=self.weight)
        self.regressor.fit(X_train, y_train[:, 0:3])
        self.classifier_floor.fit(X_train, y_train[:, 3])
        self.classifier_building.fit(X_train, y_train[:, 4])

    def predict_position_2D(self, X_test=None, y_test=None):
        prediction_2D = self.regressor.predict(X_test)
        distance = np.linalg.norm(prediction_2D[0][0:2] - y_test[0:2])        
        return prediction_2D , distance

    def predict_position_3D(self, X_test=None, y_test=None):
        prediction_3D = self.regressor.predict(X_test)
        distance = np.linalg.norm(prediction_3D[0] - y_test[0:3])
        return prediction_3D, distance

    def floor_hit_rate(self, X_test=None, y_test=None):
        prediction_floor = self.classifier_floor.predict(X_test)
        return prediction_floor

    def building_hit_rate(self, X_test=None, y_test=None):
        prediction_building = self.classifier_building.predict(X_test)
        return prediction_building
