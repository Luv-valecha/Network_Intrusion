import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import pairwise_distances

class KNNClassifier:
    """
    K-Nearest Neighbors Classifier 
    using Manhattan distance and majority voting.

    Attributes:
        k (int): Number of neighbors to consider. Fixed at 5.
        distance_metric (str): Distance metric to use ('manhattan').
    """

    # Constructor initializes fixed parameters
    def __init__(self, k=5, distance_metric='manhattan'):
        """
        Initialize the KNN classifier with default k=5 and Manhattan distance.
        """
        self.k = k
        self.distance_metric = distance_metric

    def train(self, X_train: np.array, y_train: np.array):
        """
        Store the training data.
        
        Args:
            X_train (np.array): Training feature matrix
            y_train (np.array): Training labels
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test: np.array):
        """
        Predict the labels for given test data using KNN logic.
        
        Steps:
        - Compute pairwise distances between test samples and training data
        - For each test sample, identify k-nearest neighbors
        - Perform majority voting to determine final prediction
        
        Args:
            X_test (np.array): Test feature matrix
            
        Returns:
            np.array: Predicted class labels
        """
        # Compute distances (efficient vectorized computation)
        distances = pairwise_distances(X_test, self.X_train, metric=self.distance_metric)
        
        predictions = []
        # Iterate through each test point's distances
        for dist_row in distances:
            # Find indices of k nearest neighbors
            nearest_indices = np.argsort(dist_row)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            
            # Majority vote among nearest neighbors
            prediction = np.bincount(nearest_labels.astype(int)).argmax()
            predictions.append(prediction)

        return np.array(predictions)

