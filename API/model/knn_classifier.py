import os
import joblib
import json
import pandas as pd
import numpy as np

class KNNClassifier:
    def __init__(self, saved_model_path: str = "model/saved_models"):
        """Initialize the classifier and load hyperparameters."""
        self.saved_model_path = saved_model_path
        self.k = self.load_hyperparameters()
        self.model_data = None

    def load_hyperparameters(self):
        """Load KNN hyperparameters from JSON file."""
        hyperparam_path = "API/model/Hyperparams/KNN_hparam.json"
        with open(hyperparam_path, "r") as f:
            hyperparams = json.load(f)
        return hyperparams.get("k", 5)  # Default k=5 if not specified

    def train(self, data_path: str):
        """Load training data and store it."""
        df = pd.read_csv(data_path)
        self.y_train = df['class'].values
        self.X_train = df.drop(columns=['class']).values
        self.model_data = {"X_train": self.X_train, "y_train": self.y_train, "k": self.k}
        

    def _compute_manhattan_distance(self, X1, X2):
        """Compute Manhattan distance between two arrays."""
        return np.abs(X1[:, np.newaxis] - X2).sum(axis=2)
    
    
    def predict(self, X):
        """Make predictions using KNN logic."""
        if self.model_data is None:
            raise ValueError("Model is not trained. Call train() before predicting.")
    
        
    
        distances = self._compute_manhattan_distance(X, self.model_data["X_train"])
        predictions = []
        
        for dist_row in distances:
            nearest_indices = np.argsort(dist_row)[:self.k]
            nearest_labels = self.model_data["y_train"][nearest_indices]
            prediction = np.bincount(nearest_labels.astype(int)).argmax()
            predictions.append(prediction)
        
        return np.array(predictions)


    def save_model(self, save_path: str = r"API/model/saved_models/knn_model.pkl"):
          """Save the trained model (entire KNNClassifier object)."""
          with open(save_path, "wb") as f:
              joblib.dump(self, f)  #  Save the whole class instance
              
    @staticmethod
    def load_model(model_path: str = "API/model/saved_models/knn_model.pkl"):
        """Load the trained KNN model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, "rb") as f:
            model = joblib.load(f)  #  Load full class instance
      
        return model        


if __name__ == "__main__":
    train_dataset_path = r"API\data\processed\train_data.csv"
    
    classifier = KNNClassifier()
    classifier.train(train_dataset_path)
    classifier.save_model()
