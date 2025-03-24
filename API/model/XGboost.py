import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Define the class for  XGboost
class XGBoostClassifier:
    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 65):
        """Initialize the classifier with dataset path."""
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess data."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        self.y = df['class']
        self.X = df.drop(columns=['class'])
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

    def train(self):
        """Train the XGBoost model."""
        num_classes = len(self.y_train.unique())
        self.model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes)
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model is not trained. Call train() before predicting.")
        return self.model.predict(X)

    def evaluate(self):
        """Evaluate model performance and print key metrics."""
        if self.model is None:
            raise ValueError("Model is not trained. Call train() before evaluation.")

        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        metrics = {
            "Train Accuracy": accuracy_score(self.y_train, y_train_pred),
            "Test Accuracy": accuracy_score(self.y_test, y_test_pred),
            "Precision": precision_score(self.y_test, y_test_pred, average="weighted"),
            "Recall": recall_score(self.y_test, y_test_pred, average="weighted"),
            "F1-score": f1_score(self.y_test, y_test_pred, average="weighted"),
            "Confusion Matrix": confusion_matrix(self.y_test, y_test_pred),
            "Classification Report": classification_report(self.y_test, y_test_pred)
        }

        print("\nEvaluation Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}:\n{value}")

    def save_model(self, save_path: str):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("Model is not trained. Call train() before saving.")

        self.model.save_model(save_path)
        print(f"Model saved at: {save_path}")

if __name__ == "__main__":
    # Set dataset path
    dataset_path = r"..\data\processed\train_data.csv"

    # Initialize and use the classifier
    classifier = XGBoostClassifier(data_path=dataset_path)
    classifier.train()
    classifier.evaluate()

    # Save the model
    model_save_path = r".\saved_models\xgb_model.json"
    classifier.save_model(model_save_path)
