import os
import pandas as pd
import xgboost as xgb
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


class XGBoostClassifier:
    def __init__(self, train_data_path: str, random_state: int = 65):
        """Initialize the classifier with the training dataset path."""
        self.train_data_path = train_data_path
        self.random_state = random_state
        self.model = None
        self._load_train_data()
    
    def _load_train_data(self):
        """Load and preprocess training data."""
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError(f"Training dataset not found at {self.train_data_path}")
        
        df = pd.read_csv(self.train_data_path)
        self.y_train = df['class']
        self.X_train = df.drop(columns=['class'])

    def train(self):
        """Train the XGBoost model."""

        # Load the Hyperparameters
        with open(r".\Hyperparams\Xgboost_hparam.json", "r") as f:
            best_params = json.load(f)

        num_classes = len(self.y_train.unique())
        self.model = xgb.XGBClassifier(**best_params, objective='multi:softmax', num_class=num_classes)
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model is not trained. Call train() before predicting.")
        return self.model.predict(X)

    def evaluate(self, test_data_path: str):
        """Evaluate model performance using a separate test dataset."""
        if self.model is None:
            raise ValueError("Model is not trained. Call train() before evaluation.")

        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test dataset not found at {test_data_path}")

        df_test = pd.read_csv(test_data_path)
        y_test = df_test['class']
        X_test = df_test.drop(columns=['class'])

        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(X_test)

        metrics = {
            "Train Accuracy": accuracy_score(self.y_train, y_train_pred),
            "Test Accuracy": accuracy_score(y_test, y_test_pred),
            "Precision": precision_score(y_test, y_test_pred, average="weighted"),
            "Recall": recall_score(y_test, y_test_pred, average="weighted"),
            "F1-score": f1_score(y_test, y_test_pred, average="weighted"),
            "Confusion Matrix": confusion_matrix(y_test, y_test_pred),
            "Classification Report": classification_report(y_test, y_test_pred)
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

        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
        self.model.save_model(save_path)
        print(f"Model saved at: {save_path}")

if __name__ == "__main__":
    # Set training dataset path
    train_dataset_path = r"..\data\processed\train_data.csv"
    test_dataset_path = r"..\data\processed\test_data.csv"
    # Initialize classifier with training data
    classifier = XGBoostClassifier(train_data_path=train_dataset_path)
    classifier.train()

    # Evalueate
    classifier.evaluate(test_dataset_path)

    # Save the trained model
    model_save_path = r".\saved_models\xgb_model.json"
    classifier.save_model(model_save_path)
