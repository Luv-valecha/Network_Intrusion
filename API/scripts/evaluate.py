import os,sys
import pickle
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)

class ModelEvaluator:
    def __init__(self, pkl_file: str):
        """
        Initialize the ModelEvaluator with the model and test dataset paths.
        """
        # form the saves models path
        model_path = os.path.join(r"..\model\saved_models",pkl_file)

        test_data_path = r"..\data\processed\test_data.csv"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test dataset not found at {test_data_path}")
        
        # Load the trained model
        with open (model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load the test dataset
        self.df_test = pd.read_csv(test_data_path)
        print("Test dataset loaded successfully!")

        # Split into features and labels
        self.y_test = self.df_test['class']
        self.X_test = self.df_test.drop(columns=['class'])

    def negative_predictive_value(self, y_true, y_pred):
        """Calculate Negative Predictive Value (NPV)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0

    def evaluate(self):
        """Evaluate the model and print metrics."""
        print("\nEvaluating Model...")

        # Predictions
        y_test_pred = self.model.predict(self.X_test)

        # Compute metrics
        metrics = {
            "Test Accuracy": accuracy_score(self.y_test, y_test_pred),
            "Precision": precision_score(self.y_test, y_test_pred, average="weighted"),
            "Recall": recall_score(self.y_test, y_test_pred, average="weighted"),
            "F1-score": f1_score(self.y_test, y_test_pred, average="weighted"),
            "NPV (Negative Predictive Value)": self.negative_predictive_value(self.y_test, y_test_pred),
            "Confusion Matrix": confusion_matrix(self.y_test, y_test_pred),
            "Classification Report": classification_report(self.y_test, y_test_pred)
        }

        # Print the evaluation results
        print("\nEvaluation Metrics:")
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}:\n{value}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <model_filename.pkl>")
        sys.exit(1)  # Exit with an error code

    model_filename = sys.argv[1]  # Get the pickle file name from command-line args
    evaluator = ModelEvaluator(model_filename)
    evaluator.evaluate()