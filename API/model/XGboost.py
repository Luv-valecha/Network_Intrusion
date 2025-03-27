import os, sys
import pandas as pd
import xgboost as xgb
import pickle, json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# Add the parent directory so that the 'scripts' folder is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import ModelEvaluator 

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

    def save_model(self, save_path: str):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("Model is not trained. Call train() before saving.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
        with open(save_path,"wb") as f:
            pickle.dump(self.model,f)
            print(" Model saved")

if __name__ == "__main__":
    # Set training dataset path
    train_dataset_path = r"..\data\processed\train_data.csv"
    test_dataset_path = r"..\data\processed\test_data.csv"
    # Initialize classifier with training data
    classifier = XGBoostClassifier(train_data_path=train_dataset_path)
    classifier.train()

    # Save the trained model
    model_save_path = r".\saved_models\xgb_model.pkl"
    classifier.save_model(model_save_path)

    # evaluate
    evaluater = ModelEvaluator("xgb_model.pkl")
    evaluater.evaluate()
