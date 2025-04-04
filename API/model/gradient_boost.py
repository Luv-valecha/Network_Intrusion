import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import sys

# Adding ModelEvaluator from evaluate.py
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import ModelEvaluator

class GradientBoostingModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        """
        Initialize Gradient Boosting model with scikit-learn.
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting stages/trees
        learning_rate : float
            Shrinks the contribution of each tree
        max_depth : int
            Maximum depth of individual regression trees
        random_state : int
            Controls randomness for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.feature_names_ = None
        
    def fit(self, X, y):
        """
        Fit the Gradient Boosting model to the training data.
        
        Parameters:
        -----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Training data features
        y : array-like or series of shape (n_samples,)
            Target values
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
            X_values = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
            X_values = X
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        self.model.fit(X_values, y_values)
        
        # Store classes for convenience similar to your BernoulliNB implementation
        self.classes_ = self.model.classes_
        
        return self
    
    def predict_proba(self, X):
        """
        Return probability estimates for samples in X.
        
        Parameters:
        -----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Test data features
            
        Returns:
        --------
        proba : array of shape (n_samples, n_classes)
            Probability estimates
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        return self.model.predict_proba(X_values)
    
    def predict(self, X):
        """
        Perform classification on samples in X.
        
        Parameters:
        -----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Test data features
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted classes
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        return self.model.predict(X_values)
    
    def score(self, X, y):
        """
        Calculate accuracy score for the classifier.
        
        Parameters:
        -----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Test data features
        y : array-like or series of shape (n_samples,)
            True labels
            
        Returns:
        --------
        score : float
            Mean accuracy
        """
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        return self.model.score(X, y_values)
    
    def feature_importances(self):
        """
        Return feature importance scores.
        
        Returns:
        --------
        feature_importances : dict
            Dictionary of feature names and their importance scores
        """
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names_, importances))
    
    def save_model(self, filename):
        """
        Save the model to a pickle file.
        
        Parameters:
        -----------
        filename : str
            Path to save the model
        """
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print(f"Model saved to {filename}")
        except (OSError, IOError) as e:
            print(f"Error saving model: {e}")


# Main function execution code
if __name__ == "__main__":
    BASE_DIR = r"API\data\processed"
    train_path = os.path.join(BASE_DIR, "train_data.csv")
    test_path = os.path.join(BASE_DIR, "test_data.csv")
    
    # Loading the data
    print("Loading data...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separate features and target
    target_column = train_data.columns[-1]
    
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    
    # If target is categorical, encode it
    if isinstance(y_train.iloc[0], (str, np.str_)):
        print("Encoding categorical target variable...")
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        print(f"Classes: {label_encoder.classes_}")
    
 
    # So we don't need to binarize the features
    print("Training Gradient Boosting model...")
    gb_model = GradientBoostingModel(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    
    # Evaluate model using your existing evaluator
    print("\nEvaluating model...")
    evaluator = ModelEvaluator("GradientBoostingClassifier.pkl")
    evaluator.evaluate()