import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class BernoulliNB:
    def __init__(self, alpha=1.0):
        #Initialize Bernoulli Naive Bayes classifier with Laplace smoothing.
        self.alpha = alpha
        self.classes_ = None
        self.class_priors_ = None
        self.feature_prob_ = None
        self.n_features_ = None
        self.feature_names_ = None
        
    def fit(self, X, y):
        

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
            X = X.values
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
            
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Calculate class priors (probability of each class)
        self.class_priors_ = class_counts / n_samples
        
        # Initialize feature probability matrices
        self.feature_prob_ = np.zeros((self.n_classes_, self.n_features_))
        
        # Calculate feature probabilities for each class
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]  
            n_c = X_c.shape[0]  
            
            # Count occurrences of feature=1 for each feature in class c
            feature_counts = np.sum(X_c, axis=0)
            
            # Calculate probabilities with Laplace smoothing
            self.feature_prob_[i] = (feature_counts + self.alpha) / (n_c + 2 * self.alpha)
            
        return self
    
    def predict_proba(self, X):
        #Return probability estimates for samples in X.

        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Initialize the log probability matrix
        log_proba = np.zeros((X.shape[0], self.n_classes_))
        
        # Calculate log probability for each class
        for i, c in enumerate(self.classes_):
            # Prior probability of the class (in log space)
            class_prior = np.log(self.class_priors_[i])
            
            # Calculate Bernoulli probability for features
            # P(x|c) = P(x=1|c)^x * (1-P(x=1|c))^(1-x) for each feature x
            # In log space: log(P(x|c)) = x*log(P(x=1|c)) + (1-x)*log(1-P(x=1|c))
            feature_probs_c = self.feature_prob_[i]
            
            # For x=1: log(P(x=1|c))
            term_1 = X * np.log(feature_probs_c)
            
            # For x=0: log(1-P(x=1|c))
            term_0 = (1 - X) * np.log(1 - feature_probs_c)
            
            # Sum log probabilities over all features and add class prior
            log_proba[:, i] = class_prior + np.sum(term_1 + term_0, axis=1)
        
        log_proba_exp = log_proba - np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba_exp)
        proba /= np.sum(proba, axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X):
        #Perform classification on samples in X.

        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def score(self, X, y):
        #Calculate accuracy score for the classifier.

        if isinstance(y, pd.Series):
            y = y.values
            
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


if __name__ == "__main__":
    BASE_DIR = r"C:\Users\Pratyush\OneDrive\Desktop\ChampSim-master\Network_Intrusion\API\data\processed"
    train_path = os.path.join(BASE_DIR, "train_data.csv")
    test_path = os.path.join(BASE_DIR, "test_data.csv")
    
    # Load the data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separate features and target using the approach you provided
    target_column = train_data.columns[-1]
    
    X_train = train_data.drop(columns=[target_column]).values
    y_train = train_data[target_column].values
    
    X_test = test_data.drop(columns=[target_column]).values
    y_test = test_data[target_column].values
    
    # If target is categorical, encode it
    if isinstance(y_train[0], (str, np.str_)):
        print("Encoding categorical target variable...")
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        print(f"Classes: {label_encoder.classes_}")
    
    # Convert numerical features to binary (Bernoulli NB works with binary features)
    # This step depends on your specific dataset - adjust threshold as needed
    X_train_binary = (X_train > 0).astype(int)
    X_test_binary = (X_test > 0).astype(int)
    
    # Create and train the BernoulliNB model
    bnb = BernoulliNB(alpha=1.0)
    bnb.fit(X_train_binary, y_train)
    