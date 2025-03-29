import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import BernoulliNB as SklearnBernoulliNB
from sklearn.preprocessing import LabelEncoder

class BernoulliNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = None
        self.feature_prob_ = None
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        self.class_prior_ = class_counts / n_samples
        self.feature_prob_ = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]  # Ensure y is a NumPy array to avoid issues
            feature_counts = np.sum(X_c, axis=0)
            self.feature_prob_[i] = (feature_counts + self.alpha) / (class_counts[i] + 2 * self.alpha)
        
        return self
    
    def predict_proba(self, X):
        X = np.asarray(X)
        n_samples, n_classes = X.shape[0], len(self.classes_)
        log_proba = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            log_prior = np.log(self.class_prior_[i])
            feature_present = np.log(self.feature_prob_[i]) * X
            feature_absent = np.log(1 - self.feature_prob_[i]) * (1 - X)
            log_proba[:, i] = log_prior + np.sum(feature_present + feature_absent, axis=1)
        
        log_proba_shifted = log_proba - np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba_shifted)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]