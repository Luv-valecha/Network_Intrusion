import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import pickle, json

#add ModelEvaluator from evaluate.py
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.evaluate import ModelEvaluator

class RBFKernelSVM:
    def __init__(self, C=1.0, gamma=1.0, tol=1e-3, max_iter=1000):
        self.C = C  # Regularization parameter 
        self.gamma = gamma  # RBF kernel parameter
        self.tol = tol  # Tolerance for optimization
        self.max_iter = max_iter
        self.alphas = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_indices = None
        self.model = None
        
    def _rbf_kernel(self, x1, x2):
        """
        Compute the RBF kernel between x1 and x2
        
        K(x, y) = exp(-gamma * ||x - y||^2)
        """
        if not isinstance(x1, np.ndarray):
            x1 = np.array(x1)
        if not isinstance(x2, np.ndarray):
            x2 = np.array(x2)

        if len(x1.shape) == 1:
            x1 = x1.reshape(1, -1)
        if len(x2.shape) == 1:
            x2 = x2.reshape(1, -1)
        
        # calculate squared Euclidean distance 
        x1_norm = np.sum(x1**2, axis=1).reshape(-1, 1)
        x2_norm = np.sum(x2**2, axis=1).reshape(1, -1)
        distances = x1_norm + x2_norm - 2 * np.dot(x1, x2.T)
        
        # Apply RBF kernel formula
        return np.exp(-self.gamma * np.maximum(0, distances))  
            
    def train(self, X, y, verbose=False):
        """
        Train the SVM model using a simplified SMO-like algorithm
        """

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        n_samples, n_features = X.shape
        
        # Convert y to -1, 1 format
        y_binary = np.where(y <= 0, -1, 1)
        
        # Initialize alphas and bias
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        # pre calculate the kernel matrix
        K = self._rbf_kernel(X, X)
        
        # to store optimization history
        history = []
        
        # Main optimization loop
        iterations = 0
        passes = 0
        max_passes = 5  # Maximum number of passes over the dataset without changes
        
        iterator = tqdm(range(self.max_iter)) if verbose else range(self.max_iter)
        
        while passes < max_passes and iterations < self.max_iter:
            num_changed_alphas = 0
            
            # Loop over all training examples
            for i in range(n_samples):
                # Calculate error Error_i = f(x_i) - y_i
                f_xi = np.sum(self.alphas * y_binary * K[i, :]) + self.b
                Error_i = f_xi - y_binary[i]
                
                # Check if example violates KKT conditions
                if ((y_binary[i] * Error_i < -self.tol and self.alphas[i] < self.C) or
                    (y_binary[i] * Error_i > self.tol and self.alphas[i] > 0)):
                    
                    # Randomly select second example 
                    j = np.random.randint(0, n_samples - 1)  # pick random index
                    j += (j >= i)  # increment j if j is greater or equal to i
                    
                    # Calculate Error_j
                    f_xj = np.sum(self.alphas * y_binary * K[j, :]) + self.b
                    Error_j = f_xj - y_binary[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Compute bounds L and H every iteration
                    if y_binary[i] != y_binary[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                        
                    if L == H:
                        continue
                    
                    # Compute eta = 2*K_ij - K_ii - K_jj u
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    
                    # make sure Eta be negative so that positive-definite kernel is there
                    if eta >= 0:
                        continue
                    
                    # Compute new alpha_j
                    self.alphas[j] = alpha_j_old - (y_binary[j] * (Error_i - Error_j)) / eta
                    
                    # Clip alpha_j to [L, H]
                    self.alphas[j] = max(L, min(H, self.alphas[j]))
                    
                    # Check if alpha_j changed significantly
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # change  alpha_i: alpha_i = alpha_i_old + y_i*y_j*(alpha_j_old - alpha_j)
                    self.alphas[i] = alpha_i_old + y_binary[i] * y_binary[j] * (alpha_j_old - self.alphas[j])
                    
                    # Update threshold b
                    b1 = self.b - Error_i - y_binary[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - \
                        y_binary[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    
                    b2 = self.b - Error_j - y_binary[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - \
                        y_binary[j] * (self.alphas[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            #increase the passes if num_changed_alphas is zero else make passes equal to zero
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
                
            iterations += 1
            
            if verbose and iterations % 10 == 0:
                obj = 0.5 * np.sum(self.alphas * self.alphas * y_binary * y_binary * K) - np.sum(self.alphas)
                history.append(obj)
                print(f"Iteration {iterations}, objective: {obj:.6f}, changed alphas: {num_changed_alphas}")
        
        # Extract support vectors
        sv_idx = np.where(self.alphas > 1e-5)[0]
        self.support_vector_indices = sv_idx
        self.alphas = self.alphas[sv_idx]
        self.support_vectors = X[sv_idx]
        self.support_vector_labels = y_binary[sv_idx]
            
        return history
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        decision_vals = self.decision_function(X)
        return np.where(decision_vals < 0, 0, 1)
    
    def decision_function(self, X):
        """
        Compute the decision function values
        """
        if self.support_vectors is None:
            raise ValueError("Model not fitted yet!")
            
        # Compute kernel between X and support vectors
        K = self._rbf_kernel(X, self.support_vectors)
        
        # Compute decision function
        return np.dot(K, self.alphas * self.support_vector_labels) + self.b
    
    def score(self, X, y):
        """
        Calculate accuracy
        """
        predictions = self.predict(X)
        correct_count = np.sum(predictions == y)
        return correct_count/len(y)
    
    # function to save the model in pkl file
    def save_model(self, filename):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
            print(f"Model saved to {filename}")
        except (OSError, IOError) as e:
            print(f"Error saving model: {e}")



# Main execution code
if __name__ == "__main__":

    #base directory for training and testing dataset
    BASE_DIR = r"API\data\processed"
    train_path = os.path.join(BASE_DIR, "train_data.csv")
    test_path = os.path.join(BASE_DIR, "test_data.csv")
    
    #load test and train dataset
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    target_column = train_data.columns[-1]  

    #drop target column from training and test dataset
    X_train = train_data.drop(columns=[target_column]).values
    y_train = train_data[target_column].values

    X_test = test_data.drop(columns=[target_column]).values
    y_test = test_data[target_column].values

    max_train_samples = 5000  #check training sample size if bigger then reduce size to reduce load on hardware
    if X_train.shape[0] > max_train_samples:
        
        # Ensure balanced sampling
        class_indices = {}
        for cls in np.unique(y_train):
            class_indices[cls] = np.where(y_train == cls)[0]
        
        # Calculate samples per class
        samples_per_class = max_train_samples // len(class_indices)
        
        # Sample indices
        sampled_indices = []
        for cls, indices in class_indices.items():
            if len(indices) > samples_per_class:
                sampled_indices.extend(np.random.choice(indices, samples_per_class, replace=False))
            else:
                sampled_indices.extend(indices)
        
        # Create subset
        X_train_subset = X_train[sampled_indices]
        y_train_subset = y_train[sampled_indices]
        
    else:
        X_train_subset = X_train
        y_train_subset = y_train

    rbf_svm = RBFKernelSVM(C=100.0, gamma=0.001, max_iter=100, tol=1e-3)

    loss_history = rbf_svm.train(X_train_subset, y_train_subset, verbose=True)

    # Save the trained model
    model_save_path = r"API\model\saved_models\Non_LinearSVC.pkl"
    rbf_svm.save_model(model_save_path)

    # evaluate model 
    evaluater = ModelEvaluator("Non_LinearSVC.pkl")
    evaluater.evaluate()