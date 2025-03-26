import numpy as np

class LinearSVCClassifier:
   
    def __init__(self, learning_rate=0.01, lambda_param=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def train(self, X_train, y_train):
        """
        Train the SVM model using stochastic gradient descent.
        """
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert labels from {0,1} to {-1,1} for hinge loss
        y_train = np.where(y_train <= 0, -1, 1)
        
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X_train):
                condition = y_train[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                
                if condition:
                    # No misclassification, only apply regularization
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # Misclassification, update weights and bias
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_train[idx]))
                    self.bias -= self.learning_rate * y_train[idx]
    
    def predict(self, X_test):
        """
        Predict class labels for given test data.
        """
        approx = np.dot(X_test, self.weights) + self.bias
        return np.where(approx >= 0, 1, 0)
