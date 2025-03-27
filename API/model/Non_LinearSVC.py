import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

#implementation of non linear svm using  Sequential Minimal Optimization (SMO) method
class SVM_NonLinear:
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0, tol=1e-3, max_iter=100):
        self.C = C
        self.kernel = self.rbf_kernel if kernel == 'rbf' else self.linear_kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_labels = None
        self.support_alpha = None
    
    #formulae for linear kernel
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)
    
    #formulae for Radial Basis Function (RBF)
    def rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * norm(x1 - x2) ** 2)
    
    #function to calculate error
    def compute_error(self, i):
        return self.decision_function(self.X[i]) - self.y[i]
    
    #frunction to give label
    def decision_function(self, X):
        return np.sum(self.support_alpha * self.support_labels * np.array([self.kernel(sv, X) for sv in self.support_vectors])) + self.b
    

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0

        for _ in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)  #contains previous alpha value
            
            for i in range(n_samples):
                E_i = self.compute_error(i)  #calculate error of each data points
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    E_j = self.compute_error(j)

                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    
                    #calculate L and H bound
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue

                    eta = 2 * self.kernel(X[i], X[j]) - self.kernel(X[i], X[i]) - self.kernel(X[j], X[j])
                    if eta >= 0:
                        continue
                    
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    #update alpha[i]
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    #update new bias
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * self.kernel(X[i], X[i]) - y[j] * (self.alpha[j] - alpha_j_old) * self.kernel(X[i], X[j])
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * self.kernel(X[i], X[j]) - y[j] * (self.alpha[j] - alpha_j_old) * self.kernel(X[j], X[j])
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
            
            #check whether it is converging or not
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break
        
        idx = self.alpha > 0
        self.support_vectors = X[idx]
        self.support_labels = y[idx]
        self.support_alpha = self.alpha[idx]
    
    #function for predictions
    def predict(self, X):
        return np.sign(np.array([self.decision_function(x) for x in X]))
