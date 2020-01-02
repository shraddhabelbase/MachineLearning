import numpy as np
from scipy.optimize import minimize

class LinReg: 
    
    def __init__(self, X, y):
        
        #Store the training data
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_ob = len(y)
        
        def find_sse(beta):
            beta_ = beta[1:].reshape(-1,1)
            y_hat = np.dot(self.X,beta_) + beta[0]
            y_hat = y_hat.reshape(-1,)
            e = self.y - y_hat
            return np.sum(e**2)
    
        #Find bets that minimizes the loss
        beta_guess = np.zeros(self.X.shape[1] + 1)
        min_results = minimize(find_sse, beta_guess)
        self.coefficients = min_results.x
        
        #self.y_predicted = self.predict(self.X)
        
    def predict(self, X):
            X = np.array(X)
            beta = self.coefficients
            beta_ = beta[1:].reshape(-1,1)
            y_hat = np.dot(X, beta_) +beta[0]
            y_hat = y_hat.reshape(-1,)
            return y_hat
        
    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_hat = self.predict(X)
        sse = np.sum(y - y_hat)**2
        sst = np.sum((y - np.mean(y))**2)
        return 1 - sse/sst
        
        
    #def summary(self):
        
    #def score(self, X, y):
X = [[1],[2],[3],[4],[5]]
y = [5.1, 6.8, 9.2, 10.9, 13.1]
X_new = [[1.5], [2.5]]
 
model = LinReg(X, y)
print(model.coefficients)
print(model.predict(X_new))