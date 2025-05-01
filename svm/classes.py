import numpy as np

class SVM:
   def __init__(self, learning_rate=0.001, lambda_param=0.001, n_iters=1000, C=1.0 ,  verbose=True):
      
      self.lr = learning_rate                 # Learning rate
      self.lambda_param = lambda_param        # Regularization parameter
      self.n_iters = n_iters                  # Number of iterations
      self.verbose = verbose                  # Verbose flag for training progress
      self.C = C                              # soft margin parameter
      self.w = None 
      self.b = None 

   def fit(self, X, y):
      # Ensure input is a NumPy array
      X = np.array(X)
      y = np.array(y)

      n_samples, n_features = X.shape #

      # Convert labels to -1 and 1
      y_ = np.where(y <= 0, -1, 1)

      # Initialize weights and bias
      self.w = np.zeros(n_features)
      self.b = 0

      for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X): # enumerate allows us to get both index and value
               
               condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
               
               if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
               else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - self.C * np.dot(x_i, y_[idx]))
                    self.b += self.lr * self.C * y_[idx]

            # print training progress
            if self.verbose and epoch % 1000 == 0:
               predictions = self.predict(X)
               acc = np.mean(predictions == y_)
               print(f"Epoch {epoch}: Training Accuracy = {acc:.4f}")

   def predict(self, X):
      X = np.array(X)
      approx = np.dot(X, self.w) + self.b
      return np.sign(approx)
