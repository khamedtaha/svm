import numpy as np

class SVM:
   def __init__(self, learning_rate=0.001, lambda_param=0.001, n_iters=1000, C=1.0, verbose=True):
      # Initialize hyperparameters
      self.lr = learning_rate             # Learning rate controls how much the model adjusts per update
      self.lambda_param = lambda_param    # Regularization parameter to avoid overfitting (L2)
      self.n_iters = n_iters              # Number of training iterations
      self.verbose = verbose              # If True, print training progress
      self.C = C                          # Soft margin parameter (penalty for misclassification)

      # Initialize weights and bias
      self.w = None
      self.b = None

   def fit(self, X, y):
      # Convert inputs to NumPy arrays
      X = np.array(X)
      y = np.array(y)

      n_samples, n_features = X.shape  # Number of samples and features

      # Convert labels from 0/1 to -1/1 as required by the SVM loss function
      y_ = np.where(y <= 0, -1, 1)

      # Initialize weights and bias with zeros
      self.w = np.zeros(n_features)
      self.b = 0

      # Start training
      for epoch in range(self.n_iters):
         for idx, x_i in enumerate(X):
               # Check the SVM condition: y_i (w·x_i + b) >= 1
               condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

               if condition:
                  # No misclassification; only apply regularization to weights
                  self.w -= self.lr * (2 * self.lambda_param * self.w)
               else:
                  # Misclassification; update weights and bias with hinge loss gradient
                  self.w -= self.lr * (2 * self.lambda_param * self.w - self.C * np.dot(x_i, y_[idx]))
                  self.b += self.lr * self.C * y_[idx]

         # Optionally print training accuracy every 1000 epochs
         if self.verbose and epoch % 1000 == 0:
               predictions = self.predict(X)
               acc = np.mean(predictions == y_)
               print(f"Epoch {epoch}: Training Accuracy = {acc:.4f}")

   def predict(self, X):
      # Convert inputs to NumPy array
      X = np.array(X)

      # Compute decision boundary: w·x + b
      approx = np.dot(X, self.w) + self.b

      # Return sign of the output: -1 or 1
      return np.sign(approx)
