import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.colors import ListedColormap



def plot_scatter_before_after_scaling(X_before, X_after, y, class_name , feature_names=None ):
   """
   Plot scatter plots before and after scaling with class coloring.

   Parameters:
   - X_before: array-like, shape (n_samples, n_features) — data before scaling.
   - X_after: array-like, shape (n_samples, n_features) — data after scaling.
   - y: array-like, shape (n_samples,) — target class labels (e.g., Purchased).
   - feature_names: list of str — optional names for features.
   - class_name: str — label for the class (used in legend).
   """
   if feature_names is None:
      feature_names = [f"Feature {i}" for i in range(X_before.shape[1])]

   if X_before.shape[1] < 2:
      print("At least 2 features are required for scatter plot.")
      return

   plt.figure(figsize=(12, 5))

   
   plt.subplot(1, 2, 1)
   scatter1 = plt.scatter(X_before[:, 0], X_before[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.8)
   plt.title("Before Scaling")
   plt.xlabel(feature_names[0])
   plt.ylabel(feature_names[1])
   plt.grid(True)
   plt.legend(*scatter1.legend_elements(), title=class_name)

   plt.subplot(1, 2, 2)
   scatter2 = plt.scatter(X_after[:, 0], X_after[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.8)
   plt.title("After Scaling")
   plt.xlabel(feature_names[0])
   plt.ylabel(feature_names[1])
   plt.grid(True)
   plt.legend(*scatter2.legend_elements(), title=class_name)

   plt.tight_layout()
   plt.show()



def plot_decision_boundary_test_data(X_test, y_test, classifier, sc ):

   X_set, y_set = sc.inverse_transform(X_test), y_test
   # Create a grid of points

   X1, X2 = np.meshgrid(
      np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.25),
      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.25)
   )
   # Predict for each point on the grid

   Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
   # Plot the decision boundary

   plt.contourf(X1, X2, Z, alpha=0.75, cmap = ListedColormap(['#FA8072', '#1E90FF']) )
   plt.xlim(X1.min(), X1.max())
   plt.ylim(X2.min(), X2.max())
   # Define colors for scatter plot

   colors = ['#FA8072', '#1E90FF']
   # Plot the test set points

   for i, j in enumerate(np.unique(y_set)):
      plt.scatter(
         X_set[y_set == j, 0], X_set[y_set == j, 1],
         color=colors[i], label=j
      )
   
   # Add titles and labels
   plt.title('SVM (Test set)')
   plt.xlabel('Age')
   plt.ylabel('Estimated Salary')
   plt.legend()
   plt.show()