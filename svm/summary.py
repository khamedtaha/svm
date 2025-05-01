from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt 
from tabulate import tabulate


def svm_summary(y_test ,y_pred):
      """
      Summary of SVM model performance.
      This function computes and prints the confusion matrix, accuracy, precision, recall, and F1 score.
      
      """
      cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

      plt.figure(figsize=(6,4))
      sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=["Predicted -1", "Predicted 1"], yticklabels=["Actual -1", "Actual 1"])
      plt.xlabel("Predicted")
      plt.ylabel("Actual")
      plt.title("Confusion Matrix for SVM (-1 as class 0)")
      plt.show()
      
      TN, FP, FN, TP = cm.ravel()

      confusion_data = [
            ["True Positive (TP)", TP],
            ["True Negative (TN)", TN],
            ["False Positive (FP)", FP],
            ["False Negative (FN)", FN]
      ]
      print("\nConfusion Matrix Details :")
      print(tabulate(confusion_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

      accuracy = accuracy_score(y_test, y_pred)


      # Precision and Recall and F1 Score
      precision = TP / (TP + FP) if (TP + FP) != 0 else 0
      recall = TP / (TP + FN) if (TP + FN) != 0 else 0
      f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
      
      metrics = [
         ["Accuracy", f"{accuracy * 100:.2f}%"],
         ["Precision", f"{precision * 100:.2f}%"],
         ["Recall", f"{recall * 100:.2f}%"],
         ["F1 Score", f"{f1_score * 100:.2f}%"]
      ]
      print("\nPerformance Metrics :")
      print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="fancy_grid"))