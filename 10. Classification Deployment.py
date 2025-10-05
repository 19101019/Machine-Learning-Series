from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

# --- 6. EVALUATE THE MODEL ---
# Make predictions on the unseen (scaled) test data
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(class_report)
print("\nConfusion Matrix:")
# A confusion matrix shows where the model got things right and where it got confused.
# The diagonal from top-left to bottom-right shows correct predictions.
print(conf_matrix)
