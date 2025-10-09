# Assuming 'model' is your already trained model object
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X_train_scaled, y_train)

# --- Add these lines to your code ---

import joblib

# 1. Define the filename for your model
model_filename = 'iris_classifier_model.joblib'

# 2. Save the trained model to a file
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# --- How to load and use it later ---

# 3. Load the model from the file in another script or later on
loaded_model = joblib.load(model_filename)

print("Model loaded successfully!")

# 4. You can now use the loaded_model to make predictions
# Make sure the new data has the same scaling/preprocessing as the training data
# example_prediction = loaded_model.predict(some_new_scaled_data)
# print(f"Prediction for new data: {example_prediction}")