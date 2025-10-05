
# ---------------------------------⚡ ⚡ ⚡ ⚡ ⚡ ⚡ classification ⚡ ⚡ ⚡ ⚡ ⚡ ---------------------------------


#╒═══════════════════════╕
#│ Logistic Regression   │
#╘═══════════════════════╛


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define model
model = LogisticRegression(max_iter=1000)

# Train using all columns except target
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
loss = model.score(X_test, y_test)
print(loss)




#╒═════╕
#│ SVM │
#╘═════╛

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define model
model = SVC(kernel='rbf', random_state=42)  # you can change kernel to 'linear', 'poly', 'sigmoid'

# Train using all columns except target
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Score (same as accuracy)
loss = model.score(X_test, y_test)
print("Accuracy:", loss)




#╒═══════════════╕
#│ Decision Tree │
#╘═══════════════╛

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Define model
model = DecisionTreeClassifier(random_state=42)

# Train using all columns except target
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Score (same as accuracy)
loss = model.score(X_test, y_test)
print("Accuracy:", loss)


#╒═══════════════╕
#│ Random Forest │
#╘═══════════════╛

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define model
model = RandomForestClassifier(random_state=42, n_estimators=100)

# Train using all columns except target
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Score (same as accuracy)
loss = model.score(X_test, y_test)
print("Accuracy:", loss)



#╒═════════╕
#│ XGBoost │
#╘═════════╛

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Define model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train using all columns except target
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Score (same as accuracy)
loss = model.score(X_test, y_test)
print("Accuracy:", loss)





# ---------------------------------⚡ ⚡ ⚡ ⚡ ⚡ ⚡ Regression ⚡ ⚡ ⚡ ⚡ ⚡ ---------------------------------

#╒════════╕
#│ Linear │
#╘════════╛
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Score
score = model.score(X_test, y_test)  # R^2 score
print("R2 Score (Linear Regression):", score)



#╒═══════╕
#│ Ridge │
#╘═══════╛
from sklearn.linear_model import Ridge

# Define model
model = Ridge(alpha=1.0, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Score
score = model.score(X_test, y_test)
print("R2 Score (Ridge):", score)




#╒═══════╕
#│ Lasso │
#╘═══════╛
from sklearn.linear_model import Lasso

# Define model
model = Lasso(alpha=0.1, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Score
score = model.score(X_test, y_test)
print("R2 Score (Lasso):", score)




#╒═══════════════╕
#│ ElasticNet    │
#╘═══════════════╛
from sklearn.linear_model import ElasticNet

# Define model
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Score
score = model.score(X_test, y_test)
print("R2 Score (ElasticNet):", score)



#╒═══════════════╕
#│ Decision Tree │
#╘═══════════════╛
from sklearn.tree import DecisionTreeRegressor

# Define model
model = DecisionTreeRegressor(random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Score
score = model.score(X_test, y_test)
print("R2 Score (Decision Tree):", score)



#╒═══════════════╕
#│ Random Forest │
#╘═══════════════╛
from sklearn.ensemble import RandomForestRegressor

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Score
score = model.score(X_test, y_test)
print("R2 Score (Random Forest):", score)





#╒═════════╕
#│ XGBoost │
#╘═════════╛

from xgboost import XGBRegressor

# Define model
model = XGBRegressor(random_state=42, eval_metric='rmse')

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Score
score = model.score(X_test, y_test)
print("R2 Score (XGBoost):", score)




# ---------------------------------⚡ ⚡ ⚡ ⚡ ⚡ ⚡ Clustering ⚡ ⚡ ⚡ ⚡ ⚡ ---------------------------------


#╒════════╕
#│ KMeans │
#╘════════╛
from sklearn.cluster import KMeans

# Define model
model = KMeans(n_clusters=3, random_state=42)

# Train / Fit
model.fit(X_train)  # Clustering uses features only

# Predict cluster labels
y_pred = model.predict(X_test)

# Print cluster assignments for test set
print("Cluster labels (KMeans):", y_pred)



#╒════════════════╕
#│ Hierarchical   │
#╘════════════════╛

from sklearn.cluster import AgglomerativeClustering

# Define model
model = AgglomerativeClustering(n_clusters=3)

# Train / Fit
model.fit(X_train)

# Predict cluster labels
y_pred = model.fit_predict(X_test)

# Print cluster assignments for test set
print("Cluster labels (Hierarchical):", y_pred)




#╒════════╕
#│ DBSCAN │
#╘════════╛
from sklearn.cluster import DBSCAN

# Define model
model = DBSCAN(eps=0.5, min_samples=5)

# Train / Fit
model.fit(X_train)

# Predict cluster labels
y_pred = model.fit_predict(X_test)

# Print cluster assignments for test set
print("Cluster labels (DBSCAN):", y_pred)


