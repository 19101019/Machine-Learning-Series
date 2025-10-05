
# --------------------------------
#  Split - Train-Test-Validation 
# --------------------------------

from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]  # all columns except last   --- depends as per the position of target column here its at last
y = df.iloc[:, -1]   # last column as target

# Split into training + temp (80% train, 20% temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split temp into validation + test (50% val, 50% test → each 10% of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Check shapes
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)


X_train.info()



#╒════════╤
#│ Model  │ 
#╞════════╪


#⚡  classification -- logistic - svm-decisiontree-randomforest-xgboost                  #⚡
#⚡  regression - linear-ridge-lasso-elasticnet-decisiontree-randomforest-xgboost        #⚡ 
#⚡  clustering - kmeans-hierarchical-dbscan                                             #⚡ 