
# --------------------------------
#  Split - Train-Test-Validation 
# --------------------------------

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
y = df.iloc[:, 5]          # 5th column (charges)
X = df.drop(columns=df.columns[5])  # all columns except 'charges'

# Split into training + temp (80% train, 20% temp)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None  # stratify not used for regression
)

# Split temp into validation + test (50% val, 50% test → each 10% of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=None
)

# Check shapes
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)


# Info of training features
X_train.info()

print("Target column (y):")
print(y.name)  # prints the name of the column
print(y.head())  # first 5 values of y

# Check features (X)
print("\nFeature columns (X):")
print(X.columns)  # prints all feature column names
print(X.head())




#╒════════╤
#│ Model  │ 
#╞════════╪


#⚡  classification -- logistic - svm-decisiontree-randomforest-xgboost                  #⚡
#⚡  regression - linear-ridge-lasso-elasticnet-decisiontree-randomforest-xgboost        #⚡ 
#⚡  clustering - kmeans-hierarchical-dbscan                                             #⚡ 