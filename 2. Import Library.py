

# ---------------------------
# 1. Import Libraries
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


# ---------------------------
# 2. Load Dataset
# ---------------------------
# Replace 'dataset.csv' with your dataset path
df = pd.read_csv("dataset.csv")


# Quick overview
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())