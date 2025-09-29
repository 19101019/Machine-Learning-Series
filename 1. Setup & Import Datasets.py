# =========================================================
# Machine Learning End-to-End Template
# =========================================================




# ---------------------------
# 0. Install dataset from Kaggle
# ---------------------------
# START WITH THIS ---->
# !pip install opendatasets
import opendatasets as od
od.download("https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
# ====={"username":"technozsoftware","key":"287471c7934797f80ef81bc64e4d9ea6"}=============




# ---------------------------
# 1. Install dataset from CSV file
# ---------------------------
import pandas as pd
df = pd.read_csv("file.csv")




# ---------------------------
# 2. Install dataset from Excel file
# ---------------------------
df = pd.read_excel("file.xlsx")                           # First sheet
df = pd.read_excel("file.xlsx", sheet_name="Sheet1")      # Specific sheet




# ---------------------------
#3. Install dataset from JSON file
# ---------------------------
df = pd.read_json("file.json")

#╒══════════════════════════════════════════════════════════╕
#│ ⚡ Nested JSON Example → Flatten into Pandas DataFrame     
#╞══════════════════════════════════════════════════════════╡
#│ data = {                                                  │
#│     "name": "Bob",                                        │
#│     "age": 30,                                            │
#│     "address": {                                          │
#│         "city": "Mumbai",                                 │
#│         "pin": 400001                                     │
#│     }                                                     │
#│ }                                                         │
#│                                                           │
#│ df = pd.json_normalize(data)                              │
#│ print(df)                                                 │
#╘══════════════════════════════════════════════════════════╛
#============================================================

#╒════════╤═════╤═══════════════╤═══════════════╕
#│ name   │ age │ address.city  │ address.pin   │
#╞════════╪═════╪═══════════════╪═══════════════╡
#│ Bob    │  30 │ Mumbai        │        400001 │
#╘════════╧═════╧═══════════════╧═══════════════╛
#==================================================


# ---------------------------
#⚡ Key Point:

# ⚡ pd.json_normalize() is basically a flattening tool that turns nested JSON → clean DataFrame, making it easy for analysis.

# ---------------------------