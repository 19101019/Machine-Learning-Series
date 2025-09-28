# ---------------------------
# To see all rows/columns with True/False for missing values:
df.isnull()
#---------------------------
#To count missing values per column:
df.isnull().sum()
# ---------------------------
#To check if any nulls exist at all:
df.isnull().values.any()
# ---------------------------
# Replace all NaN with 0
df = df.fillna(0)



# ---------------------------
# Handle Missing Values
# ---------------------------
# Drop rows with missing target
df = df.dropna(subset=['target'])

# Check for duplicate rows and drop them 
print(df.duplicated().sum())
df = df.drop_duplicates(inplace=True)   # Use inplace=True → when you want to overwrite the original DataFrame.
                                        # Use inplace=False (or skip it) → when you want to keep the original and create a new cleaned version.
