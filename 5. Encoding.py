
# ---------------------------
# Encode Categorical Variables
# ---------------------------


# -------------------------------------------------------------------------------------------------------------------
#  Hard-Encoding: if you know what each variety actually means like gender, then you can manually map them to numbers
#  Label-Encoding: if you have categories whose meaning is not understandable 
#  One-hot-encoding: if you have categories where you can put one in 1 and rest 0 then one hot Encoding(like yes,no)
# -------------------------------------------------------------------------------------------------------------------

# Label Encoding for binary categories : Best for ordinal data (where order matters: Low < Medium < High). Each unique category is assigned an integer.
from sklearn.preprocessing import LabelEncoder()  # NEED IMPORT IN LABEL BUT BYDEFAULT FOR ONE-HOT

df = pd.DataFrame({
    'Education': ['High School', 'Bachelors', 'Masters', 'PhD', 'Bachelors']})
y = LabelEncoder()
df['Education_Encoded'] = y.fit_transform(df['Education'])
print(df)




# One-Hot Encoding : Best for nominal categorical data (no order)
df_encoded = pd.get_dummies(df, columns=['Color'])
print(df_encoded)




# Ordinal Encoding: Assigns custom integer values based on order. Useful when categories have a natural ranking.
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({'Size': ['Small', 'Medium', 'Large', 'Medium']})
size_order = [['Small', 'Medium', 'Large']]
oe = OrdinalEncoder(categories=size_order)
df['Size_Encoded'] = oe.fit_transform(df[['Size']])
print(df)





# ---------------------------
# 5. Feature Engineering (Optional)
# ---------------------------
# Example: create BMI from weight and height
if 'weight' in df.columns and 'height' in df.columns:
    df['BMI'] = df['weight'] / (df['height']/100)**2

# Convert the datatypes to keep the same all
df["2020"] = df["2020"].astype(float)  - # for one

year_cols = ["2020","2021","2022","2023","2024","2025"] - # for multiple
df[year_cols] = df[year_cols].astype(float)
