# ---------------------------
# Interpret correlations and visualize data with various plots
# ---------------------------

# calculate correlation matrix
corr_matrix = df.corr(numeric_only=True)

# plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="gray", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Health & Lifestyle Dataset", fontsize=16)
plt.show()


# Look for features that have a strong positive or negative (absolute) correlation with your target (e.g., HeartDisease).
# Keep features highly correlated with the target and drop those with low correlation.
# Also, check for multicollinearity (features highly correlated with each other) and consider dropping one of the correlated features to reduce redundancy.

df = df.drop(columns=['ChestPainType', 'RestingECG'])

#set the descending style correlation
corr_with_target = corr_matrix['HeartDisease'].abs().sort_values(ascending=False)
print(corr_with_target)

#  Visualise the correlation matrix again after dropping some columns ------>> Graph visualisation ----->>  Scale/normalize numeric features
# Use StandardScaler or MinMaxScaler to bring values to a similar range do for all numerical ones that are non-categorical
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=np.number))






# Line Plot → Shows trends and growth patterns over time.
# Bar Chart → Compares categories at a glance.
# Histogram → Reveals distribution and frequency of values.
# Heatmap → Highlights correlations and patterns between variables.
# Boxplot → Detects outliers and visualizes data spread.
# Scatter / Bubble Plot → Shows relationships and clusters between two or more variables.
# Pie Chart → Displays proportion of categories in the dataset.




# Histogram ------- for 1 feature                                         || Categorical ||
plt.figure(figsize=(10,6))
if 'value' in df.columns:
    plt.hist(df['value'], bins=30, edgecolor='k')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Value Distribution')
    plt.show()


#  ---------------- for multiple features
import matplotlib.pyplot as plt
features = ['Sex', 'RestingBP', 'Cholesterol']
plt.figure(figsize=(15, 5))
for i, col in enumerate(features, 1):
    plt.subplot(1, 3, i)   # 1 row, 3 columns
    plt.hist(df[col], bins=30, edgecolor='k')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()




# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')    
plt.show()
 



# Scatterplot  - 1 feature                                                 ( for regression types) - 1 feature
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(df.index, df['RestingBP'], alpha=0.6, edgecolor='k')
plt.xlabel('Index')
plt.ylabel('RestingBP')
plt.title('Scatter Plot of RestingBP')
plt.show()


# Scatterplot ( for numerical data or regression types) - multiple feature

import seaborn as sns
plt.figure(figsize=(10,8))
features = ['RestingBP', 'Cholesterol', 'MaxHR', 'Age']
sns.pairplot(df[features], diag_kind='hist')
plt.show()





# Pie chart
plt.figure(figsize=(8,8))  
if 'category' in df.columns:
    df['category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
    plt.title('Pie Chart of Category Distribution')
    plt.ylabel('')
    plt.show()



# Line plot
if '2024' in df.columns and 'Country' in df.columns:  
    df_sorted = df.sort_values('2024', ascending=True) 
    plt.figure(figsize=(12,8))
    plt.barh(df_sorted['Country'], df_sorted['2024'], color='skyblue') # barh = horizontal bar , use plot() for vertical bar
    plt.xlabel("GDP in 2024")
    plt.ylabel("Country")
    plt.title("GDP by Country in 2024")
    plt.show()
else:
    print("Error: Columns not found!")

# Bar chart
plt.figure(figsize=(10,6))
if 'category' in df.columns and 'value' in df.columns:
    sns.barplot(x='category', y='value', data=df)
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('Bar Chart of Value by Category')
    plt.show()
