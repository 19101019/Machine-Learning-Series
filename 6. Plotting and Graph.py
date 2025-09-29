# ---------------------------
# 6. Exploratory Data Analysis (EDA)
# ---------------------------

# Line Plot → Shows trends and growth patterns over time.
# Bar Chart → Compares categories at a glance.
# Histogram → Reveals distribution and frequency of values.
# Heatmap → Highlights correlations and patterns between variables.
# Boxplot → Detects outliers and visualizes data spread.
# Scatter / Bubble Plot → Shows relationships and clusters between two or more variables.
# Pie Chart → Displays proportion of categories in the dataset.


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


# Histogram
plt.figure(figsize=(10,6))
if 'value' in df.columns:
    plt.hist(df['value'], bins=30, edgecolor='k')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Value Distribution')
    plt.show()


# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')    
plt.show()
 

# Pie chart
plt.figure(figsize=(8,8))  
if 'category' in df.columns:
    df['category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
    plt.title('Pie Chart of Category Distribution')
    plt.ylabel('')
    plt.show()
