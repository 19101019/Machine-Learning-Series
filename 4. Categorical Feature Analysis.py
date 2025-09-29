
# ---------------------------
# Categorical features
# ---------------------------

print("\nColumn data types:\n", df.dtypes)                      # datatype of feature

# Use of variations to be use for encoding 
for x in df.columns:
    unique_vals = df[x].unique()                                # Name of each variations
    num_unique = df[x].nunique()                                # How many Types of variations
    value_counts = df[x].value_counts()                         # Qty of each variation
    unique_str = ", ".join(map(str, unique_vals))               # Problem: ", ".join() works only on strings.
    #     "separator".join(iterable_of_strings)                 ( If your unique values are numbers, it will throw an error. So you convert everything to string   )
    #                                                           ( using map(str, unique_vals) before joining. With f-strings, the conversion happens automatically,)
    #                                                           ( so .join() can work directly                                                                     )
    unique_stri = ", ".join([f"{val} ({cnt})" for val, cnt in value_counts.items()])    # unique values and the Qty of each variation inside that column
    print(f"\nFeature Name: {x}")
    print(f"No of Variations: {num_unique}")
    print(f"Name of each variations: {unique_str}")
    print(f"Variation Name & count: {unique_stri}")

   # *********** or ****************
for x in df.columns:
  print(f"{x}:\nName of variations = {df[x].unique()}: \nNumber of variations = {df[x].nunique()}\nQty of variations =\n{df[x].value_counts()}\n")
  print("******"*20)



# All Objects must be handled before model training
# Categorical features must be converted to numerical format using encoding techniques like One-Hot Encoding, Label Encoding, or Ordinal Encoding.
# Missing values in categorical features can be handled by imputing the most frequent category or using a placeholder category like 'Unknown'.  
# Drop features with too many unique categories (high cardinality) as they can lead to overfitting and increased model complexity.