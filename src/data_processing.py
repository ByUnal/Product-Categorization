import pandas as pd
from tqdm import tqdm

df = pd.read_csv("../data/ProductData.csv", low_memory=False)

# Fill null values as "Unclassified
df["category"] = df["category"].fillna("Unclassified")
df["subcategory"] = df["subcategory"].fillna("Unclassified")
df["detail_category"] = df["detail_category"].fillna("Unclassified")

# Create new DataFrame with necessary field for training
data = df[["product_name", "category", "subcategory", "detail_category"]].copy()

print("Sum of null values in DataFrame: ", data.isna().sum().sum())

# drop null values
data = data.dropna()

print("Duplicated values in DataFrame: ", data.duplicated(keep='last').sum())

# Delete duplicated values by keeping last duplicate row
data = data.drop_duplicates(keep='last')

print(f"Final shape of DataFrame is {data.shape}")

# Merge relevant categories into one column
data['categories'] = data['category'].astype(str) + "," + data['subcategory'].astype(str) + "," + data[
    'detail_category'].astype(str)

# How many unique terms?
print("Count of Unique terms:", data["categories"].nunique())
print("Data occurrence as low as 1:", sum(data["categories"].value_counts() == 1))

# As observed above, out of 4,191 unique combinations of terms, 1,144 entries have the lowest occurrence.
# To prepare our train, validation, and test sets with stratification, we need to drop these terms.

# Filtering the rare terms.
data = data.groupby("categories").filter(lambda x: len(x) > 1)

for idx, row in tqdm(data.iterrows()):
    categories = row["categories"].split(",")
    category_list = list(set(categories))
    categories = ",".join(cl for cl in category_list)

    # To make label format compatible with MultiLabelBinarizer, label list is added to empty list
    # final label shape is like : [[a,b,c]]
    data["categories"][idx] = categories

data.to_csv("../data/categories.csv")
