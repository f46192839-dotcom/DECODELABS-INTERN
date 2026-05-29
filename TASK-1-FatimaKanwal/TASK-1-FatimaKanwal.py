# DECODELABS DATA SCIENCE INTERNSHIP
# TASK 1: Data Collection & Dataset Understanding
# Dataset: Iris Flower Dataset (loaded via sklearn)

import pandas as pd
from sklearn.datasets import load_iris

# STEP 1: Load the Dataset
iris_raw = load_iris()

df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)

df['species_id'] = iris_raw.target

species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species_id'].map(species_map)

df.drop(columns=['species_id'], inplace=True)

print("=" * 55)
print("       TASK 1: DATA COLLECTION & UNDERSTANDING")
print("=" * 55)

# STEP 2: Preview the Dataset
print("\n📋 First 5 Rows of the Dataset:")
print("-" * 55)
print(df.head())

print("\n📋 Last 5 Rows of the Dataset:")
print("-" * 55)
print(df.tail())

# STEP 3: Dataset Size & Shape
print("\n📐 Dataset Shape (rows x columns):")
print(f"   Rows    : {df.shape[0]}")
print(f"   Columns : {df.shape[1]}")

# STEP 4: Column Names & Data Types
print("\n🏷️  Column Names and Data Types:")
print("-" * 55)
print(df.dtypes)

# STEP 5: Check for Missing Values
print("\n🔍 Missing Values per Column:")
print("-" * 55)
missing = df.isnull().sum()
print(missing)
print(f"\n   Total missing values: {missing.sum()}")

# STEP 6: Unique Value Counts
print("\n🌸 Unique Species in Dataset:")
print("-" * 55)
print(df['species'].value_counts())

# STEP 7: Basic Statistical Summary
print("\n📊 Statistical Summary (Numeric Columns):")
print("-" * 55)
print(df.describe().round(2))

# STEP 8: Dataset Description
print("\n📝 What This Dataset Represents:")
print("-" * 55)
description = """
The Iris dataset is one of the most well-known datasets in
data science, introduced by statistician Ronald Fisher in 1936.

It contains 150 records of iris flowers from 3 species:
  • Iris Setosa
  • Iris Versicolor
  • Iris Virginica

Each record has 4 numeric features (measurements in cm):
  1. Sepal Length  - length of the outer petal-like leaf
  2. Sepal Width   - width of the outer petal-like leaf
  3. Petal Length  - length of the inner colored petal
  4. Petal Width   - width of the inner colored petal

Use Case: Commonly used for classification tasks to predict
which species a flower belongs to based on measurements.
"""
print(description)

print("=" * 55)
print("  ✅ Task 1 Complete: Dataset Loaded & Understood")
print("=" * 55)