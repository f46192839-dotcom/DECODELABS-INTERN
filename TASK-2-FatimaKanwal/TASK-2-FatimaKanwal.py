# DECODELABS DATA SCIENCE INTERNSHIP
# TASK 2: Data Cleaning & Preprocessing
# Dataset: Iris (with intentionally injected dirty data)

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# STEP 1: Load Clean Dataset First
iris_raw = load_iris()
df_original = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df_original['species'] = pd.Categorical(
    [species_map[i] for i in iris_raw.target]
)

print("=" * 58)
print("      TASK 2: DATA CLEANING & PREPROCESSING")
print("=" * 58)

# STEP 2: Inject Dirty Data to Simulate Real-World Problems
print("\n⚠️  Injecting dirty data for demonstration purposes...")

df_dirty = df_original.copy()
np.random.seed(42)

# 2a. Introduce ~10% missing values randomly
num_missing = int(0.10 * df_dirty.shape[0])
for col in ['sepal length (cm)', 'petal width (cm)']:
    missing_idx = np.random.choice(df_dirty.index, size=num_missing, replace=False)
    df_dirty.loc[missing_idx, col] = np.nan

# 2b. Add 10 duplicate rows
duplicates = df_dirty.sample(10, random_state=7)
df_dirty = pd.concat([df_dirty, duplicates], ignore_index=True)

# 2c. Introduce incorrect data types (species stored as integer codes)
df_dirty['species'] = df_dirty['species'].map(
    {'setosa': 1, 'versicolor': 2, 'virginica': 3}
)

# 2d. Add some inconsistent whitespace in a string column (simulate)
df_dirty['extra_notes'] = np.random.choice(
    ['  healthy  ', 'wilting ', ' fresh', 'dried  ', '  unknown '],
    size=len(df_dirty)
)

print("   Dirty dataset created.\n")

# STEP 3: Inspect the Dirty Dataset
print("🔍 Dirty Dataset Overview:")
print(f"   Shape           : {df_dirty.shape}")
print(f"   Duplicate Rows  : {df_dirty.duplicated().sum()}")
print(f"\n   Missing Values per Column:")
print(df_dirty.isnull().sum().to_string(index=True))

# STEP 4: Start Cleaning — Work on a copy
df_clean = df_dirty.copy()

# 4a. Remove Duplicate Rows
before_dedup = len(df_clean)
df_clean.drop_duplicates(inplace=True)
df_clean.reset_index(drop=True, inplace=True)
after_dedup = len(df_clean)
print(f"\n✅ [STEP 1] Duplicates Removed: {before_dedup - after_dedup} rows dropped")
print(f"   Rows before: {before_dedup}  |  Rows after: {after_dedup}")

# 4b. Handle Missing Values
# Strategy: fill numeric missing values with column median
#           (median is robust to outliers, unlike mean)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n✅ [STEP 2] Handling Missing Values (strategy: median fill)")
for col in numeric_cols:
    missing_count = df_clean[col].isnull().sum()
    if missing_count > 0:
        median_val = df_clean[col].median()
        df_clean.fillna({col: median_val}, inplace=True)
        print(f"   '{col}': {missing_count} nulls filled with median = {median_val:.3f}")

# Confirm no missing values remain
remaining_nulls = df_clean.isnull().sum().sum()
print(f"   Remaining missing values: {remaining_nulls}")

# 4c. Fix Data Types
print("\n✅ [STEP 3] Fixing Data Types")
species_reverse_map = {1: 'setosa', 2: 'versicolor', 3: 'virginica'}
df_clean['species'] = df_clean['species'].map(species_reverse_map)
df_clean['species'] = df_clean['species'].astype('category')
print("   'species' column converted from int → categorical string")

# 4d. Strip Whitespace from String Columns
print("\n✅ [STEP 4] Stripping Whitespace from Text Columns")
df_clean['extra_notes'] = df_clean['extra_notes'].str.strip()
sample_after = df_clean['extra_notes'].unique()
print(f"   Cleaned 'extra_notes' values: {list(sample_after)}")

# 4e. Drop Columns That Aren't Needed
print("\n✅ [STEP 5] Dropping Unnecessary Columns")
df_clean.drop(columns=['extra_notes'], inplace=True)
print("   Dropped: 'extra_notes' (not relevant for analysis)")

# 4f. Rename Columns for Cleaner Access
print("\n✅ [STEP 6] Renaming Columns for Clarity")
rename_map = {
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)' : 'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)' : 'petal_width'
}
df_clean.rename(columns=rename_map, inplace=True)
print(f"   New column names: {list(df_clean.columns)}")

# 4g. Reset Index
df_clean.reset_index(drop=True, inplace=True)

# STEP 5: Validate the Cleaned Dataset
print("\n" + "=" * 58)
print("  📋 CLEANED DATASET SUMMARY")
print("=" * 58)
print(f"   Final Shape     : {df_clean.shape}")
print(f"   Missing Values  : {df_clean.isnull().sum().sum()}")
print(f"   Duplicates      : {df_clean.duplicated().sum()}")
print(f"\n   Data Types:")
print(df_clean.dtypes.to_string())
print(f"\n   First 5 Rows:")
print(df_clean.head())

# STEP 6: Save Cleaned Dataset to CSV
output_path = 'iris_cleaned.csv'
df_clean.to_csv(output_path, index=False)
print(f"\n💾 Cleaned dataset saved to: {output_path}")

print("\n" + "=" * 58)
print("  ✅ Task 2 Complete: Data Cleaned & Ready for Analysis")
print("=" * 58)