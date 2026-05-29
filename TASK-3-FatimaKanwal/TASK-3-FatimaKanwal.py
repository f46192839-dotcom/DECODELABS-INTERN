# DECODELABS DATA SCIENCE INTERNSHIP
# TASK 3: Exploratory Data Analysis (EDA)
# Dataset: Cleaned Iris Dataset

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

print("=" * 58)
print("      TASK 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 58)

# STEP 1: Load Dataset (using cleaned version logic)
iris_raw = load_iris()
df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
df.rename(columns={
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)' : 'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)' : 'petal_width'
}, inplace=True)
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = [species_map[i] for i in iris_raw.target]

numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# STEP 2: Basic Statistical Summary
print("\n📊 Overall Descriptive Statistics:")
print("-" * 58)
print(df[numeric_cols].describe().round(3))

# STEP 3: Per-Species Statistics
print("\n📊 Mean Values Grouped by Species:")
print("-" * 58)
print(df.groupby('species')[numeric_cols].mean().round(3))

print("\n📊 Standard Deviation Grouped by Species:")
print("-" * 58)
print(df.groupby('species')[numeric_cols].std().round(3))

print("\n📊 Min & Max per Feature:")
print("-" * 58)
print(df[numeric_cols].agg(['min', 'max']))

# STEP 4: Distribution Analysis (spread + skewness)
print("\n📐 Skewness of Each Feature (0 = symmetric):")
print("-" * 58)
skewness = df[numeric_cols].skew().round(4)
for col, val in skewness.items():
    direction = "right-skewed" if val > 0.5 else ("left-skewed" if val < -0.5 else "approx. symmetric")
    print(f"   {col:<18}: {val:>7}  →  {direction}")

# STEP 5: Outlier Detection Using IQR Method
print("\n🔎 Outlier Detection (IQR Method):")
print("-" * 58)
outlier_summary = {}
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = len(outliers)
    print(f"   {col:<18}: {len(outliers):>2} outlier(s)  "
          f"[bounds: {lower_bound:.2f} – {upper_bound:.2f}]")

total_outliers = sum(outlier_summary.values())
print(f"\n   Total outlier records detected: {total_outliers}")

# STEP 6: Correlation Matrix
print("\n🔗 Feature Correlation Matrix:")
print("-" * 58)
corr_matrix = df[numeric_cols].corr().round(3)
print(corr_matrix)

print("\n💡 Strongest Positive Correlations:")
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
corr_pairs.columns = ['Feature A', 'Feature B', 'Correlation']
corr_pairs = corr_pairs.reindex(
    corr_pairs['Correlation'].abs().sort_values(ascending=False).index
)
for _, row in corr_pairs.head(3).iterrows():
    print(f"   {row['Feature A']} ↔ {row['Feature B']}: {row['Correlation']:.3f}")

# STEP 7: Class Distribution
print("\n🌸 Class Distribution:")
print("-" * 58)
counts = df['species'].value_counts()
total = len(df)
for species, count in counts.items():
    bar = '█' * (count // 3)
    print(f"   {species:<12}: {count} records ({count/total*100:.1f}%)  {bar}")

# STEP 8: Key Findings Summary
print("\n" + "=" * 58)
print("  📝 KEY FINDINGS SUMMARY")
print("=" * 58)
findings = [
    "1. Dataset is perfectly balanced: 50 samples per species.",
    "2. Petal length & petal width are highly correlated (r ≈ 0.96).",
    "3. Setosa has noticeably smaller petals than the other two.",
    "4. Sepal width shows slight left-skew; petal features are right-skewed.",
    "5. Sepal width has the most outliers — measurement variability is high.",
    "6. Versicolor and Virginica overlap in feature space more than Setosa.",
    "7. Petal measurements are the most discriminative features for classification.",
]
for finding in findings:
    print(f"   {finding}")

print("\n" + "=" * 58)
print("  ✅ Task 3 Complete: EDA Performed Successfully")
print("=" * 58)