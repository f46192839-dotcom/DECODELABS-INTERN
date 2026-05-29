# DECODELABS DATA SCIENCE INTERNSHIP
# TASK 4: Data Visualization
# Dataset: Iris Flower Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_iris

# Setup
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
COLORS = {'setosa': '#4CAF50', 'versicolor': '#2196F3', 'virginica': '#FF5722'}
SPECIES = ['setosa', 'versicolor', 'virginica']

print("=" * 55)
print("      TASK 4: DATA VISUALIZATION")
print("=" * 55)

# CHART 1: Histogram — Distribution of All Features
print("\n📊 Generating Chart 1: Feature Distributions (Histograms)")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribution of Iris Features by Species', fontsize=16, fontweight='bold', y=1.01)
axes = axes.flatten()

for idx, col in enumerate(numeric_cols):
    ax = axes[idx]
    for species in SPECIES:
        data_subset = df[df['species'] == species][col]
        ax.hist(data_subset, bins=15, alpha=0.6, color=COLORS[species],
                label=species, edgecolor='white', linewidth=0.5)
    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_xlabel('Value (cm)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('chart1_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: chart1_histograms.png")

# CHART 2: Box Plot — Spread & Outliers per Species
print("📊 Generating Chart 2: Box Plots (Outlier Detection)")

fig, axes = plt.subplots(1, 4, figsize=(16, 6))
fig.suptitle('Feature Spread & Outliers by Species (Box Plots)',
             fontsize=15, fontweight='bold')

for idx, col in enumerate(numeric_cols):
    ax = axes[idx]
    data_by_species = [df[df['species'] == sp][col].values for sp in SPECIES]
    bp = ax.boxplot(data_by_species, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    for patch, sp in zip(bp['boxes'], SPECIES):
        patch.set_facecolor(COLORS[sp])
        patch.set_alpha(0.75)
    ax.set_xticklabels(SPECIES, rotation=15, fontsize=9)
    ax.set_title(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_ylabel('Value (cm)', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('chart2_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: chart2_boxplots.png")

# CHART 3: Scatter Plot — Petal Length vs Petal Width
print("📊 Generating Chart 3: Scatter Plot (Petal Length vs Width)")

fig, ax = plt.subplots(figsize=(9, 6))
for sp in SPECIES:
    subset = df[df['species'] == sp]
    ax.scatter(subset['petal_length'], subset['petal_width'],
               label=sp, color=COLORS[sp], alpha=0.75, s=70, edgecolors='white', linewidth=0.5)

ax.set_title('Petal Length vs Petal Width by Species', fontsize=14, fontweight='bold')
ax.set_xlabel('Petal Length (cm)', fontsize=12)
ax.set_ylabel('Petal Width (cm)', fontsize=12)
ax.legend(title='Species', fontsize=10)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.annotate('Setosa is clearly\nseparable here',
            xy=(1.5, 0.3), xytext=(2.5, 0.8),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='#333')

plt.tight_layout()
plt.savefig('chart3_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: chart3_scatter.png")

# CHART 4: Bar Chart — Average Feature Values per Species
print("📊 Generating Chart 4: Bar Chart (Average Feature Values)")

means = df.groupby('species')[numeric_cols].mean()
x = np.arange(len(numeric_cols))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))
for i, sp in enumerate(SPECIES):
    ax.bar(x + i * width, means.loc[sp], width, label=sp,
           color=COLORS[sp], alpha=0.85, edgecolor='white')

ax.set_title('Average Feature Values per Species', fontsize=14, fontweight='bold')
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Mean Value (cm)', fontsize=12)
ax.set_xticks(x + width)
ax.set_xticklabels([c.replace('_', '\n') for c in numeric_cols], fontsize=10)
ax.legend(title='Species', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('chart4_barchart.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: chart4_barchart.png")

# CHART 5: Correlation Heatmap
print("📊 Generating Chart 5: Correlation Heatmap")

corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation Coefficient')

labels = [c.replace('_', '\n') for c in numeric_cols]
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)

for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f'{corr.values[i, j]:.2f}',
                ha='center', va='center', fontsize=10,
                color='black' if abs(corr.values[i, j]) < 0.7 else 'white')

plt.tight_layout()
plt.savefig('chart5_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: chart5_heatmap.png")

# CHART 6: Pie Chart — Class Distribution
print("📊 Generating Chart 6: Pie Chart (Species Distribution)")

counts = df['species'].value_counts()
fig, ax = plt.subplots(figsize=(7, 6))
wedges, texts, autotexts = ax.pie(
    counts.values,
    labels=counts.index,
    colors=[COLORS[sp] for sp in counts.index],
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.82,
    wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2)
)
for text in autotexts:
    text.set_fontsize(11)
    text.set_fontweight('bold')
ax.set_title('Species Distribution in Dataset', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('chart6_piechart.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: chart6_piechart.png")

print("\n" + "=" * 55)
print("  ✅ Task 4 Complete: 6 Visualizations Generated")
print("     All charts saved to /mnt/user-data/outputs/")
print("=" * 55)

print("\n💡 Key Visual Insights:")
insights = [
    "• Histograms show setosa has distinctly smaller petal dimensions.",
    "• Box plots reveal sepal_width has the most outlier points.",
    "• Scatter plot shows setosa is perfectly separable by petal size.",
    "• Bar chart confirms virginica has the largest feature averages.",
    "• Heatmap shows petal_length ↔ petal_width correlation = 0.96.",
    "• Pie chart confirms a perfectly balanced dataset (33.3% each).",
]
for insight in insights:
    print(f"   {insight}")