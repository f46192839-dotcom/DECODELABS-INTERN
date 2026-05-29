# DECODELABS DATA SCIENCE INTERNSHIP
# TASK 5: Predictive Model — Iris Species Classifier
# Model: K-Nearest Neighbors (KNN) Classifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

print("=" * 60)
print("   TASK 5: PREDICTIVE MODEL — IRIS SPECIES CLASSIFIER")
print("=" * 60)

# STEP 1: Load & Prepare Dataset
print("\n📦 Step 1: Loading Dataset...")
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

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = iris_raw.target  # numeric labels: 0, 1, 2
label_names = list(species_map.values())

print(f"   Feature matrix shape : {X.shape}")
print(f"   Target vector shape  : {y.shape}")
print(f"   Classes              : {label_names}")

# STEP 2: Train-Test Split (80% train / 20% test)
print("\n✂️  Step 2: Splitting into Train & Test Sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")

# STEP 3: Feature Scaling (StandardScaler)
print("\n⚖️  Step 3: Scaling Features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # use train's mean/std only
print("   Features scaled to zero mean and unit variance.")

# STEP 4: Find Best K using Cross-Validation
print("\n🔍 Step 4: Finding Optimal K (1 to 15)...")
k_range = range(1, 16)
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_range.start + cv_scores.index(max(cv_scores))
print(f"   Best K found: {best_k}  (CV Accuracy: {max(cv_scores)*100:.2f}%)")

# Plot K vs Accuracy
plt.figure(figsize=(9, 5))
plt.plot(list(k_range), cv_scores, marker='o', color='#2196F3', linewidth=2, markersize=7)
plt.axvline(x=best_k, color='#FF5722', linestyle='--', linewidth=1.5, label=f'Best K = {best_k}')
plt.title('Cross-Validation Accuracy vs K (Number of Neighbors)', fontsize=13, fontweight='bold')
plt.xlabel('K Value', fontsize=11)
plt.ylabel('CV Accuracy', fontsize=11)
plt.xticks(list(k_range))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('chart7_knn_k_selection.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Chart saved: chart7_knn_k_selection.png")

# STEP 5: Train Final Model with Best K
print(f"\n🏋️  Step 5: Training Final KNN Model (K={best_k})...")
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)
print("   Model trained successfully.")

# STEP 6: Make Predictions & Evaluate
print("\n📈 Step 6: Evaluating Model on Test Set...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   🎯 Test Accuracy : {accuracy * 100:.2f}%")
print(f"   Correct   : {sum(y_pred == y_test)} / {len(y_test)}")
print(f"   Incorrect : {sum(y_pred != y_test)} / {len(y_test)}")

print("\n📋 Classification Report:")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=label_names))

# STEP 7: Confusion Matrix Plot
print("📊 Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — KNN Classifier', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('chart8_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Chart saved: chart8_confusion_matrix.png")

# STEP 8: Cross-Validation on Full Training Set
print("\n🔁 Step 7: 5-Fold Cross-Validation Summary:")
final_cv = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"   CV Scores  : {[round(s, 4) for s in final_cv]}")
print(f"   Mean       : {final_cv.mean()*100:.2f}%")
print(f"   Std Dev    : ± {final_cv.std()*100:.2f}%")

# STEP 9: Feature Importance via Correlation with Target
print("\n📌 Step 8: Feature Importance (Correlation with Target):")
print("-" * 60)
df_corr = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df_corr['target'] = y
correlations = df_corr.corr()['target'].drop('target').abs().sort_values(ascending=False)
for feat, corr_val in correlations.items():
    bar = '█' * int(corr_val * 20)
    print(f"   {feat:<18}: {corr_val:.4f}  {bar}")

# STEP 10: Predict on a New Unseen Sample
print("\n🌸 Step 9: Predicting a New Unseen Flower Sample:")
print("-" * 60)
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # likely Setosa
new_scaled = scaler.transform(new_sample)
prediction = model.predict(new_scaled)
probabilities = model.predict_proba(new_scaled)[0]

print(f"   Input  : sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
print(f"   Predicted Species : ✅ {species_map[prediction[0]].upper()}")
print(f"   Prediction Confidence:")
for sp, prob in zip(label_names, probabilities):
    bar = '█' * int(prob * 20)
    print(f"     {sp:<12}: {prob*100:.1f}%  {bar}")

print("\n" + "=" * 60)
print("  📝 MODEL SUMMARY")
print("=" * 60)
summary = f"""
   Algorithm        : K-Nearest Neighbors (KNN)
   Best K           : {best_k}
   Feature Scaling  : StandardScaler (z-score normalization)
   Train/Test Split : 80% / 20%
   Test Accuracy    : {accuracy*100:.2f}%
   CV Mean Accuracy : {final_cv.mean()*100:.2f}% ± {final_cv.std()*100:.2f}%

   Most Important Features (by correlation):
     1. petal_length  (highest predictive power)
     2. petal_width
     3. sepal_length
     4. sepal_width   (least predictive)

   Conclusion:
     The KNN model performs extremely well on the Iris dataset.
     Petal features are the strongest predictors of species.
     Setosa is the easiest class to classify (0 errors expected).
"""
print(summary)
print("=" * 60)
print("  ✅ Task 5 Complete: Predictive Model Built & Evaluated")
print("=" * 60)