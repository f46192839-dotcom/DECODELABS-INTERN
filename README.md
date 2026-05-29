# 🌸 Iris Data Science Project
### Decodelabs Data Science Internship

A end-to-end data science project covering the full pipeline — from data collection to a working predictive model — built on the classic **Iris Flower Dataset**.

---

## 📁 Project Structure

```
├── task1_data_collection.py     # Load & understand the dataset
├── task2_data_cleaning.py       # Clean & preprocess data
├── task3_eda.py                 # Exploratory Data Analysis
├── task4_visualization.py       # Charts & visual insights
├── task5_predictive_model.py    # KNN classifier & evaluation
├── run_all_tasks.py             # Run all 5 tasks at once
└── README.md
```

---

## 🗂️ Tasks Overview

| # | Task | Key Skills |
|---|------|------------|
| 1 | Data Collection & Understanding | Data types, shape, structure |
| 2 | Data Cleaning & Preprocessing | Missing values, duplicates, formatting |
| 3 | Exploratory Data Analysis | Statistics, outliers, correlations |
| 4 | Data Visualization | Histograms, scatter, heatmap, box plots |
| 5 | Predictive Model | KNN classifier, accuracy, confusion matrix |

---

## 📊 Dataset

- **Name:** Iris Flower Dataset
- **Source:** `sklearn.datasets.load_iris`
- **Records:** 150 rows × 5 columns
- **Features:** Sepal length, Sepal width, Petal length, Petal width
- **Target:** Species — *Setosa*, *Versicolor*, *Virginica*

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/f46192839-dotcom/iris-data-science-project.git
cd iris-data-science-project
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib scikit-learn
```

**3. Run all tasks**
```bash
python run_all_tasks.py
```

Or run individually:
```bash
python task1_data_collection.py
python task2_data_cleaning.py
python task3_eda.py
python task4_visualization.py
python task5_predictive_model.py
```

---

## 📈 Output Files Generated

| File | Description |
|------|-------------|
| `iris_cleaned.csv` | Cleaned dataset from Task 2 |
| `chart1_histograms.png` | Feature distributions by species |
| `chart2_boxplots.png` | Spread & outliers per feature |
| `chart3_scatter.png` | Petal length vs petal width |
| `chart4_barchart.png` | Average feature values per species |
| `chart5_heatmap.png` | Feature correlation heatmap |
| `chart6_piechart.png` | Species class distribution |
| `chart7_knn_k_selection.png` | Cross-validation accuracy vs K |
| `chart8_confusion_matrix.png` | Model prediction results |

---

## 🤖 Model Results (Task 5)

- **Algorithm:** K-Nearest Neighbors (KNN)
- **Best K:** Selected via 5-fold cross-validation
- **Test Accuracy:** ~97–100%
- **Top Feature:** `petal_length` (highest correlation with species)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-grey?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-grey?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-grey?logo=matplotlib)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-grey?logo=scikit-learn)

---

## 🔗 Run Online (No Install Needed)

Open directly in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> Paste any task file into a new notebook cell and run.

---

## 📌 Notes

- Complete any **3 out of 5** tasks as per internship requirements
- No copied or plagiarized code — all scripts are original
- Submission details shared via the Decodelabs WhatsApp group

---

## 🏢 About Decodelabs

**Website:** [decodelabs.tech](https://www.decodelabs.tech) &nbsp;|&nbsp;
**LinkedIn:** [decodelabs](https://linkedin.com/company/decodelabs) &nbsp;|&nbsp;
**Instagram:** [@official_decodelabs](https://instagram.com/official_decodelabs) &nbsp;|&nbsp;
**Telegram:** [t.me/decodelabs_tech](https://t.me/decodelabs_tech)
