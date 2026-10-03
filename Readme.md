# 🚚 Late Delivery Risk & Operational Segmentation

![Python](https://img.shields.io/badge/Python-Data%20Science-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FKeras-red)
![Status](https://img.shields.io/badge/Project-Completed-success)

> **Data Science & AI/ML Practical Exam — Set C**  
> Predict late deliveries and identify operational segments using Statistics, Machine Learning, Clustering and ANN.

---

## 👩‍💻 Student Information

| Details          | Information                    |
| ---------------- | ------------------------------ |
| **Student Name** | Janki Dholariya                |
| **Exam Set**     | Set C                          |
| **Project Type** | Data Science & AI/ML Practical |
| **Language**     | Python                         |
| **Notebook**     | `EXAMS.ipynb`                   |

---

## 📌 Project Overview

This project analyzes a **synthetic delivery-risk dataset** to:

- Understand the data using descriptive statistics.
- Perform statistical inference using a Welch two-sample t-test.
- Calculate covariance, eigenvalues and variance share.
- Clean and preprocess the dataset without data leakage.
- Predict whether a delivery will be late using Logistic Regression.
- Create operational segments using K-Means clustering.
- Build a small Artificial Neural Network (ANN).
- Compare Logistic Regression and ANN on the same untouched test set.

The project follows the required **fit / validation / test** workflow with fixed random seed `42`.

---

## 📊 Dataset

The supplied generator creates **305 rows and 7 columns**, including **5 exact duplicate rows**. After removing duplicates, there are **300 unique records**.

### Dataset Link

-  [set_b.csv](data/raw/set_b.csv)

### Dataset Columns

| Column | Description |
|---|---|
| `record_id` | Unique record identifier; excluded from models |
| `distance` | Synthetic distance index |
| `load` | Synthetic load index |
| `traffic` | Synthetic traffic index |
| `staff` | Synthetic staff index |
| `group` | Operational cohort: G1 or G2 |
| `late` | Target: `1 = late`, `0 = not late` |

### Engineered Feature

```text
engineered_feature = load / (staff + 1)
```

The feature is created **after numeric median imputation and before scaling**.

---

## 🧹 Data Preparation

### Raw Dataset Audit

- Raw shape: **305 × 7**
- Exact duplicates: **5**
- Clean records: **300**
- Missing `distance`: **15**
- Missing `load`: **15**
- Target `late = 0`: **156**
- Target `late = 1`: **149**

### Data Split

| Partition | Records |
|---|---:|
| Fit / Train | 192 |
| Validation | 48 |
| Test | 60 |

- Stratified split
- `random_state = 42`
- Fit, validation and test IDs are disjoint.
- Test data is kept untouched until final evaluation.

---

## 📈 1. Maths & Advanced Statistics

### Descriptive Statistics — Distance

| Statistic | Result |
|---|---:|
| Observed n | 184 |
| Mean | 49.7444 |
| Median | 50.20 |
| Sample Standard Deviation | 10.8359 |

### Welch's t-test

**H₀:** Mean distance of G1 and G2 is equal.  
**H₁:** Mean distance of G1 and G2 is different.

| Result | Value |
|---|---:|
| G1 count | 104 |
| G2 count | 80 |
| t-statistic | 0.7062 |
| p-value | 0.4811 |
| 95% CI for overall mean | [48.1683, 51.3205] |

The test is interpreted as a comparison of group means and does not establish causation.

### Covariance & Eigenvalue

| Measure | Result |
|---|---:|
| Largest eigenvalue | 117.5670 |
| Variance share | 52.2869% |
| Principal direction | [-0.9926, 0.1211] |

---

## ⚙️ 2. Data Preprocessing & Feature Engineering

### Preprocessing Steps

1. Remove exact duplicate rows.
2. Create stratified fit/validation/test partitions.
3. Apply median imputation to numeric columns using fit data only.
4. Create `engineered_feature`.
5. Apply `StandardScaler` using fit data only.
6. One-hot encode `group`.
7. Keep `group` one-hot columns unscaled.
8. Exclude `record_id` and `late` from model features.
9. Verify all transformed values are finite.

### Leakage Prevention

The imputer, encoder and scaler are fitted only on the **192 fit records**.

The **60 test records remain untouched** until final model evaluation.

---

## 🤖 3. Supervised Learning

### Baseline

A `DummyClassifier(strategy="most_frequent")` was used as the baseline.

### Logistic Regression

Settings:

```text
Model: LogisticRegression
max_iter: 1000
random_state: 42
Threshold: 0.5
```

### Test Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Dummy Classifier | 0.5167 | — | — | — |
| Logistic Regression | **0.8500** | **0.8846** | **0.7931** | **0.8364** |

### Confusion Matrix Findings

- False Positives: **3**
- False Negatives: **6**

A false positive means an on-time delivery is predicted as late.  
A false negative means a late delivery is predicted as on-time.

---

## 🔵 4. Unsupervised Learning — K-Means

K-Means was tested with:

```text
k = 2, 3, 4
n_init = 10
random_state = 42
```

### K Selection

| k | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 719.6321 | **0.2463** |
| 3 | 599.6594 | 0.2071 |
| 4 | 535.3835 | 0.1902 |

The highest silhouette score was obtained with **k = 2**.

### Cluster Profiles

| Cluster | Distance | Load | Traffic | Staff | Engineered Feature |
|---:|---:|---:|---:|---:|---:|
| 0 | 48.4190 | 47.1280 | 49.0663 | 54.0350 | 0.8633 |
| 1 | 52.8695 | 57.4616 | 49.8562 | 41.4267 | 1.3730 |

Cluster IDs are labels only and do not represent target classes.

---

## 🧠 5. Artificial Neural Network

### ANN Architecture

```text
Input
  ↓
Dense(16, ReLU)
  ↓
Dense(8, ReLU)
  ↓
Dense(1, Sigmoid)
```

### Training Settings

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss | Binary Cross-Entropy |
| Batch Size | 16 |
| Maximum Epochs | 50 |
| Actual Epochs | 36 |
| Early Stopping Patience | 5 |
| Random Seed | 42 |
| Trainable Parameters | 273 |

### ANN Test Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8500 | 0.8846 | 0.7931 | **0.8364** |
| ANN | 0.8000 | 0.8400 | 0.7241 | **0.7778** |

Both models were evaluated on the **same 60 untouched test records**.

---

## 🏁 Key Findings

1. The raw dataset contained **305 rows**, and removing 5 exact duplicates produced **300 unique records**.
2. Logistic Regression achieved **0.85 test accuracy and 0.8364 F1**.
3. K-Means selected **k = 2** because it produced the highest silhouette score (**0.2463**).
4. The ANN used **273 trainable parameters** and stopped after **36 epochs** using early stopping.
5. On the same test set, Logistic Regression had F1 **0.8364**, while ANN had F1 **0.7778**.

### Practical Recommendation

For this synthetic practice dataset, the final comparison shows that Logistic Regression provides a simpler baseline with stronger held-out F1 than the ANN. Operational actions should still be validated with larger and real-world delivery data before deployment.

### Limitation

The dataset is synthetic and the evaluation uses one small holdout test set. Therefore, the results should not be treated as deployment-ready evidence.

---

## 🖼️ Screenshots / Project Evidence

Upload your screenshots inside the `screenshots/` folder using the filenames below. The links are already prepared in this README.

### Histogram

- ![histogram](OUTPUT\figures\histogram.png)

### line_chart

- ![line_chart](OUTPUT\figures\line_chart.png)

### confusen_matrix

- ![confusen_matrix](OUTPUT\figures\confusen_matrix.png)
---

## 🎥 Project Explanation Video

🎬 **[Watch Project Explanation Video](YOUR_VIDEO_URL)**

**Duration:** `YOUR VIDEO DURATION`

---

## 📁 Project Structure

```text
project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── generate_data.py
│
├── data/
│   └── raw/
│       └── set_b.csv
│
├── notebooks/
│   └── exam.ipynb
│
├── outputs/
│   ├── splits.csv
│   ├── statistics/
│   ├── predictions/
│   └── figures/
│
├── models/
│   └── delivery_risk_ann.keras
│
├── screenshots/
│   ├── 01_dataset_preview.png
│   ├── 02_statistics.png
│   ├── 03_logistic_confusion_matrix.png
│   ├── 04_kmeans_selection.png
│   ├── 05_ann_loss_curve.png
│   └── 06_model_comparison.png
│
├── logistic_predictions.csv
├── ann_predictions.csv
├── k_selection_scores.csv
├── cluster_profiles.csv
└── ann_logistic_comparison.csv
```

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook
- GitHub

---

## 📚 References

- Supplied Data Science & AI/ML Practical Exam — Set C
- Supplied dataset generator
- Scikit-learn documentation
- TensorFlow/Keras documentation
- SciPy documentation

---

### ⭐ Project Summary

**Problem → Data Generation → Cleaning → Statistics → Preprocessing → Logistic Regression → K-Means → ANN → Model Comparison → Interpretation**


# 👩‍💻 Author

**Janki Dholariya**

Data Science & AI/ML Practical Project – Set C
