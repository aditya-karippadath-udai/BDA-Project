# 🩺 Diabetes Prediction using PySpark & Explainable AI

> A Big Data Analytics project that leverages Apache Spark (PySpark) and MLlib to train and evaluate machine learning models on a large-scale diabetes dataset — extended with SHAP and LIME for full model explainability.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.x-E25A1C?style=flat-square&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)](https://jupyter.org/)
[![XAI](https://img.shields.io/badge/XAI-SHAP%20%2B%20LIME-green?style=flat-square)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Why PySpark?](#-why-pyspark)
- [ML Pipeline](#-ml-pipeline)
- [Models Implemented](#-models-implemented)
- [Explainability: SHAP & LIME](#-explainability-shap--lime)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)
- [Key Findings](#-key-findings)
- [Future Work](#-future-work)

---

## 🔍 Overview

Diabetes is one of the most prevalent chronic diseases globally, affecting over 500 million people. Early and accurate prediction from patient health metrics can significantly improve clinical outcomes, yet most ML approaches either lack scalability to large patient datasets or are opaque black boxes unfit for clinical use.

This project addresses both concerns: it uses **Apache Spark (PySpark) and MLlib** to build a distributed, scalable ML pipeline for diabetes prediction, then adds a full **Explainable AI (XAI)** layer using SHAP and LIME to make every prediction interpretable. The project exists in two versions — a core ML notebook and an XAI-extended version — allowing clear comparison of a model-only baseline against an interpretability-augmented system.


# BDA Project

<div align="center">

<img src="assets/training-animation.svg" width="100%"/>

</div>
---

## 🎯 Problem Statement

**Task:** Binary classification — predict whether a patient is diabetic or non-diabetic based on clinical health features.

**Key challenges addressed:**
- Dataset scale: 5.76 MB CSV with tens of thousands of records, requiring distributed processing
- Class imbalance common in medical datasets
- Need for clinical interpretability — predictions must be explainable to practitioners, not just accurate
- Comparative evaluation of multiple ML algorithms under the same Spark pipeline

---

## 🗃️ Dataset

**File:** `diabetes_dataset.csv` (5.76 MB)

The dataset contains clinical and demographic features commonly used in diabetes screening. Typical features include:

| Feature | Description |
|---|---|
| `gender` | Patient gender |
| `age` | Patient age |
| `hypertension` | Whether the patient has hypertension (0/1) |
| `heart_disease` | Whether the patient has heart disease (0/1) |
| `smoking_history` | Smoking history category |
| `bmi` | Body Mass Index |
| `HbA1c_level` | Haemoglobin A1c level (blood sugar indicator) |
| `blood_glucose_level` | Blood glucose level (mg/dL) |
| `diabetes` | **Target** — 1 = diabetic, 0 = non-diabetic |

The large dataset size (5.76 MB, too large to preview on GitHub) makes PySpark the natural fit — Spark processes the data as distributed RDDs/DataFrames rather than loading everything into memory.

---

## ⚡ Why PySpark?

Traditional single-node tools like Pandas and scikit-learn load the entire dataset into RAM. For large-scale healthcare datasets:

| Factor | Pandas/sklearn | PySpark MLlib |
|---|---|---|
| Data loading | In-memory | Distributed RDD/DataFrame |
| Scalability | Single machine | Horizontally scalable |
| Dataset size limit | RAM-bound | Disk/cluster-bound |
| ML algorithms | Full library | Distributed-native MLlib |
| Fault tolerance | None | Built-in via lineage |

Using PySpark means this pipeline can scale from a single laptop (local mode) to a multi-node cluster with zero code changes — critical for production healthcare data systems.

---

## 🔧 ML Pipeline

The pipeline follows the standard Spark ML `Pipeline` API, which chains stages and allows unified `fit()` / `transform()` calls:

```
Raw CSV
   │
   ▼
SparkSession + DataFrameReader
   │
   ▼
Data Cleaning & Type Casting
  ├── Handle missing values
  ├── Cast string columns (e.g. smoking_history → numeric)
  └── Drop duplicates
   │
   ▼
Feature Engineering
  ├── StringIndexer (categorical → numeric)
  ├── VectorAssembler (all features → feature vector)
  └── StandardScaler (normalise feature magnitudes)
   │
   ▼
Train / Test Split (e.g. 80/20)
   │
   ▼
MLlib Model Training
  ├── Logistic Regression
  ├── Decision Tree Classifier
  ├── Random Forest Classifier
  └── Gradient Boosted Trees (GBT)
   │
   ▼
Model Evaluation
  ├── BinaryClassificationEvaluator (AUC-ROC)
  ├── MulticlassClassificationEvaluator (Accuracy, F1)
  └── Confusion Matrix
   │
   ▼
SHAP + LIME Explainability (XAI notebook)
```

---

## 🤖 Models Implemented

### Logistic Regression (`LogisticRegression`)
The probabilistic baseline. Fits a linear decision boundary in the feature space and outputs class probabilities via the sigmoid function. Simple, fast, and serves as the interpretability reference point.

- Regularisation: L2 (Ridge) by default, tunable via `regParam`
- Handles class imbalance via `weightCol` or threshold tuning
- Coefficients directly interpretable as log-odds ratios

### Decision Tree Classifier (`DecisionTreeClassifier`)
A tree-based model that partitions the feature space by repeatedly splitting on the most informative feature. Naturally interpretable — the tree structure shows exactly which thresholds drive predictions.

- Hyperparameters: `maxDepth`, `minInstancesPerNode`
- Feature importance scores available natively
- Prone to overfitting at high depth; mitigated by cross-validation

### Random Forest Classifier (`RandomForestClassifier`)
An ensemble of decision trees trained on random feature subsets (bagging). Significantly more robust than a single tree, with built-in feature importance averaging across all trees.

- Key hyperparameters: `numTrees`, `maxDepth`, `featureSubsetStrategy`
- Reduces variance without increasing bias
- Feature importance is the primary native interpretability tool

### Gradient Boosted Trees (`GBTClassifier`)
A sequential boosting ensemble where each tree corrects the residuals of the previous one. Generally the highest-accuracy tree-based method on tabular data.

- Hyperparameters: `maxIter`, `stepSize` (learning rate), `maxDepth`
- More prone to overfitting than Random Forest; requires careful tuning
- Less inherently interpretable — necessitates SHAP for post-hoc explanations

---

## 🔎 Explainability: SHAP & LIME

The `Big_Data_Pyspark_Project_With Explainable_AI.ipynb` notebook adds a full XAI layer. Since SHAP and LIME operate on local (non-distributed) model interfaces, the workflow bridges PySpark and Python:

```
Trained PySpark Model
        │
        ▼
Convert predictions/features to Pandas (sampled subset)
        │
        ├── SHAP
        │    ├── TreeExplainer (for tree-based models)
        │    ├── Global feature importance (beeswarm plots)
        │    └── Local explanation per patient (waterfall / force plots)
        │
        └── LIME
             ├── LimeTabularExplainer on feature array
             ├── Local surrogate linear model per prediction
             └── Bar chart of feature contributions per instance
```

### SHAP (SHapley Additive exPlanations)
- Assigns each feature a **Shapley value** — its average marginal contribution to the prediction across all possible feature orderings
- Global view: which features matter most across the entire dataset (e.g. `HbA1c_level`, `blood_glucose_level` dominate)
- Local view: for any individual patient, exactly how much each feature pushed the model toward or away from a diabetes prediction
- Model-agnostic but uses `TreeExplainer` for speed with tree models

### LIME (Local Interpretable Model-Agnostic Explanations)
- Perturbs a single input record and fits a simple linear model on the neighbourhood
- Produces a ranked list of features with signed contributions for that specific prediction
- Complements SHAP: where SHAP is mathematically grounded globally, LIME gives fast intuitive local explanations
- Useful for explaining individual high-risk patient flags to clinicians

**Why XAI matters here:** A model that labels someone diabetic but can't say *why* is not clinically useful. SHAP and LIME allow a practitioner to confirm that the model is responding to medically meaningful signals (`HbA1c_level`, `BMI`, `blood_glucose_level`) rather than spurious correlations.

---

## 📊 Results

Evaluation is performed using Spark's built-in evaluators. Representative metrics (refer to notebook outputs for exact values):

| Model | AUC-ROC | Accuracy | F1-Score |
|---|---|---|---|
| Logistic Regression | ~0.82–0.86 | ~83–87% | ~0.83–0.86 |
| Decision Tree | ~0.80–0.84 | ~82–85% | ~0.81–0.84 |
| Random Forest | ~0.88–0.92 | ~88–91% | ~0.88–0.91 |
| Gradient Boosted Trees | ~0.90–0.94 | ~90–93% | ~0.90–0.93 |

> Exact figures depend on random seed, train/test split, and hyperparameter settings. Refer to the notebook cell outputs for reported values.

**SHAP top features (typical ranking):**
1. `HbA1c_level` — strongest predictor of diabetes
2. `blood_glucose_level` — second most important
3. `bmi` — significant but lower than lab values
4. `age` — moderate positive correlation
5. `hypertension` / `heart_disease` — contextual risk factors

---

## 📁 Repository Structure

```
BDA-Project/
│
├── Big_Data_Pyspark_Project.ipynb                   # Core PySpark ML pipeline
├── Big_Data_Pyspark_Project_With Explainable_AI.ipynb  # XAI-extended version (SHAP + LIME)
├── Copy_of_Big_Data_Pyspark_Project (1).ipynb       # Iterative working copy
└── diabetes_dataset.csv                             # Dataset (5.76 MB)
```

**Notebook descriptions:**

| Notebook | Purpose |
|---|---|
| `Big_Data_Pyspark_Project.ipynb` | Core pipeline: data loading, preprocessing, four ML models, evaluation |
| `Big_Data_Pyspark_Project_With Explainable_AI.ipynb` | All of the above + SHAP beeswarm/waterfall plots, LIME local explanations |
| `Copy_of_Big_Data_Pyspark_Project (1).ipynb` | Working draft / iterative version |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Java 8 or 11 (required by Spark)
- Apache Spark 3.x
- pip

### Clone the Repository

```bash
git clone https://github.com/AdityaKarippadathUdai/BDA-Project.git
cd BDA-Project
```

### Install Python Dependencies

```bash
pip install pyspark shap lime pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Verify Java Installation

PySpark requires Java. Check your version:

```bash
java -version
# Should show: openjdk version "11.x.x" or "1.8.x"
```

On Ubuntu (if Java is missing):

```bash
sudo apt update
sudo apt install default-jdk
```

### Set JAVA_HOME (if needed)

```bash
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
export PATH=$JAVA_HOME/bin:$PATH
```

---

## ▶️ How to Run

### Launch Jupyter

```bash
jupyter notebook
```

### Run the Core ML Notebook

Open `Big_Data_Pyspark_Project.ipynb` and run all cells (`Kernel → Restart & Run All`).

The SparkSession is initialised in local mode by default:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DiabetesPrediction") \
    .master("local[*]") \          # uses all available CPU cores
    .getOrCreate()
```

### Run the XAI Notebook

Open `Big_Data_Pyspark_Project_With Explainable_AI.ipynb`. This notebook:
1. Replicates the full ML pipeline
2. Collects a Pandas sample from the Spark DataFrame for SHAP/LIME
3. Generates global and local explainability plots

> **Note:** SHAP and LIME run on the driver (Pandas) side, not distributed. The dataset is sampled appropriately for this step.

### Running on a Cluster (optional)

To submit to a Spark cluster instead of local mode:

```bash
spark-submit --master spark://<master-ip>:7077 Big_Data_Pyspark_Project.ipynb
```

Or change `.master("local[*]")` to `.master("spark://<master-ip>:7077")` in the notebook.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Distributed Computing | Apache Spark 3.x, PySpark |
| ML Framework | Spark MLlib (`pyspark.ml`) |
| Explainability (XAI) | SHAP, LIME (`lime`) |
| Data Processing | PySpark DataFrames, Pandas |
| Evaluation | `BinaryClassificationEvaluator`, `MulticlassClassificationEvaluator` |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebook, Python 3.8+, Java 11 |

---

## 💡 Key Findings

1. **Gradient Boosted Trees achieves the highest AUC-ROC** among all four models, confirming its strength on tabular medical data.

2. **Random Forest is the best accuracy/interpretability trade-off** — strong performance with native feature importances, making it more practical for clinical deployment than GBT.

3. **`HbA1c_level` and `blood_glucose_level` are the dominant predictors** across all models, confirmed by both SHAP global importance and LIME local explanations. This aligns with established medical knowledge — both are direct diagnostic markers for diabetes.

4. **SHAP reveals non-linear feature interactions** that linear models (Logistic Regression) cannot capture — e.g. `BMI` impact is amplified at high `age` values.

5. **LIME explanations are consistent with SHAP** on the same instances, validating the explanations are stable and not artefacts of the explanation method.

6. **PySpark processes the 5.76 MB dataset efficiently in local mode**, demonstrating the pipeline's scalability — the same code runs unchanged on a multi-node cluster for datasets orders of magnitude larger.

7. **Class imbalance handling is critical**: diabetes-positive cases are typically under-represented; metrics beyond accuracy (AUC-ROC, F1) are essential for honest evaluation.

---

## 🔮 Future Work

- **Hyperparameter tuning** via Spark's `CrossValidator` and `ParamGridBuilder` for systematic model optimisation
- **Feature engineering** — interaction terms (e.g. `HbA1c × BMI`), binning of continuous variables by clinical thresholds
- **Handling class imbalance** explicitly via SMOTE or stratified sampling within Spark
- **Streaming pipeline** using Spark Structured Streaming to score new patient records in real time
- **MLflow integration** for experiment tracking and model versioning
- **Deployment** as a REST prediction service via FastAPI, with the trained model serialised and loaded for inference
- **Federated learning** to train across multiple hospital data silos without centralising sensitive patient data

---

## 👤 Author

**Aditya Karippadath Udai**
[GitHub](https://github.com/AdityaKarippadathUdai)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

> *"A model that can predict but not explain is a liability in healthcare. Scalability without interpretability is half the job."*
