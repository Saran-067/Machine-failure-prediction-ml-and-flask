
# ⚙️ Machine Failure Prediction: Predictive Maintenance
A machine learning model to predict whether a failure occurs in a machine or not using svm,logistic regression and random forest algorithms.


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?logo=googlecolab)](https://colab.research.google.com/)

An end-to-end Machine Learning project to predict whether a machine will fail and identify the specific type of failure using industrial sensor data. This project addresses the critical real-world problem of **Predictive Maintenance**, helping industries reduce downtime and repair costs.

---

## 📌 Project Overview
Industrial machines generate various sensor readings. By monitoring these in real-time, we can predict failures before they occur. This project classifies machine status into **No Failure** or one of **5 specific failure types**.

* **Dataset:** 10,000 instances of industrial machine data.
* **Techniques:** SMOTE (Oversampling), Feature Engineering, Multi-class Classification.
* **Deployment:** Flask-based Web Interface (Ready for deployment).

---

## 🔍 Feature Analysis
The model uses the following physical parameters to detect specific failure modes:

| Feature | Description | Related Failure Type |
| :--- | :--- | :--- |
| **Air Temperature [K]** | Ambient temperature around the machine | Heat Dissipation Failure (HDF) |
| **Process Temp [K]** | Internal temperature during operation | Heat Dissipation Failure (HDF) |
| **Rotational Speed [rpm]** | How fast the tool is spinning | Power (PWF) & Overstrain (OSF) |
| **Torque [Nm]** | The rotational force applied | Power (PWF) & Overstrain (OSF) |
| **Tool Wear [min]** | Usage duration of the cutting tool | Tool Wear Failure (TWF) |
| **Type** | Load level: **L** (Low), **M** (Medium), **H** (High) | General Stress Levels |

---

## 🛠️ Technical Workflow

### 1. Data Imbalance & SMOTE
Real-world failure data is highly imbalanced (failures are rare).
* **Before Balancing:** Normal cases: 9,670 | Failures: ~330 total.
* **Solution:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic failure cases.
* **After Balancing:** Each class (including specific failures) has **9,670** samples, totaling **48,350** instances for robust training.

### 2. Failure Types Classified
1. **No Failure:** Machine is operating normally.
2. **Heat Dissipation Failure (HDF):** Overheating due to ventilation issues.
3. **Power Failure (PWF):** Electrical breakdown or power loss.
4. **Overstrain Failure (OSF):** Mechanical load beyond limits.
5. **Tool Wear Failure (TWF):** The cutting tool has reached its wear limit.
6. **Random Failure:** Unforeseen mechanical glitches.

---

## 📈 Model Performance
Multiple algorithms were evaluated. The ensemble methods provided the highest accuracy:

| Model | Accuracy |
| :--- | :--- |
| **Random Forest** | **99.4%** ✅ |
| **Decision Tree** | **99.2%** |
| **SVM** | **98.1%** |
| **Logistic Regression** | **95.7%** |

---

## 🚀 Installation & Setup

### 1. Run on Google Colab
1. Upload your dataset (`machine_failure.csv`) to your Drive or local Colab session.
2. Install the necessary libraries:
   ```python
   !pip install imbalanced-learn flask opencv-python
