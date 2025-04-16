# 🛡️ Network Intrusion Detection System (NIDS) using Machine Learning

A machine learning-based project focused on detecting network intrusions using classical models and ensemble learning techniques. The system processes network traffic data, classifies it into normal or anomaly classes, and provides a real-time web interface for prediction and live packet capture.

---

## 📂 Table of Contents

- [Overview](#📌-overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Data Preprocessing](#data-preprocessing)
- [Models Implemented](#models-implemented)
- [Evaluation Metrics](#evaluation-metrics)
- [Frontend Features](#frontend-features)
- [Live Packet Capture](#live-packet-capture)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Scope](#future-scope)
- [Contributors](#contributors)
- [Contact and Feedback](#contact-&-feedback)

---

## 📌 Overview

This project presents a scalable, real-time **Network Intrusion Detection System (NIDS)** using a blend of machine learning models. We evaluated a wide range of classifiers to detect anomalies in network traffic. Additionally, we developed a web-based frontend to interact with the system, making it accessible and intuitive for end users.

---

## 📊 Dataset

- **Name**: [Network Intrusion Dataset](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection/data)
- **Size**: ~25,000 rows × 42 columns
- **Class Distribution**:
  - `Normal`: 13,449
  - `Anomaly`: 11,743
  - **Ratio**: 1.15:1 (balanced)

---

## ⚙️ Tech Stack

### 🔍 Backend & ML
- Python
- Scikit-learn
- XGBoost
- Optuna
- Pandas, NumPy
- Matplotlib / Seaborn (for visualization)

### 💻 Frontend
- React.js
- RESTful APIs
- Vercel Deployment

---

## 🧹 Data Preprocessing

- Label encoding for categorical features (`protocol_type`, `service`, `flag`, `class`)
- Removed:
  - Columns with >50% missing data
  - Low-variance features (only one unique value)
  - Sparse features (>90% zeros)
- Applied `SelectKBest` with `mutual_info_classif` for feature selection
- 80-20 Train-Test split
- Final dataset saved as `train_data.csv` and `test_data.csv`

---

## 🧠 Models Implemented

| Type             | Algorithms Used |
|------------------|-----------------|
| Classical Models | Logistic Regression, KNN, Decision Tree, Naive Bayes |
| Ensemble Models  | Random Forest, AdaBoost, XGBoost, Voting Classifier |
| SVM Variants     | Linear SVM, Non-linear SVM (RBF kernel) |
| Neural Network   | Artificial Neural Network (ANN) |

Each model was optimized using `GridSearchCV` and evaluated on cross-validation folds.

---

## 📈 Evaluation Metrics

All models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🖥️ Frontend Features

Frontend built using React.js and deployed on **Vercel**. Key features include:

- 🔍 **Single Packet Prediction** — Enter packet features manually
- 📤 **CSV Upload** — Upload datasets for bulk classification
- 🌐 **Auto-Capture Mode** — Analyze real-time packet traffic (backend setup required)
- 📚 **History Section** — View past predictions (stored in localStorage)
- ℹ️ **About Section** — Overview of project, team, and usage instructions

---

## 🌐 Live Packet Capture

We provide a backend script (`packetcapture.py`) for real-time packet analysis.

### Requirements:
- `scapy`
- Admin/root permissions to capture packets

### How it works:
- Run the Python backend locally
- Hit "Start Capture" in the frontend
- Packets from your network interface will be captured, processed, and classified

---

## 📊 Results

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| XGBoost           | 99.66%   | 99.66%    | 99.66% | 99.66%   |
| Random Forest     | 99.50%   | 99.50%    | 99.50% | 99.50%   |
| KNN               | 99.36%   | 99.50%    | 99.50% | 99.50%   |
| Decision Tree     | 99.44%   | 99.44%    | 99.44% | 99.44%   |
| Non-Linear SVM    | 97.14%   | 95.24%    | 99.59% | 97.39%   |
| ANN               | 95.87%   | 95.87%    | 95.87% | 95.87%   |
| AdaBoost          | 96.59%   | 96.59%    | 96.59% | 96.59%   |
| Bernoulli Naive Bayes | 92.72% | 92.72% | 92.72% | 92.71%   |
| Logistic Regression | 88.71% | 88.72%    | 88.71% | 88.70%   |
| Linear SVM        | 89.86%   | 85.29%    | 97.72% | 91.13%   |

---

## 🧪 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/network-intrusion-detection.git
cd network-intrusion-detection
```

### 2. Setup Python Environment

```bash
pip install -r requirements.txt
```

### 3. Run Backend
```bash
python app.py
```

### 4. Setup and Run Frontend
```bash
cd frontend
npm install
npm run dev
```

🔧 Make sure to update the backend API URL in your frontend configuration if needed.

## 🚀 Future Scope

- Integrate deep learning models like CNNs or LSTMs for improved anomaly detection.

- Explore newer and more comprehensive datasets like CICIDS2017 or UNSW-NB15.

- Add real-time threat mitigation capabilities and alert systems.

- Deploy in enterprise environments with scalable cloud infrastructure.

- Integrate the system with SIEM (Security Information and Event Management) tools and dashboards.

---

## 📬 Contact & Feedback

If you have any suggestions, feedback, or encounter any issues:

- 🐛 Create an Issue
- 📬 Reach out to any contributor via email or GitHub
- 🤝 Feel free to fork the repository and contribute through Pull Requests

---

Thank you for exploring our project! 🚀

