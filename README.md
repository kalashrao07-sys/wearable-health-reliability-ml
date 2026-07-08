# 🩺 Reliability of Health Data from Wearables using Machine Learning-Based Human Activity Recognition

> A Machine Learning framework for evaluating the reliability of wearable health data through Human Activity Recognition (HAR) using the PAMAP2 dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Application-black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-success)

---

## 📌 Project Overview

Wearable devices such as smartwatches and fitness trackers continuously collect health-related sensor data. However, these measurements are often affected by motion artifacts, sensor noise, environmental factors, and subject variability, making their reliability uncertain.

This project develops a **Machine Learning-based Human Activity Recognition (HAR)** system that not only predicts physical activities but also evaluates the **reliability of wearable health data** using confidence-based analysis.

Unlike traditional HAR systems that only classify activities, this framework estimates the trustworthiness of predictions, making it more suitable for healthcare and fitness monitoring applications.

---

## 🎯 Objectives

- Develop an accurate Human Activity Recognition system.
- Improve prediction performance using Ensemble Learning.
- Evaluate the reliability of wearable sensor predictions.
- Provide confidence-based reliability classification.
- Build a real-time Flask-based demonstration dashboard.

---

## 🌍 Real-World Applications

- 🏥 Healthcare Monitoring
- ⌚ Smart Wearables
- 👴 Elderly Care
- 🏃 Fitness Tracking
- 🦿 Rehabilitation
- 📊 Sports Performance Analysis

---

# 📂 Dataset

**Dataset:** PAMAP2 Physical Activity Monitoring Dataset

**Source:** UCI Machine Learning Repository

The dataset contains synchronized wearable sensor recordings collected from multiple subjects performing daily activities.

### Sensors Used

- Hand IMU
- Chest IMU
- Ankle IMU
- Heart Rate Monitor

### Activities

- Lying
- Sitting
- Standing
- Walking
- Running
- Cycling
- Nordic Walking
- Ascending Stairs
- Descending Stairs
- Vacuum Cleaning
- Ironing

---

# ⚙️ Project Workflow

```
Raw PAMAP2 Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Sliding Window Segmentation
        │
        ▼
Feature Engineering
(308 Features)
        │
        ▼
Machine Learning Models
        │
        ▼
Voting Ensemble
        │
        ▼
Activity Prediction
        │
        ▼
Reliability Analysis
        │
        ▼
Visualization Dashboard
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Missing value handling
- Removal of invalid activities
- Sliding window segmentation
- Window purity filtering
- Feature extraction
- Temporal smoothing

### Parameters

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 100 Hz |
| Window Size | 100 Samples |
| Step Size | 50 Samples |
| Purity Threshold | 85% |
| Smoothing Window | 19 |

---

# 📈 Feature Engineering

More than **308 engineered features** were extracted from each sensor window.

### Time Domain

- Mean
- Standard Deviation
- RMS
- Energy
- Range
- Signal Magnitude Area

### Frequency Domain

- FFT
- Dominant Frequency
- Spectral Entropy

### Advanced Features

- Gyroscope Magnitude
- Zero Crossing Rate
- Autocorrelation
- Skewness
- Kurtosis
- Rolling Stability Features

---

# 🤖 Machine Learning Models

The following algorithms were evaluated:

| Model | Purpose |
|--------|---------|
| Decision Tree | Baseline |
| K-Nearest Neighbors | Distance-based Learning |
| Random Forest | Ensemble Trees |
| XGBoost | Gradient Boosting |
| **Voting Ensemble** | Final Model |

---

# 📊 Model Performance

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|--------|----------|-------------------|----------|
| Decision Tree | 81.6% | 79.9% | 0.794 |
| KNN | 89.5% | 88.1% | 0.885 |
| Random Forest | 91.4% | 90.5% | 0.905 |
| XGBoost | 91.4% | 90.5% | 0.905 |
| ⭐ Voting Ensemble | **92.3%** | **91.3%** | **0.914** |

---

# 🚦 Reliability Analysis

The system evaluates prediction confidence and classifies each prediction into:

🟢 **Green** → Reliable

🟡 **Yellow** → Moderately Reliable

🔴 **Red** → Unreliable

This additional reliability layer helps determine whether wearable sensor predictions can be trusted.

---

# 📊 Visualizations

The project generates:

- Activity Distribution
- Sensor Distribution
- Heart Rate Analysis
- Raw Signal Analysis
- Feature Importance
- Model Comparison
- Confusion Matrix
- LOSO Results
- Reliability Distribution
- Confidence Calibration
- Sensor Importance
- Noise Injection Analysis

---

# 🖥️ Web Dashboard

The Flask dashboard provides:

- Live Sensor Simulation
- Activity Prediction
- Confidence Score
- Reliability Status
- Interactive Visualizations

---

# 📁 Repository Structure

```
wearable-health-reliability-ml
│
├── app/
├── scripts/
├── graphs/
├── results/
├── notebooks/
├── data/
├── models/
├── presentation/
├── paper/
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

```bash
git clone <repository-url>

cd wearable-health-reliability-ml

pip install -r requirements.txt
```

---

# ▶️ Running the Project

Train the model

```bash
python scripts/pamap2_ml_model.py
```

Run the Flask application

```bash
python app/app.py
```

Open:

```
http://localhost:5000
```

---

# 📌 Key Contributions

✔ Comprehensive preprocessing pipeline

✔ 308 engineered wearable sensor features

✔ Ensemble Machine Learning framework

✔ Confidence-based reliability analysis

✔ Flask web application

✔ Interactive visualization dashboard

---

# 🔮 Future Work

- Deep Learning (CNN/LSTM/Transformer)
- Real-time wearable deployment
- IoT integration
- Personalized activity recognition
- Mobile application development

---

# 👨‍💻 Authors

**Kalash Rao**

Computer Science & Engineering (AI)

KLE Technological University, Belagavi

---

# 📜 License

This project is released under the MIT License.

---

# 📖 Citation

If you use this work for academic or research purposes, please cite the repository after publication.

---

⭐ If you find this project useful, consider giving the repository a star.
