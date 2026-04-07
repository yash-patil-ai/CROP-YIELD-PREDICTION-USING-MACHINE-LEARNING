
<div align="center">

# 🌾 Crop Yield Prediction using Machine Learning

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&pause=1000&color=2ECC71&center=true&vCenter=true&width=600&lines=Predicting+Crop+Yields+with+AI+%F0%9F%8C%B1;Empowering+Farmers+with+Data+Science+%F0%9F%9A%9C;Smart+Agriculture+for+a+Better+Tomorrow+%F0%9F%8C%8D" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> 🚜 *Harnessing the power of Machine Learning to help farmers predict crop yields and make smarter agricultural decisions.*

</div>

---

## 📌 Table of Contents

- [📖 About the Project](#-about-the-project)
- [✨ Features](#-features)
- [🧠 ML Models Used](#-ml-models-used)
- [📊 Dataset](#-dataset)
- [🗂️ Project Structure](#️-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [📈 Results](#-results)
- [🛠️ Tech Stack](#️-tech-stack)
- [🤝 Contributing](#-contributing)
- [👨‍💻 Author](#-author)

---

## 📖 About the Project

<img align="right" src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdXJhcnFvbnpvbzFtNmx5bHFkZmx6ZGJ0Y3VmbzYxcHBleWNwNWFuYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/RbDKaczqWovIugyJmW/giphy.gif" width="250"/>

Agriculture is the backbone of India's economy, yet farmers face immense uncertainty around crop output every season. This project bridges the gap between **data science** and **farming** by building a machine learning model that predicts crop yield based on key environmental and agricultural parameters.

By analyzing historical data on:
- 🌡️ Temperature & Rainfall
- 🌱 Soil Nutrients (N, P, K)
- 📍 Region / State / District
- 🗓️ Season & Crop Type
- 📐 Area of agricultural land

…the model empowers farmers and policymakers to **plan better, reduce losses, and improve food security**.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🔍 **Data Analysis** | In-depth EDA with visualizations of crop patterns |
| 🧹 **Data Preprocessing** | Handles missing values, encoding, and scaling |
| 🤖 **Multiple ML Models** | Trained & compared across several algorithms |
| 📊 **Performance Metrics** | R² Score, RMSE, MAE evaluated for each model |
| 🌾 **Yield Prediction** | Predict output (in kg/hectare) for given inputs |
| 📓 **Jupyter Notebook** | Clean, well-documented notebook workflow |

---

## 🧠 ML Models Used

```
📦 Models Trained & Compared
 ┣ 📌 Linear Regression
 ┣ 📌 Decision Tree Regressor
 ┣ 📌 Random Forest Regressor       ← 🏆 Best Performer
 ┣ 📌 Gradient Boosting Regressor
 ┗ 📌 Support Vector Regressor (SVR)
```

> ✅ **Random Forest** achieved the highest accuracy with an R² score of **~87%**

---

## 📊 Dataset

The dataset includes historical crop production data with the following key features:

| Column | Description |
|--------|-------------|
| `State_Name` | State where crop is grown |
| `District_Name` | District-level granularity |
| `Crop_Year` | Year of cultivation |
| `Season` | Kharif / Rabi / Whole Year |
| `Crop` | Type of crop |
| `Area` | Area under cultivation (hectares) |
| `Production` | Crop production (tonnes) — **Target Variable** |

> 📁 Dataset Source: [data.gov.in](https://data.gov.in) / Kaggle Agriculture Dataset

---

## 🗂️ Project Structure

```
📦 CROP-YIELD-PREDICTION-USING-MACHINE-LEARNING
 ┣ 📂 dataset/
 ┃  ┗ 📄 crop_data.csv
 ┣ 📂 notebooks/
 ┃  ┗ 📓 Crop_Yield_Prediction.ipynb
 ┣ 📂 models/
 ┃  ┗ 🤖 random_forest_model.pkl
 ┣ 📂 static/
 ┃  ┗ 🖼️ visualizations/
 ┣ 📄 app.py
 ┣ 📄 requirements.txt
 ┗ 📄 README.md
```

---

## ⚙️ Installation

### Prerequisites
Make sure you have **Python 3.8+** installed.

### Steps

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yash-patil-ai/CROP-YIELD-PREDICTION-USING-MACHINE-LEARNING.git

# 2️⃣ Navigate into the project folder
cd CROP-YIELD-PREDICTION-USING-MACHINE-LEARNING

# 3️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 4️⃣ Install dependencies
pip install -r requirements.txt

# 5️⃣ Launch the Jupyter Notebook
jupyter notebook
```

---

## 🚀 Usage

```python
# Quick prediction example
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('models/random_forest_model.pkl', 'rb'))

# Input: [State, Crop, Season, Area, Year] (encoded)
input_data = np.array([[5, 12, 1, 1500, 2023]])

# Predict
predicted_yield = model.predict(input_data)
print(f"Predicted Crop Yield: {predicted_yield[0]:.2f} kg/hectare")
```

---

## 📈 Results

<div align="center">

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.72 | 1842 | 1120 |
| Decision Tree | 0.78 | 1534 | 980 |
| **Random Forest** | **0.87** | **1102** | **742** |
| Gradient Boosting | 0.85 | 1230 | 810 |
| SVR | 0.69 | 1980 | 1350 |

</div>

> 📌 *Random Forest Regressor* outperformed all other models and was selected as the **final model**.

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat-square&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/-Seaborn-4c72b0?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>

---

## 🤝 Contributing

Contributions are always welcome! Here's how:

```bash
# Fork the repo → Create your branch → Make changes → Open a Pull Request
git checkout -b feature/your-feature-name
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on the code of conduct.

---

## 👨‍💻 Author

<div align="center">

**Yash Patil**

[![GitHub](https://img.shields.io/badge/GitHub-yash--patil--ai-181717?style=for-the-badge&logo=github)](https://github.com/yash-patil-ai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/yash-patil)

*"Using AI to solve real-world problems — one harvest at a time."* 🌾

</div>

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

![Footer](https://capsule-render.vercel.app/api?type=waving&color=2ECC71&height=100&section=footer)

</div>

