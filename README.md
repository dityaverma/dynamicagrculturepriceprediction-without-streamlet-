# 🌾 Dynamic Agriculture Price Prediction

## 📌 Overview
This project predicts the **modal price of agricultural commodities** using machine learning techniques. It leverages historical market data and commodity-specific attributes to help understand price trends and support data-driven decision-making.

---

## 🎯 Objective
To build a predictive system that estimates agricultural commodity prices based on:
- Market location
- Commodity type
- Arrival date
- Other influencing factors

This can assist farmers, traders, and policymakers in making informed decisions.

---

## 🧠 Tech Stack
- **Python**
- **scikit-learn**
- **XGBoost**
- **Pandas, NumPy**
- **Matplotlib**
- **Joblib**

---

## 📊 Dataset
- Source: **data.gov.in**
- Region: Nashik district, Maharashtra  
- Key Features:
  - Market
  - Commodity
  - Variety
  - Grade
  - Arrival Date
  - Commodity Code  
- Target Variable:
  - **Modal Price**

### 🔧 Data Processing
- Encoding categorical variables  
- Date feature transformation  
- Cleaning and structuring raw data  

---

## 🚀 Features
- End-to-end data preprocessing pipeline  
- Multiple ML models (Linear Regression, XGBoost)  
- Model persistence using Joblib  
- Structured and modular codebase  

---

## ⚙️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run prediction script
python prediction_using_model.py
