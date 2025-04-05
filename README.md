# 🚨 ChurnShield Pro

**AI-Powered Customer Churn Prediction Dashboard**

ChurnShield Pro is a Streamlit-based interactive dashboard that predicts the risk of customer churn using a trained machine learning model. It provides actionable insights, visualizations, and personalized recommendations based on user inputs.

---

## 📦 Features

- Interactive form-based input for customer metrics
- Real-time churn prediction using a trained ML pipeline
- Risk classification (High / Medium / Low)
- Visualization of key churn drivers
- Smart recommendations based on risk level
- Responsive design and custom CSS for modern UX
- Sidebar with usage instructions and model performance

---

## 🧠 Model Overview

- **Algorithm**: Custom ML model (XGBoost, LGBoost and CatBoost)
- **Preprocessing**: Handled via scikit-learn pipeline
- **Performance**:
  - **Recall**: 92%
  - **Precision**: 85%
  - **F1 Score**: 0.88

The model was trained and saved as a `joblib` file, with metadata and a preprocessor embedded.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/AyobamiMichael/ChurnShield.git
cd ChurnShield.git

