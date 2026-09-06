# Heart Disease Prediction App

An end-to-end machine learning project that predicts a patient's risk of heart disease based on clinical parameters, deployed as an interactive web application using **Streamlit**.



---

##  Overview

This project takes raw clinical data, cleans and analyzes it, engineers features using statistical tests, trains and compares multiple machine learning models, and deploys the best-performing model as a user-friendly web app that predicts heart disease risk in real time.

---

## Dataset

- **Source:** `heart.csv`
- **Records:** 918 patients
- **Features:** Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina, Oldpeak (ST Depression), ST Slope
- **Target:** `HeartDisease` (1 = presence of heart disease, 0 = no heart disease)

---

##  Project Workflow

1. **Data Cleaning**
   - Handled invalid zero-values in `Cholesterol` and `RestingBP` by replacing them with the column mean.
2. **Exploratory Data Analysis (EDA)**
   - Distribution plots, count plots, and correlation analysis using Seaborn/Matplotlib.
3. **Encoding**
   - One-hot encoding of categorical variables using `pd.get_dummies()`.
4. **Feature Selection**
   - **Pearson correlation** for numerical/binary features.
   - **Chi-Square test** for categorical features to statistically validate feature relevance before finalizing the feature set.
5. **Model Training & Comparison**
   - Trained and evaluated 5 classification models on an 67/33 train-test split with standardized features.
6. **Model Deployment**
   - Best model, scaler, and feature columns serialized using `joblib` and deployed via a Streamlit web app.

---

##  Model Performance

| Model               | Accuracy | F1 Score |
|---------------------|----------|----------|
| Logistic Regression | 0.8515   | 0.8696   |
| K-Nearest Neighbors | 0.8482   | 0.8671   |
| Naive Bayes         | 0.8548   | 0.8728   |
| Decision Tree       | 0.7360   | 0.7561   |
| **SVM (Selected)**  | **0.8647** | **0.8845** |

The **Support Vector Machine (SVM)** model achieved the best accuracy and F1 score and was selected as the final production model.

---

##  Web Application

The app is built with **Streamlit** and allows users to:
- Enter patient health parameters through an interactive sidebar
- View a summary of the entered details
- Get an instant prediction — **High Risk** or **Low Risk** of heart disease
- View the exact data sent to the model for transparency

### Screenshot

_(Add a screenshot of the app here — drag and drop an image into this README on GitHub, e.g. `![App Screenshot](screenshot.png)`)_

---

##  Running the App Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/ishikasharma871/Heart-Disease-Prediction-App.git
   cd Heart-Disease-Prediction-App
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Statistical Testing:** SciPy (Pearson correlation, Chi-Square test)
- **Machine Learning:** Scikit-learn (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM)
- **Model Persistence:** Joblib
- **Web App / Deployment:** Streamlit

---

##  Repository Structure

```
Heart-Disease-Prediction-App/
├── app.py                       # Streamlit web app
├── EDA_1Heart_complete.ipynb    # Full EDA, feature selection & model training notebook
├── heart.csv                    # Dataset
├── SVM_heart.pkl                # Trained SVM model
├── scaler.pkl                   # StandardScaler used for feature scaling
├── columns.pkl                  # Feature column order expected by the model
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

##  Future Improvements

- Enable `probability=True` in the SVM model to display prediction confidence scores in the app
- Deploy the app publicly via Streamlit Community Cloud
- Add model explainability (e.g., SHAP values) to show which features drove a given prediction
- Experiment with hyperparameter tuning to improve the Decision Tree's performance

---

##  Author

**Ishika Sharma**
Aspiring Data Analyst | BCA Student

 If you found this project useful, consider giving it a star!
