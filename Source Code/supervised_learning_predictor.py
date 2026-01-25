\"\"\"
-----------------------------------------------------------------------------------------
THE SPARKS FOUNDATION - DATA SCIENCE AND BUSINESS ANALYTICS INTERNSHIP (GRIP JULY 2021)
-----------------------------------------------------------------------------------------
TASK 1: PREDICTION USING SUPERVISED MACHINE LEARNING
GOAL: To predict the percentage of a student based on the number of study hours.
MODEL: Simple Linear Regression (as it involves just two variables).

AUTHORS:
- Amey Thakur (https://github.com/Amey-Thakur)
- Mega Satish (https://github.com/msatmod)

REPOSITORY: https://github.com/Amey-Thakur/TSF-SUPERVISED-MACHINE-LEARNING

DESCRIPTION:
This scholarly implementation demonstrates the application of classical linear regression 
to establish a predictive relationship between academic study time and achievement scores. 
The analysis encompasses data pre-processing, exploratory visualization, model training, 
robust evaluation using academic metrics (R-squared, Adjusted R-squared, MAE), 
and a final inference for a specific query (9.25 hours of study).

TECHNOLOGIES: Python 3, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn.
DATA SOURCE: http://bit.ly/w-data
-----------------------------------------------------------------------------------------
\"\"\"

# =========================================================================================
# STEP 1: IMPORTING ESSENTIAL SCHOLARLY LIBRARIES
# =========================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from math import sqrt

# Setting visual aesthetics for scholarly presentation
sns.set(style="whitegrid")
plt.style.use('ggplot')

def main():
    \"\"\"
    Main execution pipeline for the Supervised Learning Task.
    \"\"\"
    
    # =====================================================================================
    # STEP 2: DATA ACQUISITION AND EXPLORATORY DATA ANALYSIS (EDA)
    # =====================================================================================
    # Loading the dataset from the local environment
    dataset_path = "scores.csv"
    try:
        data = pd.read_csv(dataset_path)
        print("[INFO] Dataset successfully loaded from '{}'.".format(dataset_path))
    except FileNotFoundError:
        print("[ERROR] 'scores.csv' not found. Please ensure it is in the correct directory.")
        return

    # Visualizing the first few observations for structural understanding
    print("\n--- First 10 Observations ---")
    print(data.head(10))

    # Descriptive statistical summary
    print("\n--- Statistical Descriptive Analysis ---")
    print(data.describe())

    # Visualizing the underlying linear distribution
    print("\n[V] Generating distribution plot...")
    plt.figure(figsize=(10, 6))
    data.plot(x='Hours', y='Scores', color='green', style='o')  
    plt.title('Hours Studied vs. Percentage Scored (Initial Distribution)')  
    plt.xlabel('Study Hours (Independent Variable)')  
    plt.ylabel('Achievement Score (Dependent Variable)')  
    plt.grid(True)
    plt.show()

    # =====================================================================================
    # STEP 3: DATA PRE-PROCESSING AND PARTITIONING
    # =====================================================================================
    # Extracting features (X) and target (y) vectors
    # iloc is used for precise integer-location based indexing
    X = data.iloc[:, :-1].values  # Study Hours
    y = data.iloc[:, 1].values   # Scores

    # Splitting the corpus into Training and Validation sets (80/20 ratio)
    # Random state ensures consistency/reproducibility of scientific results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print("\n[INFO] Data partitioned into Training (80%) and Test (20%) sets.")

    # =====================================================================================
    # STEP 4: MODEL SPECIFICATION AND TRAINING
    # =====================================================================================
    # Initializing the Ordinary Least Squares (OLS) Linear Regression model
    regressor = LinearRegression()
    
    # Fitting the model to the theoretical space of the training data
    regressor.fit(X_train, y_train)
    print("[INFO] Model training completed successfully.")

    # =====================================================================================
    # STEP 5: DIAGNOSTIC VISUALIZATION AND PREDICTION
    # =====================================================================================
    # Visualizing the regression line relative to the training data
    print("[V] Generating training regression fit visualization...")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='green', label='Observed Data')
    plt.plot(X_train, regressor.predict(X_train), color='blue', linewidth=2, label='Regression Line')
    plt.title('Achievement Score Prediction: Training Set Regression Analysis')
    plt.xlabel('Study Hours')
    plt.ylabel('Achievement Score')
    plt.legend()
    plt.show()

    # Predicting results for the unseen Test Set
    y_pred = regressor.predict(X_test)

    # Comparison analysis: Empirical vs. Predicted values
    print("\n--- Empirical Performance Comparison (Actual vs Predicted) ---")
    comparison_df = pd.DataFrame({'Empirical_Actual': y_test, 'Model_Predicted': y_pred})
    print(comparison_df)

    # =====================================================================================
    # STEP 6: SCHOLARLY EVALUATION METRICS
    # =====================================================================================
    # Calculating key diagnostic parameters to validate model integrity
    k = X_test.shape[1]  # Number of predictors
    n = len(X_test)      # Number of observations in test set
    
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Model Evaluation Results ---")
    print("Coefficient of Determination (R2 Score): {:.4f}".format(r2))
    print("Adjusted R-Squared: {:.4f}".format(adj_r2))
    print("Mean Absolute Error (MAE): {:.4f}".format(mae))

    # =====================================================================================
    # STEP 7: INFERENCE FOR SPECIFIC QUERY (9.25 HOURS)
    # =====================================================================================
    query_hours = 9.25
    final_prediction = regressor.predict([[query_hours]])
    
    print("\n--- Final Inference Result ---")
    print("Input: Study Intensity of {} hours per day.".format(query_hours))
    print("Inference: Predicted Achievement Score = {:.2f}%".format(final_prediction[0]))

if __name__ == "__main__":
    main()
