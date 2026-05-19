import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

def main():
    print("Generating synthetic dataset...")
    # 1. Create a synthetic dataset
    np.random.seed(42)
    n_students = 100
    
    # Features
    study_hours = np.random.uniform(1, 10, n_students)
    sleep_hours = np.random.uniform(4, 10, n_students)
    attendance_pct = np.random.uniform(50, 100, n_students)
    prev_grade = np.random.uniform(40, 100, n_students)
    
    # Suppose the relationship is: Marks = 5 + 3*study + 1.5*sleep + 0.2*attendance + 0.35*prev + noise
    noise = np.random.normal(0, 4, size=n_students)
    marks = 5 + (3.0 * study_hours) + (1.5 * sleep_hours) + (0.2 * attendance_pct) + (0.35 * prev_grade) + noise
    # Clip marks to a maximum of 100
    marks = np.clip(marks, 0, 100)

    data = pd.DataFrame({
        'Study_Hours': study_hours,
        'Sleep_Hours': sleep_hours,
        'Attendance_Pct': attendance_pct,
        'Prev_Grade': prev_grade,
        'Marks': marks
    })
    data.to_csv('student_scores.csv', index=False)
    print("Dataset saved as 'student_scores.csv'")

    # 2. Split dataset into training and testing sets
    X = data[['Study_Hours', 'Sleep_Hours', 'Attendance_Pct', 'Prev_Grade']].values
    y = data['Marks'].values

    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training the model...")
    # 3. Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Make predictions
    y_pred = model.predict(X_test)

    # 5. Evaluate the model
    r2 = r2_score(y_test, y_pred)
    print(f"Model Summary:\n  Coefficients: {model.coef_}")
    print(f"  Intercept (b): {model.intercept_:.2f}")
    print(f"  R2 Score: {r2:.4f}")

    # 6. Plotting
    print("Generating visualization...")
    plt.figure(figsize=(10, 6))

    # Plot Actual vs Predicted
    plt.scatter(y_test, y_pred, color='purple', s=80, alpha=0.7, label='Predictions')
    
    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction (y=x)')

    plt.title('Actual vs Predicted Marks (Multiple Regression)', fontsize=14)
    plt.xlabel('Actual Marks', fontsize=12)
    plt.ylabel('Predicted Marks', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('regression_plot.png', dpi=300)
    print("Saved plot to 'regression_plot.png'")

    # Save the model to disk so we can load it interactively
    import joblib
    joblib.dump(model, 'student_model.pkl')
    print("Trained model saved to 'student_model.pkl'")

if __name__ == "__main__":
    main()
