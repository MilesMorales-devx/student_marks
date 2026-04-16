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
    # 25 students, studying between 1 and 10 hours
    hours = np.linspace(1, 10, 25) 
    
    # Suppose the relationship is: Marks = 5 + 9 * hours + noise
    noise = np.random.normal(0, 5, size=len(hours))
    marks = 5 + 9 * hours + noise
    # Clip marks to a maximum of 100
    marks = np.clip(marks, 0, 100)

    data = pd.DataFrame({'Hours': hours, 'Marks': marks})
    data.to_csv('student_scores.csv', index=False)
    print("Dataset saved as 'student_scores.csv'")

    # 2. Split dataset into training and testing sets
    X = data[['Hours']].values
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
    print(f"Model Summary:\n  Coefficient (Slope/m): {model.coef_[0]:.2f}")
    print(f"  Intercept (b): {model.intercept_:.2f}")
    print(f"  R2 Score: {r2:.4f}")

    # 6. Plotting
    print("Generating visualization...")
    plt.figure(figsize=(10, 6))

    # Plot training data
    plt.scatter(X_train, y_train, color='blue', label='Training Data', alpha=0.7)

    # Plot testing data
    plt.scatter(X_test, y_test, color='green', label='Testing Data (Actual)', s=80, marker='x')

    # Plot predictions for testing data
    plt.scatter(X_test, y_pred, color='purple', label='Testing Data (Predicted)', s=80, marker='+')

    # Plot regression line
    line = model.coef_ * X + model.intercept_
    plt.plot(X, line, color='red', label='Regression Line (y = mx + b)')

    plt.title('Student Marks vs Study Hours', fontsize=14)
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Marks Obtained', fontsize=12)
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
