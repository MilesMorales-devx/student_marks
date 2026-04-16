import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    # 1. Generate Fake Dataset
    np.random.seed(42)
    # Study hours mostly between 1 and 10
    hours = np.random.uniform(1, 10, 100)
    # Marks depend linearly on hours, with some noise
    marks = 15 + 8.2 * hours + np.random.normal(0, 6, 100)
    marks = np.clip(marks, 0, 100) # Ensure marks are within 0-100

    data = pd.DataFrame({'Hours': hours, 'Marks': marks})
    data.to_csv('dataset.csv', index=False)
    print("Dataset 'dataset.csv' created.")

    # 2. Load Dataset
    df = pd.read_csv('dataset.csv')
    X = df[['Hours']]
    y = df['Marks']

    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Predict
    y_pred = model.predict(X_test)

    # 6. Evaluation
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2:.4f}")

    # 7. Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='#3498db', label='Actual Data', alpha=0.7, edgecolors='k')
    plt.plot(X_test, y_pred, color='#e74c3c', linewidth=2, label='Regression Line (Predicted)')
    
    plt.title('Student Marks Prediction: Study Hours vs. Marks', fontsize=14, fontweight='bold')
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Marks Obtained', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add formula
    m = model.coef_[0]
    b = model.intercept_
    formula_text = f'y = {m:.2f}x + {b:.2f}'
    plt.text(2, 90, formula_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('regression_plot.png', dpi=300)
    print("Plot saved as 'regression_plot.png'.")

if __name__ == "__main__":
    main()
