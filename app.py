from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('student_model.pkl')
except:
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    hours = None
    error = None
    plot_url = None
    
    if request.method == 'POST':
        hours_str = request.form.get('hours')
        try:
            hours = float(hours_str)
            if hours < 0 or hours > 24:
                error = "Please enter a logical number of hours (0-24)."
            elif model is not None:
                pred = model.predict(np.array([[hours]]))[0]
                prediction = round(np.clip(pred, 0, 100), 2)
                
                # Generate dynamic plot
                plt.figure(figsize=(8, 4.5))
                # Plot the regression line
                x_range = np.linspace(0, max(15, hours + 2), 100)
                y_range = model.coef_[0] * x_range + model.intercept_
                
                plt.plot(x_range, y_range, color='#6366f1', linewidth=2, label='Regression Line')
                plt.scatter([hours], [prediction], color='#ef4444', s=150, zorder=5, edgecolors='white', linewidth=2, label='Your Prediction')
                plt.axvline(x=hours, color='gray', linestyle='--', alpha=0.4)
                plt.axhline(y=prediction, color='gray', linestyle='--', alpha=0.4)
                
                # Set text color to white for dark theme
                plt.title('Your Prediction Visualization', color='white', fontsize=14, pad=10)
                plt.xlabel('Hours Studied', color='white')
                plt.ylabel('Predicted Marks', color='white')
                plt.tick_params(colors='white')
                
                # Legend with transparent background
                legend = plt.legend(facecolor='#1e293b', edgecolor='none')
                for text in legend.get_texts():
                    text.set_color('white')
                    
                plt.grid(True, linestyle='--', alpha=0.2)
                plt.tight_layout()
                
                # Save plot to base64 string
                img = io.BytesIO()
                plt.savefig(img, format='png', transparent=True, dpi=120)
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close()
                
            else:
                error = "Model not found. Please train the model first."
        except ValueError:
            error = "Please enter a valid number."
            
    return render_template('index.html', prediction=prediction, hours=hours, error=error, plot_url=plot_url)

if __name__ == '__main__':
    # Run the app on all interfaces to make sure it's accessible
    app.run(host='127.0.0.1', port=5000)
