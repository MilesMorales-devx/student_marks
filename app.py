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
    form_data = {'study': '', 'sleep': '', 'attendance': '', 'prev': ''}
    error = None
    plot_url = None
    
    if request.method == 'POST':
        form_data['study'] = request.form.get('study')
        form_data['sleep'] = request.form.get('sleep')
        form_data['attendance'] = request.form.get('attendance')
        form_data['prev'] = request.form.get('prev')
        
        try:
            study = float(form_data['study'])
            sleep = float(form_data['sleep'])
            attendance = float(form_data['attendance'])
            prev = float(form_data['prev'])
            
            # Basic validation
            if not (0 <= study <= 24 and 0 <= sleep <= 24 and 0 <= attendance <= 100 and 0 <= prev <= 100):
                error = "Please enter logical values (Hours: 0-24, Percentages: 0-100)."
            elif model is not None:
                # Predict
                features = np.array([[study, sleep, attendance, prev]])
                pred = model.predict(features)[0]
                prediction = round(np.clip(pred, 0, 100), 2)
                
                # Generate dynamic personalized 2D plot (Study Hours vs Marks)
                plt.figure(figsize=(8, 4.5))
                
                # Calculate the personalized intercept (holding sleep, attendance, prev constant)
                # Equation: y = m1*study + m2*sleep + m3*att + m4*prev + b
                # y = m1*study + (m2*sleep + m3*att + m4*prev + b)
                personalized_intercept = (model.coef_[1] * sleep) + (model.coef_[2] * attendance) + (model.coef_[3] * prev) + model.intercept_
                m1 = model.coef_[0]
                
                x_range = np.linspace(0, max(15, study + 2), 100)
                y_range = m1 * x_range + personalized_intercept
                
                plt.plot(x_range, y_range, color='#6366f1', linewidth=2, label='Personalized Trend')
                plt.scatter([study], [prediction], color='#ef4444', s=150, zorder=5, edgecolors='white', linewidth=2, label='Your Prediction')
                plt.axvline(x=study, color='gray', linestyle='--', alpha=0.4)
                plt.axhline(y=prediction, color='gray', linestyle='--', alpha=0.4)
                
                plt.title('How Study Hours Affect Your Score', color='white', fontsize=14, pad=10)
                plt.xlabel('Study Hours', color='white')
                plt.ylabel('Predicted Marks', color='white')
                plt.tick_params(colors='white')
                
                legend = plt.legend(facecolor='#1e293b', edgecolor='none')
                for text in legend.get_texts():
                    text.set_color('white')
                    
                plt.grid(True, linestyle='--', alpha=0.2)
                plt.tight_layout()
                
                img = io.BytesIO()
                plt.savefig(img, format='png', transparent=True, dpi=120)
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close()
                
            else:
                error = "Model not found. Please train the model first."
        except ValueError:
            error = "Please enter valid numbers for all fields."
            
    return render_template('index.html', prediction=prediction, form_data=form_data, error=error, plot_url=plot_url)

if __name__ == '__main__':
    # Run the app on all interfaces to make sure it's accessible
    app.run(host='127.0.0.1', port=5000)
