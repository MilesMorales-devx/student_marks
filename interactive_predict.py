import joblib
import numpy as np
import warnings
import sys

# Suppress sklearn warnings about feature names lacking for single predictions
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    print("="*50)
    print(" Interactive Student Marks Predictor ")
    print("="*50)
    
    try:
        # Load the saved model
        model = joblib.load('student_model.pkl')
    except FileNotFoundError:
        print("Error: Model file 'student_model.pkl' not found.")
        print("Please run 'python predict_marks.py' first to train and save the model.")
        sys.exit(1)

    print("Model loaded successfully! Type 'exit' or 'quit' to close the program.\n")

    while True:
        user_input = input("Enter hours studied (or 'exit'): ").strip().lower()
        
        if user_input in ['exit', 'quit']:
            print("Closing interactive predictor. Goodbye!")
            break
        
        try:
            # Convert user input to float
            hours = float(user_input)
            
            # Predict the score (Sklearn expects a 2D array, so we reshape)
            prediction = model.predict(np.array([[hours]]))
            
            # Clip between 0 and 100 since marks cannot logically exceed this bounds
            score = np.clip(prediction[0], 0, 100)
            
            print(f">>> PREDICTION: If you study {hours} hours, your predicted marks are {score:.2f}/100\n")
            
        except ValueError:
            print("Invalid input. Please enter a numerical value (e.g., 4.5).\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Handle ctrl+c gracefully
        print("\nProgram interrupted. Goodbye!")
        sys.exit(0)
