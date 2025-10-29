from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
df = pd.read_csv('data.csv')

# Prepare data
X = df[['Hours']]
y = df['Marks']

# Train model
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours_input = request.form['hours']

    # Check for empty input
    if hours_input == '' or hours_input is None:
        return render_template('index.html', prediction_text="⚠️ Please enter study hours before predicting.")

    try:
        hours = float(hours_input)
        pred = model.predict([[hours]])
        return render_template('index.html', prediction_text=f'✅ Predicted Marks: {pred[0]:.2f}')
    except ValueError:
        return render_template('index.html', prediction_text="❌ Invalid input. Please enter a valid number.")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
