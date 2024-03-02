from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('XGBoost', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Access form data
    radius_mean = request.form['radius_mean']
    texture_mean = request.form['texture_mean']
    perimeter_mean = request.form['perimeter_mean']
    area_mean = request.form['area_mean']
    smoothness_mean = request.form['smoothness_mean']
    compactness_mean = request.form['compactness_mean']
    concavity_mean = request.form['concavity_mean']
    concave_points_mean = request.form['concave points_mean']
    symmetry_mean = request.form['symmetry_mean']
    fractal_dimension_mean = request.form['fractal_dimension_mean']
    radius_se = request.form['radius_se']
    # ... continue for other columns

    # Convert input to a format suitable for your model
    input_data = [[float(radius_mean), float(texture_mean), float(perimeter_mean),float(area_mean),float(smoothness_mean),float(compactness_mean),float(concavity_mean),float(concave_points_mean),float(symmetry_mean),float(fractal_dimension_mean),float(radius_se)]]

    # Make prediction
    prediction = model.predict(input_data)

    # Convert the prediction to a readable format
    if prediction == 0:
        result = 'No cancer detected'
    else:
        result = 'Cancer detected'

    return result

if __name__ == '__main__':
    app.run(debug=True)