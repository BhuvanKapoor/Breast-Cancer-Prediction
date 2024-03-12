import pickle
import numpy as np
import tkinter as tk
from tkinter import messagebox
import warnings
warnings.filterwarnings("ignore")

# Load the trained model
model = pickle.load(open('Random Forest', 'rb'))

# Load the fitted scaler
scaler = pickle.load(open('scaler','rb'))

# Define the input field names
input_field_names = ['Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean',
                     'Smoothness Mean', 'Compactness Mean', 'Concavity Mean',
                     'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean']

# Create the main window
window = tk.Tk()
window.title("Cancer Prediction")

# Create input fields
input_fields = []
for name in input_field_names:
    tk.Label(window, text=name).grid(row=input_field_names.index(name), column=0)
    entry = tk.Entry(window)
    entry.grid(row=input_field_names.index(name), column=1)
    input_fields.append(entry)

def predict():
    try:
        data = [float(field.get()) if field.get() else np.nan for field in input_fields]
        if np.nan not in data:
            data = np.array(data).reshape(1, -1)
            data_scaled = scaler.transform(data)
            prediction = model.predict(data_scaled)
            message = "Cancer: {}".format("Benign" if prediction[0] == 1 else "Malignant")
            messagebox.showinfo("Prediction", message)
        else:
            messagebox.showerror("Error", "Please fill in all the input fields with numeric values")
    except ValueError:
        messagebox.showerror("Error", "Invalid input. Please fill in all the input fields with numeric values")


tk.Button(window, text="Predict", command=predict).grid(row=len(input_field_names), columnspan=2)

# Run the main loop
window.mainloop()