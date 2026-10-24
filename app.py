import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pickle
import numpy as np

# Load assets
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('country_lookup.pkl', 'rb') as f:
        lookup_database = pickle.load(f)
except FileNotFoundError:
    print("Error: Missing pipeline binaries. Run 'train_model.py' first.")
    exit()

def toggle_input_mode():
    """Switches the UI visibility depending on selected option."""
    mode = mode_selection.get()
    if mode == "country":
        # Show country elements
        frame_country.pack(pady=10, fill="x")
        # Hide custom features elements
        frame_features.pack_forget()
    else:
        # Hide country elements
        frame_country.pack_forget()
        # Show custom features elements
        frame_features.pack(pady=10)

def predict_status():
    mode = mode_selection.get()
    ordered_features = []

    if mode == "country":
        user_input = entry_country.get().strip().lower()
        if not user_input:
            messagebox.showwarning("Input Error", "Please provide a valid country name.")
            return
        
        if user_input in lookup_database:
            f_dict = lookup_database[user_input]
            ordered_features = [
                f_dict['GDP_pc_2023'], f_dict['HDI_Value'], f_dict['Life_Expectancy'],
                f_dict['Expected_Schooling'], f_dict['Mean_Schooling'], f_dict['GNI_per_Capita']
            ]
        else:
            label_result.config(text="Status: Country Not Found in Records", fg="#c62828")
            return
            
    elif mode == "features":
        try:
            # Parse raw user metrics entries from the text boxes
            ordered_features = [
                float(entry_gdp.get().replace(',', '')),
                float(entry_hdi.get()),
                float(entry_life.get()),
                float(entry_exp_school.get()),
                float(entry_mean_school.get()),
                float(entry_gni.get().replace(',', ''))
            ]
        except ValueError:
            messagebox.showerror("Format Error", "All metrics must be valid numbers.")
            return

    # Run core prediction loop using the extracted vector array
    input_array = np.array([ordered_features])
    prediction = model.predict(input_array)
    
    if prediction[0] == 1:
        label_result.config(text="Prediction: DEVELOPED STATUS", fg="#2e7d32")
    else:
        label_result.config(text="Prediction: DEVELOPING STATUS", fg="#e65100")


# --- UI Architecture Layout Setup ---
window = tk.Tk()
window.title("Universal Development Classifier")
window.geometry("500x520")
window.configure(bg="#f8f9fa")

# Main Title Header
tk.Label(window, text="Development Predictor Portal", font=("Helvetica", 16, "bold"), bg="#f8f9fa", fg="#212529").pack(pady=15)

# Selection Mode Switch Control
tk.Label(window, text="Choose Input Methodology:", font=("Helvetica", 10, "bold"), bg="#f8f9fa", fg="#495057").pack()
mode_selection = tk.StringVar(value="country")

radio_frame = tk.Frame(window, bg="#f8f9fa")
radio_frame.pack(pady=5)
tk.Radiobutton(radio_frame, text="By Country Name Only", variable=mode_selection, value="country", command=toggle_input_mode, bg="#f8f9fa", font=("Helvetica", 10)).pack(side="left", padx=15)
tk.Radiobutton(radio_frame, text="By Custom Metric Parameters", variable=mode_selection, value="features", command=toggle_input_mode, bg="#f8f9fa", font=("Helvetica", 10)).pack(side="left", padx=15)

# --- Frame A: Country Search Name Input ---
frame_country = tk.Frame(window, bg="#f8f9fa")
tk.Label(frame_country, text="Enter Official Country Name:", font=("Helvetica", 10), bg="#f8f9fa").pack()
entry_country = tk.Entry(frame_country, font=("Helvetica", 11), width=35, justify="center")
entry_country.pack(pady=5)
frame_country.pack(pady=10, fill="x") # Initial state visible

# --- Frame B: Manual Grid Custom Feature Entry ---
frame_features = tk.Frame(window, bg="#f8f9fa")

labels_text = [
    "GDP per Capita (USD 2023):", "HDI Value Score (0.0 to 1.0):", 
    "Life Expectancy at Birth (Years):", "Expected Schooling (Years):", 
    "Mean Schooling (Years):", "GNI per Capita (PPP $):"
]

entry_gdp = tk.Entry(frame_features, font=("Helvetica", 10), width=20)
entry_hdi = tk.Entry(frame_features, font=("Helvetica", 10), width=20)
entry_life = tk.Entry(frame_features, font=("Helvetica", 10), width=20)
entry_exp_school = tk.Entry(frame_features, font=("Helvetica", 10), width=20)
entry_mean_school = tk.Entry(frame_features, font=("Helvetica", 10), width=20)
entry_gni = tk.Entry(frame_features, font=("Helvetica", 10), width=20)

entries = [entry_gdp, entry_hdi, entry_life, entry_exp_school, entry_mean_school, entry_gni]

for i in range(len(labels_text)):
    tk.Label(frame_features, text=labels_text[i], font=("Helvetica", 10), anchor="w", bg="#f8f9fa").grid(row=i, column=0, sticky="w", pady=4, padx=10)
    entries[i].grid(row=i, column=1, pady=4, padx=10)

# Submit Predict Button
btn_submit = tk.Button(window, text="Calculate Prediction Status", bg="#0d6efd", fg="white", font=("Helvetica", 11, "bold"), padx=15, pady=6, command=predict_status)
btn_submit.pack(pady=15)

# Output Results Interface Screen
label_result = tk.Label(window, text="Prediction: Awaiting Calculation Submission", font=("Helvetica", 12, "bold"), bg="#f8f9fa", fg="#6c757d")
label_result.pack(pady=15)

window.mainloop()