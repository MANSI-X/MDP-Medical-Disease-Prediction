# 🩺 GlucoGuide — Diabetes Classification Demo

A beginner-friendly Streamlit website that demonstrates binary diabetes classification using a Random Forest machine-learning model.

> **Important medical notice:** This is an academic demonstration, not a medical device. It is not clinically validated and must not be used to diagnose, rule out, treat, or make medical decisions about diabetes. A negative result does not rule out diabetes. Consult a qualified healthcare professional and use appropriate laboratory tests.

## What's included

- Attractive, responsive Streamlit interface with a patient form and three tabs.
- Random Forest classifier with median imputation.
- Stratified 80/20 holdout evaluation.
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and 5-fold cross-validation.
- Optional CSV upload to use another compatible dataset.
- Clear descriptions and help text for each input.

## Can this guarantee 95% accuracy?

**No.** Accuracy cannot honestly be guaranteed by changing code or increasing the number of trees. The original project uses a relatively small diabetes dataset. Its performance may vary by split and may not generalize to other populations. This version reports measured test and cross-validation scores instead of hard-coding or promising 95%.

To make a credible claim near 95%, you would need a much larger, representative, independently collected dataset, clear outcome definitions, leakage checks, external validation, and clinical review. Even then, accuracy alone is not enough: sensitivity, specificity, calibration, and confidence intervals matter.

## Project structure

```text
GlucoGuide/
├── app.py
├── README.md
└── requirements.txt
```

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.9 or newer
- Internet connection for the default sample dataset (or upload a local CSV)

## Run on Windows using VS Code

1. Download/extract the project folder.
2. Open the folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

5. Activate it:

   ```powershell
   .venv\Scripts\activate
   ```

6. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Start the website:

   ```powershell
   python -m streamlit run app.py
   ```

8. Open the local URL shown in the terminal (usually `http://localhost:8501`).

To stop the website, press `Ctrl + C` in the terminal.

## Dataset format

By default, the app reads the sample CSV from:

`https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv`

You can instead upload a CSV from the sidebar. It must include these columns:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (0 or 1)

For this dataset convention, zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are treated as missing and imputed with the training-set median. Check this assumption before using a different dataset.

## Technologies

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn

## Troubleshooting

**`streamlit` is not recognized**

Run the app using:

```powershell
python -m streamlit run app.py
```

**Dataset could not be loaded**

Check your internet connection or upload a compatible CSV using the sidebar.

**CSV columns error**

Make sure the uploaded CSV uses the exact column names listed above.

## Academic use

This project is suitable for learning Streamlit, preprocessing, Random Forest classification, and model evaluation. Do not present its output as a clinically accurate diabetes diagnosis or as a guaranteed 95% model.
