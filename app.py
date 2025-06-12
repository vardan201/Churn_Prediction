import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load original dataset and model
df_1 = pd.read_csv("first_telc.csv")
model = pickle.load(open("model.sav", "rb"))

# Clean and prepare original dataset
df_1['tenure'] = pd.to_numeric(df_1['tenure'], errors='coerce').fillna(0).astype(int)
labels = [f"{i} - {i + 11}" for i in range(1, 72, 12)]
df_1['tenure_group'] = pd.cut(df_1['tenure'], range(1, 80, 12), right=False, labels=labels)
df_1.drop(columns=['tenure'], inplace=True)

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    # Input
    input_data = [request.form[f'query{i}'] for i in range(1, 20)]

    new_df = pd.DataFrame([input_data], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ])

    # Preprocess new input
    new_df['tenure'] = pd.to_numeric(new_df['tenure'], errors='coerce').fillna(0).astype(int)
    new_df['tenure_group'] = pd.cut(new_df['tenure'], range(1, 80, 12), right=False, labels=labels)
    new_df.drop(columns=['tenure'], inplace=True)

    # Combine with full data
    df_combined = pd.concat([df_1, new_df], ignore_index=True)

    # One-hot encode
    cat_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
    ]
    df_dummies = pd.get_dummies(df_combined[cat_cols])

    # Extract only new input row
    input_features = df_dummies.tail(1)

    # Align input with model's expected features
    if hasattr(model, 'feature_names_in_'):
        expected_cols = model.feature_names_in_
    else:
        expected_cols = input_features.columns  # fallback

    # Add any missing columns
    for col in expected_cols:
        if col not in input_features.columns:
            input_features[col] = 0

    # Reorder columns
    input_features = input_features[expected_cols]

    # Predict
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    output1 = "This customer is likely to be churned!!" if prediction == 1 else "This customer is likely to continue!!"
    output2 = f"Confidence: {probability * 100:.2f}%"

    return render_template('home.html',
                           output1=output1,
                           output2=output2,
                           **{f"query{i}": request.form[f"query{i}"] for i in range(1, 20)})

if __name__ == "__main__":
    app.run(debug=True)
