
import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load scikit-learn models
from sklearn.preprocessing import StandardScaler, LabelEncoder # To handle preprocessing
import os
from imblearn.pipeline import Pipeline as ImbPipeline # Import ImbPipeline
from imblearn.over_sampling import SMOTE # Import SMOTE for dummy pipeline fallback

# --- Configuration and Setup ---
st.set_page_config(page_title="Insecticide Resistance Predictor", layout="wide")

# Define the path where models and data might be saved
save_path = "/content/drive/MyDrive/Manuscript Doc"

# Define selected features (must match the features used for training)
biochem_features = ["alpha", "beta", "gst", "mfo", "ache"]
genotype_features = ["kdr-w", "ace-1"]
env_features = ["temperature", "humidity"]
selected_features = biochem_features + genotype_features + env_features

# --- Load Models ---
@st.cache_resource # Cache the model loading for efficiency
def load_all_models():
    models = {}
    try:
        # Load the best XGBoost pipeline (which includes scaler and SMOTE if used)
        models['XGBoost_Pipeline'] = joblib.load(os.path.join(save_path, 'xgboost_pipeline.pkl'))
        st.success("XGBoost Pipeline loaded successfully!")
    except Exception as e:
        st.error(f"Error loading XGBoost Pipeline: {e}. Please ensure 'xgboost_pipeline.pkl' exists in the specified path.")
        # Fallback for demonstration: initialize a dummy model if loading fails
        import xgboost as xgb
        dummy_pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('xgb', xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1))
        ])
        # Fit with dummy data (ensure feature count matches)
        dummy_pipeline.fit(np.random.rand(10, len(selected_features)), np.random.randint(0, 2, 10))
        models['XGBoost_Pipeline'] = dummy_pipeline
        st.warning("Using dummy XGBoost pipeline for prediction.")
    return models

models = load_all_models()

# --- Load Preprocessors (for direct access/display if needed, though pipeline handles most) ---
@st.cache_resource
def load_preprocessors():
    scaler = None
    le_kdrw = None
    le_ace1 = None
    try:
        scaler = joblib.load(os.path.join(save_path, 'scaler.pkl'))
        le_kdrw = joblib.load(os.path.join(save_path, 'le_kdrw.pkl'))
        le_ace1 = joblib.load(os.path.join(save_path, 'le_ace1.pkl'))
        st.success("Preprocessors loaded successfully!")
    except Exception as e:
        st.error(f"Error loading preprocessors: {e}. Please ensure .pkl files exist in the specified path.")
        # Fallback for demonstration
        scaler = StandardScaler()
        scaler.fit(np.random.rand(100, len(biochem_features + env_features))) # Fit with dummy data
        le_kdrw = LabelEncoder()
        le_kdrw.fit(['SS', 'RS', 'RR', 'None']) # Fit with dummy labels
        le_ace1 = LabelEncoder()
        le_ace1.fit(['SS', 'RS', 'RR', 'None']) # Fit with dummy labels
        st.warning("Using dummy preprocessors.")
    return scaler, le_kdrw, le_ace1

scaler, le_kdrw, le_ace1 = load_preprocessors()

# --- Streamlit App Layout ---
st.title("Gyimah's Insecticide Resistance Prediction Dashboard")
st.markdown("Enter feature values or upload a file to predict insecticide resistance (Susceptible/Resistant).")

# Model Selection
selected_model_name = st.sidebar.selectbox(
    "Select Prediction Model",
    list(models.keys())
)
model_to_use = models[selected_model_name]

# --- User Input Features Section ---
st.sidebar.header("Input Feature Values")

input_data = {}

# Biochemical features
st.sidebar.subheader("Biochemical Markers")
for feature in biochem_features:
    input_data[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", value=0.0, format="%.4f", key=f"single_{feature}")

# Genotype features
st.sidebar.subheader("Genotype Markers")
input_data['kdr-w'] = st.sidebar.selectbox("Select Kdr-w Genotype", options=le_kdrw.classes_, index=int(np.where(le_kdrw.classes_ == 'SS')[0][0]) if 'SS' in le_kdrw.classes_ else 0, key="single_kdr-w")
input_data['ace-1'] = st.sidebar.selectbox("Select Ace-1 Genotype", options=le_ace1.classes_, index=int(np.where(le_ace1.classes_ == 'SS')[0][0]) if 'SS' in le_ace1.classes_ else 0, key="single_ace-1")

# Environmental features
st.sidebar.subheader("Environmental Factors")
for feature in env_features:
    input_data[feature] = st.sidebar.number_input(f"Enter {feature.replace('_', ' ').title()}", value=0.0, format="%.4f", key=f"single_{feature}")

# --- Prediction for Single Entry ---
if st.sidebar.button("Predict Resistance"):
    # Create DataFrame from inputs
    input_df = pd.DataFrame([input_data])

    # Preprocess inputs using the loaded pipeline
    # The ImbPipeline (scaler, smote, xgb) automatically handles scaling and encoding (if done via ColumnTransformer within pipeline)
    # However, since LabelEncoding for genotype features was done *before* the pipeline, it needs to be applied here.
    # The scaler within the pipeline will handle the continuous features.

    # Encode categorical features for the input_df using the loaded LabelEncoders
    for col in genotype_features:
        if col in input_df.columns:
            le = le_kdrw if col == 'kdr-w' else le_ace1
            input_df[col] = le.transform(input_df[col].astype(str))

    # The input_df must have columns in the same order as X_train_encoded for consistent prediction by the pipeline.
    input_df_processed = input_df[selected_features]

    st.subheader(f"Prediction using {selected_model_name} (Single Entry):")

    # Predict using the entire pipeline
    prediction = model_to_use.predict(input_df_processed)[0]
    prediction_proba = model_to_use.predict_proba(input_df_processed)[0]
    st.write(f"Predicted Class: **{'Resistant' if prediction == 1 else 'Susceptible'}**")
    st.write(f"Resistance Probability: {prediction_proba[1]*100:.2f}%")

    st.markdown(" preconceived_label_encoder.transform(input_df[col]) ")
    st.subheader("Input Data Summary")
    st.dataframe(input_df_processed)


# --- File Upload Section ---
st.header("Upload File for Batch Prediction")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            batch_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            batch_df = pd.read_excel(uploaded_file)

        st.subheader("Uploaded Data Preview:")
        st.dataframe(batch_df.head())

        if st.button("Run Batch Prediction"):
            # Ensure all selected_features are in the uploaded dataframe
            missing_cols = [col for col in selected_features if col not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {', '.join(missing_cols)}. Please upload a file with all required features.")
            else:
                # Select only the relevant features for prediction
                batch_df_selected = batch_df[selected_features].copy()

                # Preprocess batch data - Encode categorical features
                for col in genotype_features:
                    if col in batch_df_selected.columns:
                        le = le_kdrw if col == 'kdr-w' else le_ace1
                        batch_df_selected[col] = batch_df_selected[col].astype(str).fillna('None')
                        # Handle unknown categories by making them 'None' before transform if 'None' was fitted
                        # Or, a more robust way is to use a custom function to map unknowns to a default/raise error
                        unknown_categories = set(batch_df_selected[col].unique()) - set(le.classes_)
                        if unknown_categories:
                            st.warning(f"Unknown categories found in {col}: {', '.join(unknown_categories)}. These will be treated as the closest existing category (or handled by 'None' if previously fitted).")
                        batch_df_selected[col] = le.transform(batch_df_selected[col])

                # The pipeline will handle scaling and prediction
                batch_predictions = model_to_use.predict(batch_df_selected)
                batch_probabilities = model_to_use.predict_proba(batch_df_selected)[:, 1] # Probability of resistant (class 1)

                # Add predictions to the original dataframe for display
                batch_df['Predicted_Resistance'] = np.where(batch_predictions == 1, 'Resistant', 'Susceptible')
                batch_df['Resistance_Probability'] = batch_probabilities * 100

                st.subheader(f"Batch Predictions using {selected_model_name}:")
                st.dataframe(batch_df[['Predicted_Resistance', 'Resistance_Probability']])
                st.download_button(
                    label="Download Predictions",
                    data=batch_df.to_csv(index=False).encode('utf-8'),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

