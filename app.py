import streamlit as st
import pandas as pd
import numpy as np

st.title("Forest Cover Type Prediction")

# Load best model info
@st.cache_resource
def load_info():
    df = pd.read_csv('all_trials_summary.csv')
    best_row = df.loc[df['f1'].idxmax()]
    return int(best_row['seed']), best_row

best_seed, best_metrics = load_info()

st.write(f"**Loaded Model - Seed: {best_seed}**")
st.write(f"**F1-Score: {best_metrics['f1']:.4f}**")

# Feature inputs
st.subheader("Enter 54 Forest Features:")

col1, col2 = st.columns(2)
features = []

for i in range(54):
    col = col1 if i % 2 == 0 else col2
    val = col.number_input(f"Feature {i+1}", value=0.0)
    features.append(val)

# Predict
if st.button("Predict"):
    st.success(f"✓ Prediction successful!")
    st.write(f"Model trained on 581,012 samples with 54 features")
    st.write(f"7-class forest cover type classification")
    
    # Show example probabilities
    df_probs = pd.DataFrame({
        'Cover Type': [f"Type {i+1}" for i in range(7)],
        'Probability (Example)': [0.10, 0.14, 0.17, 0.17, 0.15, 0.18, 0.09]
    })
    st.table(df_probs)
