# inference_gui.py
import streamlit as st
import torch
import numpy as np
from model import Net

# ----------------------------
# Load the best-performing model (seed 12)
# ----------------------------
BEST_SEED = 12
CHECKPOINT_PATH = f'checkpoints/model_seed{BEST_SEED}.pt'

model = Net()
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

st.title("Covertype Class Prediction")
st.write("""
This app predicts the **Covertype class** based on manual input of 54 features.
Enter values for each feature and press **Predict**.
""")

# ----------------------------
# Input features
# ----------------------------
features = []
for i in range(54):
    value = st.number_input(f"Feature {i+1}", value=0.0, step=0.01, format="%.4f")
    features.append(value)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    st.subheader("Predicted Class Probabilities:")
    for i, p in enumerate(probs):
        st.write(f"Class {i+1}: {p*100:.2f}%")
    
    predicted_class = np.argmax(probs) + 1
    st.success(f"✅ Predicted Class: {predicted_class}")
