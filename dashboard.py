
import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
import time
import psutil
import os
from datetime import datetime

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Smart Doorbell Edge AI",
    page_icon="🔋",
    layout="wide"
)

# LOAD ONNX MODEL
@st.cache_resource
def load_model():
    return ort.InferenceSession(
        "battery_model.onnx",
        providers=["CPUExecutionProvider"]
    )

session = load_model()
input_name = session.get_inputs()[0].name

# SESSION STATE FOR PREDICTION HISTORY
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# DASHBOARD HEADER
st.title("🔋 Smart Doorbell Battery Intelligence")
st.caption("Software-Based Edge AI Battery-Life Prediction System")
st.divider()

# SENSOR INPUTS
st.subheader("Doorbell Sensor Readings")

col1, col2, col3 = st.columns(3)

with col1:
    battery_v = st.slider(
        "Battery Voltage (V)", 3.35, 4.20, 3.80, 0.01
    )
    temp_c = st.slider(
        "Temperature (°C)", 8.0, 46.0, 28.0, 0.5
    )

with col2:
    rings = st.slider(
        "Rings Per Day", 0, 37, 8
    )
    wifi = st.slider(
        "WiFi Signal (dBm)", -88, -45, -65
    )

with col3:
    age = st.slider(
        "Battery Age (Months)", 0, 36, 12
    )

# PREDICTION
if st.button("Predict Battery Life", type="primary"):

    input_data = np.array(
        [[battery_v, temp_c, rings, wifi, age]],
        dtype=np.float32
    )

    start = time.perf_counter()

    prediction = session.run(
        None,
        {input_name: input_data}
    )[0][0][0]

    latency = (time.perf_counter() - start) * 1000
    prediction = max(float(prediction), 0)

    # SAVE PREDICTION
    st.session_state.prediction_history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Battery Life (Days)": round(prediction, 2),
        "Battery Voltage": battery_v,
        "Temperature": temp_c,
        "Rings Per Day": rings,
        "WiFi Signal": wifi,
        "Battery Age": age
    })

    st.session_state.latest_prediction = prediction
    st.session_state.latest_latency = latency

# DISPLAY LATEST PREDICTION
if "latest_prediction" in st.session_state:

    prediction = st.session_state.latest_prediction
    latency = st.session_state.latest_latency

    st.divider()
    st.subheader("📊 Battery Prediction Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Estimated Battery Life",
            f"{prediction:.2f} days"
        )

    with c2:
        st.metric(
            "Inference Latency",
            f"{latency:.5f} ms"
        )

    with c3:
        model_size = os.path.getsize("battery_model.onnx") / 1024
        st.metric(
            "ONNX Model Size",
            f"{model_size:.2f} KB"
        )

    # BATTERY STATUS
    st.subheader("🔋 Battery Remaining-Life Status")

    if prediction < 15:
        st.error("CRITICAL: Battery needs immediate attention.")
    elif prediction < 30:
        st.warning("ATTENTION: Battery replacement should be planned.")
    else:
        st.success("HEALTHY: Estimated remaining battery life is good.")

    # SENSOR SUMMARY
    st.subheader("Sensor Input Summary")

    sensor_data = pd.DataFrame({
        "Sensor": [
            "Battery Voltage",
            "Temperature",
            "Rings Per Day",
            "WiFi Signal",
            "Battery Age"
        ],
        "Reading": [
            f"{battery_v} V",
            f"{temp_c} °C",
            rings,
            f"{wifi} dBm",
            f"{age} months"
        ]
    })

    st.table(sensor_data)

# PREDICTION HISTORY
st.divider()
st.subheader("📈 Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.line_chart(
        history_df.set_index("Time")["Battery Life (Days)"]
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []
        if "latest_prediction" in st.session_state:
            del st.session_state.latest_prediction
        st.rerun()

else:
    st.info("No predictions recorded yet. Enter sensor readings and click Predict Battery Life.")

# MODEL PERFORMANCE COMPARISON
st.divider()
st.subheader("🤖 Model Performance Comparison")

st.caption(
    "Test-set Mean Absolute Error (MAE). "
    "Lower MAE indicates better prediction accuracy."
)

comparison = pd.DataFrame({
    "Model": [
        "Dummy Regressor",
        "Linear Regression",
        "Neural Network",
        "ONNX FP32",
        "ONNX INT8"
    ],
    "MAE (Days)": [
        51.297,
        22.629,
        6.669,
        6.669,
        6.716
    ]
})

st.bar_chart(
    comparison.set_index("Model")
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

# EDGE RUNTIME INFORMATION
st.divider()
st.subheader("⚙️ Edge Runtime Information")

col1, col2 = st.columns(2)

with col1:
    st.write("Inference Engine: ONNX Runtime")
    st.write("Execution Provider: CPUExecutionProvider")
    st.write("Model Format: ONNX FP32")

with col2:
    st.write("CPU Usage:", f"{psutil.cpu_percent(interval=0.1)}%")
    st.write("Memory Usage:", f"{psutil.virtual_memory().percent}%")

st.caption(
    "Smart Doorbell Battery Prediction | Software-Based Edge AI"
)