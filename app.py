import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Concrete Strength AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/concrete_model.pkl")
scaler = joblib.load("models/concrete_scaler.pkl")


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ============================================================
# DARK RED UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(150, 0, 0, 0.45),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(255, 30, 30, 0.18),
            transparent 25%
        ),
        linear-gradient(
            135deg,
            #020000,
            #180000,
            #070000,
            #010101
        );
}


/* Page width */

.block-container {
    max-width: 1150px;
    padding-top: 35px;
}


/* Title */

.title {
    text-align: center;
    color: #ff3030;
    font-size: 46px;
    font-weight: 900;
    text-shadow:
        0 0 10px rgba(255,0,0,0.8),
        0 0 30px rgba(180,0,0,0.5);
}


/* Subtitle */

.subtitle {
    text-align: center;
    color: #dddddd;
    font-size: 18px;
}


/* Version */

.version {
    text-align: center;
    color: #998080;
    font-size: 13px;
}


/* Card */

.card {
    background: rgba(20, 5, 5, 0.88);
    border: 1px solid rgba(255, 60, 60, 0.35);
    border-radius: 22px;
    padding: 30px;
    margin-top: 30px;
    box-shadow:
        0 20px 50px rgba(0,0,0,0.7),
        inset 0 1px 1px rgba(255,255,255,0.08);
}


/* Card heading */

.card-heading {
    color: #ff4141;
    font-size: 25px;
    font-weight: 800;
}


/* Metric */

.metric {
    background: rgba(20, 5, 5, 0.90);
    border: 1px solid rgba(255,60,60,0.35);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    box-shadow:
        0 12px 30px rgba(0,0,0,0.6);
}


/* Metric value */

.metric-number {
    color: #ff3838;
    font-size: 28px;
    font-weight: 900;
}


/* Metric name */

.metric-name {
    color: #bdaaaa;
    font-size: 14px;
}


/* Result */

.result {
    background: rgba(80,0,0,0.35);
    border: 1px solid #ff3030;
    border-radius: 22px;
    padding: 35px;
    text-align: center;
    box-shadow:
        0 20px 50px rgba(0,0,0,0.7);
}


.result-label {
    color: #dddddd;
    font-size: 19px;
}


.result-number {
    color: #ff3030;
    font-size: 48px;
    font-weight: 900;
    text-shadow:
        0 0 20px rgba(255,0,0,0.6);
}


/* Buttons */

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 13px;
    border: 1px solid #ff5555;
    background:
        linear-gradient(
            135deg,
            #8b0000,
            #d00000,
            #700000
        );
    color: white;
    font-weight: 800;
    font-size: 16px;
    box-shadow:
        0 10px 25px rgba(120,0,0,0.5);
}


.stButton > button:hover {
    background:
        linear-gradient(
            135deg,
            #b00000,
            #ff0000,
            #850000
        );
}


/* Number input */

div[data-baseweb="input"] {
    background: #100606 !important;
    border: 1px solid #633333 !important;
    border-radius: 10px !important;
}


div[data-baseweb="input"] input {
    color: white !important;
}


/* Remove +/- */

input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
}


input[type=number] {
    -moz-appearance: textfield;
}


/* Footer */

.footer {
    text-align: center;
    color: #806d6d;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1
# ============================================================

if st.session_state.page == 1:

    st.markdown(
        '<div class="title">🏗️ CONCRETE STRENGTH AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Intelligent Concrete Compressive Strength Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="version">'
        'AI ENGINE • XGBoost Regression • Version 1.0.0'
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-heading">About the System</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.write(
        "This intelligent system uses XGBoost Machine Learning "
        "to estimate the compressive strength of concrete."
    )

    st.write("")

    st.write(
        "The prediction is based on eight important concrete "
        "mixture parameters and curing age."
    )

    st.write("")

    st.write(
        "The final prediction is displayed in Megapascals (MPa)."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    st.write("")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            '<div class="metric">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-number">92.58%</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-name">R² Score</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            '<div class="metric">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-number">4.70 MPa</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-name">RMSE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            '<div class="metric">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-number">3.26 MPa</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-name">MAE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    st.write("")
    st.write("")


    # --------------------------------------------------------
    # NEXT
    # --------------------------------------------------------

    if st.button("🚀 NEXT → ENTER CONCRETE DATA"):

        st.session_state.page = 2

        st.rerun()


    st.markdown(
        '<div class="footer">'
        'Concrete Strength Prediction System • Version 1.0.0'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 2
# ============================================================

elif st.session_state.page == 2:

    st.markdown(
        '<div class="title">🧪 CONCRETE MIXTURE DATA</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter concrete composition and curing age'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="version">'
        'XGBoost Prediction Engine • Version 1.0.0'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-heading">Concrete Parameters</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        cement = st.number_input(
            "Cement (kg/m³)",
            min_value=0.0,
            value=300.0,
            step=0.1,
            format="%.1f",
            key="cement"
        )

        slag = st.number_input(
            "Blast Furnace Slag (kg/m³)",
            min_value=0.0,
            value=100.0,
            step=0.1,
            format="%.1f",
            key="slag"
        )

        fly_ash = st.number_input(
            "Fly Ash (kg/m³)",
            min_value=0.0,
            value=50.0,
            step=0.1,
            format="%.1f",
            key="flyash"
        )

        water = st.number_input(
            "Water (kg/m³)",
            min_value=0.0,
            value=180.0,
            step=0.1,
            format="%.1f",
            key="water"
        )


    with col2:

        superplasticizer = st.number_input(
            "Superplasticizer (kg/m³)",
            min_value=0.0,
            value=5.0,
            step=0.1,
            format="%.1f",
            key="superplasticizer"
        )

        coarse = st.number_input(
            "Coarse Aggregate (kg/m³)",
            min_value=0.0,
            value=950.0,
            step=0.1,
            format="%.1f",
            key="coarse"
        )

        fine = st.number_input(
            "Fine Aggregate (kg/m³)",
            min_value=0.0,
            value=750.0,
            step=0.1,
            format="%.1f",
            key="fine"
        )

        age = st.number_input(
            "Age (days)",
            min_value=1,
            value=28,
            step=1,
            key="age"
        )


    st.write("")


    col1, col2 = st.columns(2)


    with col1:

        if st.button("⬅ PREVIOUS"):

            st.session_state.page = 1

            st.rerun()


    with col2:

        if st.button("PREDICT STRENGTH ➜"):

            input_data = pd.DataFrame(
                [[
                    cement,
                    slag,
                    fly_ash,
                    water,
                    superplasticizer,
                    coarse,
                    fine,
                    age
                ]],
                columns=[
                    "Cement",
                    "Blast Furnace Slag",
                    "Fly Ash",
                    "Water",
                    "Superplasticizer",
                    "Coarse Aggregate",
                    "Fine Aggregate",
                    "Age"
                ]
            )

            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)[0]

            st.session_state.prediction = float(prediction)

            st.session_state.page = 3

            st.rerun()


# ============================================================
# PAGE 3
# ============================================================

elif st.session_state.page == 3:

    prediction = st.session_state.prediction


    st.markdown(
        '<div class="title">📊 PREDICTION RESULT</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-based concrete compressive strength prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="version">'
        'XGBoost Prediction Engine • Version 1.0.0'
        '</div>',
        unsafe_allow_html=True
    )


    # RESULT

    st.markdown(
        '<div class="result">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-label">'
        'Predicted Concrete Compressive Strength'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="result-number">'
        f'{prediction:.2f} MPa'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    st.write("")


    # MODEL INFORMATION

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            '<div class="metric">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-number">XGBoost</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-name">Selected Model</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            '<div class="metric">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-number">92.58%</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-name">R² Score</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            '<div class="metric">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-number">4.70 MPa</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="metric-name">RMSE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    st.write("")


    # SUMMARY

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-heading">Prediction Summary</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.write(
        f"The XGBoost model predicts a concrete compressive "
        f"strength of {prediction:.2f} MPa."
    )

    st.write(
        "The prediction is based on the concrete mixture "
        "composition and curing age provided by the user."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    st.write("")


    # NAVIGATION

    col1, col2 = st.columns(2)


    with col1:

        if st.button("⬅ MODIFY INPUT"):

            st.session_state.page = 2

            st.rerun()


    with col2:

        if st.button("🏠 HOME"):

            st.session_state.page = 1

            st.rerun()


    st.markdown(
        '<div class="footer">'
        'Concrete Strength Prediction System • Version 1.0.0'
        '</div>',
        unsafe_allow_html=True
    )