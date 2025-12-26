import streamlit as st
import numpy as np

# ===== Load model info =====
coef = np.load("coef.npy")
intercept = float(np.load("intercept.npy"))
means = np.load("mean.npy")
stds = np.load("std.npy")
ocean_mapping = {
    "<1H OCEAN": 0,
    "INLAND": 1,
    "NEAR OCEAN": 2,
    "NEAR BAY": 3,
    "ISLAND": 4
}

# ===== Modern CSS =====
st.markdown("""
    <style>
    .main > div {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    h1 {
        font-weight: 700;
        color: #1e40af;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 12px;
        height: 3.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
    }
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9;
    }
    .css-1v3fvbn {  /* input labels */
        font-weight: 600;
        color: #374151;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 2rem;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #4b5563;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1e40af;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== Page config =====
st.set_page_config(
    page_title="Dự đoán giá nhà California",
    page_icon="🏡",
    layout="wide"
)

# ===== Header =====
st.title("🏡 Dự đoán giá nhà California")
st.markdown("*Mô hình Linear Regression được huấn luyện trên California Housing Dataset*")

st.divider()

# ===== Layout =====
left, right = st.columns([1.2, 1], gap="large")

# ===== INPUT =====
with left:
    st.subheader("📥 Nhập thông tin")

    with st.container(border=True):
        st.markdown("**📍 Vị trí địa lý**")
        col_lon, col_lat = st.columns(2)
        with col_lon:
            longitude = st.number_input("Kinh độ (Longitude)", -125.0, -113.0, -118.24, step=0.01)
        with col_lat:
            latitude = st.number_input("Vĩ độ (Latitude)", 32.5, 42.0, 37.77, step=0.01)

        ocean_label = st.selectbox("Khoảng cách đến biển (Ocean Proximity)", list(ocean_mapping.keys()))
        ocean_proximity = ocean_mapping[ocean_label]

    with st.container(border=True):
        st.markdown("**🏠 Thông tin nhà**")
        col1, col2 = st.columns(2)
        with col1:
            housing_median_age = st.slider("Tuổi trung bình nhà (năm)", 1, 52, 29)
        with col2:
            total_rooms = st.number_input("Tổng số phòng", 100, 40000, 2635)

        col3, col4 = st.columns(2)
        with col3:
            total_bedrooms = st.number_input("Tổng số phòng ngủ", 10, 6500, 537)

    with st.container(border=True):
        st.markdown("**👥 Dân số & thu nhập**")
        col5, col6 = st.columns(2)
        with col5:
            population = st.number_input("Dân số khu vực", 100, 36000, 1425)
        with col6:
            households = st.number_input("Số hộ gia đình", 50, 6000, 499)

        median_income = st.slider("Thu nhập trung bình (x10k USD)", 0.5, 15.0001, 3.87, step=0.01)

# ===== RESULT =====
with right:
    st.subheader("💰 Kết quả dự đoán")

    st.markdown("Nhập đầy đủ thông tin bên trái và nhấn nút để xem kết quả.")

    input_data = [
        longitude, latitude,
        housing_median_age, total_rooms, total_bedrooms,
        population, households, median_income,
        ocean_proximity
    ]

    if st.button("🔮 Dự đoán giá nhà", type="primary", use_container_width=True):
        x = np.array(input_data)
        x_scaled = (x - means) / stds
        prediction = np.dot(x_scaled, coef) + intercept

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Giá nhà ước tính (Median House Value)</div>
                <div class="metric-value">${prediction:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success("Dự đoán hoàn tất!")
        st.info("Giá trị là trung bình của khu vực (block group), đơn vị USD.")

st.divider()
st.caption("Dự án học thuật • Spark MLlib • Linear Regression • Streamlit UI")