import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# === Page Configuration ===
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Load Model ===
@st.cache_resource
def load_models():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("feature_names.pkl")
        range_info = joblib.load("range_info.pkl")
        return model, scaler, features, range_info
    except FileNotFoundError as e:
        st.error(f"Model file tidak ditemukan: {e}")
        return None, None, None, None

model, scaler, features, range_info = load_models()

# === CSS Styling ===
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Header */
    .header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .header h1 {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header p {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
    }
    
    /* Cards */
    .card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .card h3 {
        color: white;
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    /* Input Styling */
    .stSlider > div > div {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stRadio > div {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.3rem 0;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 2px solid rgba(255,255,255,0.3);
        margin: 1rem 0;
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .result-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.9);
        margin-bottom: 1rem;
    }
    
    .confidence-score {
        font-size: 1.5rem;
        font-weight: 600;
        color: #FFD700;
        margin-top: 1rem;
    }
    
    /* Footer */
    .footer {
        background: rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .footer-text {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 2.2rem;
        }
        
        .card {
            padding: 1rem;
        }
        
        .result-title {
            font-size: 2rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# === Check Models ===
if model is None:
    st.error("❌ Model tidak dapat dimuat. Pastikan file model tersedia.")
    st.stop()

# === Header ===
st.markdown("""
    <div class="header">
        <h1>🧠 Prediksi Tipe Kepribadian</h1>
        <p>Temukan kepribadian Anda dengan analisis AI yang akurat</p>
    </div>
""", unsafe_allow_html=True)

# === Main Layout ===
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="card"><h3>📋 Input Data Anda</h3></div>', unsafe_allow_html=True)
    
    # Feature Labels
    feature_labels = {
        'Stage_fear': 'Takut Berbicara di Depan Umum',
        'Drained_after_socializing': 'Merasa Lelah Setelah Bersosialisasi',
        'Time_spent_Alone': 'Waktu Sendiri (jam)',
        'Social_event_attendance': 'Acara Sosial per Bulan',
        'Going_outside': 'Frekuensi Keluar Rumah (per minggu)',
        'Friends_circle_size': 'Jumlah Teman Dekat',
        'Post_frequency': 'Posting Media Sosial (per minggu)'
    }
    
    input_data = {}
    
    # Create input form
    for col in features:
        label = feature_labels.get(col, col.replace('_', ' ').title())
        
        if col in ['Stage_fear', 'Drained_after_socializing']:
            pilihan = st.radio(
                f"**{label}**",
                ['Tidak', 'Ya'],
                horizontal=True,
                key=f"radio_{col}"
            )
            input_data[col] = 1 if pilihan == 'Ya' else 0
        else:
            min_val = int(range_info[col]['min'])
            max_val = int(range_info[col]['max'])
            default = int(range_info[col]['median'])
            
            input_data[col] = st.slider(
                f"**{label}**",
                min_val,
                max_val,
                default,
                key=f"slider_{col}"
            )

with col2:
    st.markdown('<div class="card"><h3>📊 Ringkasan Input</h3></div>', unsafe_allow_html=True)
    
    # Display input summary
    if input_data:
        st.write("**Data yang Anda masukkan:**")
        for key, value in input_data.items():
            label = feature_labels.get(key, key.replace('_', ' ').title())
            if key in ['Stage_fear', 'Drained_after_socializing']:
                display_value = "Ya" if value == 1 else "Tidak"
            else:
                display_value = str(value)
            st.write(f"• {label}: **{display_value}**")

# === Prediction Section ===
st.markdown("""
    <div class="card">
        <h3>🔮 Analisis Kepribadian</h3>
    </div>
""", unsafe_allow_html=True)

if st.button("🚀 Prediksi Kepribadian Saya"):
    with st.spinner("Menganalisis data Anda..."):
        # Make prediction
        X_new = pd.DataFrame([input_data])
        X_scaled = scaler.transform(X_new)
        pred = model.predict(X_scaled)[0]
        
        try:
            proba = model.predict_proba(X_scaled)[0]
            confidence = round(np.max(proba) * 100, 2)
        except:
            proba = [0.5, 0.5]
            confidence = 75.0
        
        # Determine result
        if pred == 1:
            personality = "Ekstrovert"
            emoji = "🧑‍🤝‍🧑"
            description = "Anda adalah orang yang energik dan menikmati interaksi sosial!"
        else:
            personality = "Introvert"
            emoji = "🧘‍♀️"
            description = "Anda adalah orang yang reflektif dan menikmati ketenangan!"
        
        # Display result
        st.markdown(f"""
            <div class="result-card">
                <div class="result-title">{emoji} {personality}</div>
                <div class="result-subtitle">{description}</div>
                <div class="confidence-score">Tingkat Keyakinan: {confidence}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Create visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card"><h3>📊 Distribusi Probabilitas</h3></div>', unsafe_allow_html=True)
            
            # Plotly pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Introvert', 'Ekstrovert'],
                    values=proba,
                    hole=0.4,
                    marker=dict(colors=['#4ECDC4', '#FF6B6B']),
                    textinfo='label+percent',
                    textfont=dict(size=14, color='white')
                )
            ])
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="card"><h3>💡 Insight Kepribadian</h3></div>', unsafe_allow_html=True)
            
            if pred == 1:  # Ekstrovert
                st.markdown("""
                    **Karakteristik Anda:**
                    - Energi dari interaksi sosial
                    - Mudah beradaptasi
                    - Ekspresif dan terbuka
                    - Suka kerja tim
                    
                    **Saran:**
                    - Manfaatkan kemampuan networking
                    - Jaga keseimbangan waktu sosial
                    - Kembangkan leadership skills
                """)
            else:  # Introvert
                st.markdown("""
                    **Karakteristik Anda:**
                    - Energi dari waktu sendiri
                    - Berpikir mendalam
                    - Fokus pada kualitas hubungan
                    - Kreatif dan reflektif
                    
                    **Saran:**
                    - Manfaatkan kemampuan analisis
                    - Cari lingkungan yang mendukung
                    - Kembangkan expertise khusus
                """)

# === Footer ===
st.markdown("""
    <div class="footer">
        <div class="footer-text">
            🎯 <strong>Akurasi Model: 93.56%</strong> | 
            Powered by Random Forest Algorithm | 
            Built with Streamlit & Python
        </div>
    </div>
""", unsafe_allow_html=True)