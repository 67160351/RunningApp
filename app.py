import streamlit as st
import pandas as pd
import numpy as np
import joblib
import math

# --- Page Configuration ---
st.set_page_config(
    page_title="Running Decision Engine",
    page_icon="🏃",
    layout="centered",
    initial_sidebar_state="expanded"
)


# --- Custom CSS for Professional Styling ---
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .safe-card { background-color: #d4edda; border-left: 8px solid #28a745; color: #155724; }
    .caution-card { background-color: #fff3cd; border-left: 8px solid #ffc107; color: #856404; }
    .extreme-card { background-color: #ffeeba; border-left: 8px solid #fd7e14; color: #856404; }
    .danger-card { background-color: #f8d7da; border-left: 8px solid #dc3545; color: #721c24; }

    .advice-title {
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


load_css()


# --- Load Artifacts ---
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('running_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, scaler, le
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ Model หรือ Scaler กรุณาตรวจสอบไฟล์ใน Directory")
        st.stop()


model, scaler, le = load_model_artifacts()


# --- Scientific Calculations ---
def calculate_heat_index_celsius(temp_c, humidity):
    """
    คำนวณ Heat Index (ดัชนีความร้อน) ด้วยสมการ Rothfusz Regression
    คืนค่าเป็นองศาเซลเซียส
    """
    if temp_c < 27 or humidity < 40:
        return temp_c

    T = temp_c * 9 / 5 + 32
    RH = humidity

    HI = (-42.379 + 2.04901523 * T + 10.14333127 * RH - .22475541 * T * RH
          - .00683783 * T * T - .05481717 * RH * RH + .00122874 * T * T * RH
          + .00085282 * T * RH * RH - .00000199 * T * T * RH * RH)

    if (RH < 13 and T >= 80 and T <= 112):
        HI -= ((13 - RH) / 4) * math.sqrt((17 - abs(T - 95)) / 17)
    elif (RH > 85 and T >= 80 and T <= 87):
        HI += ((RH - 85) / 10) * ((87 - T) / 5)

    return (HI - 32) * 5 / 9


def get_sports_science_advice(label, hi_c, pm25):
    """
    ปรับภาษาให้เป็นภาษาพูดง่ายๆ เข้าใจได้ทันทีสำหรับทุกคน
    """
    if label == "Safe":
        return {
            "header": "วันนี้อากาศดีมาก! ซ้อมหนักแค่ไหนก็ได้เลย",
            "program": "จะลงโปรแกรมวิ่งไกล วิ่งเร็ว หรือซ้อมเพื่อสุขภาพก็สบาย อย่าพึ่งหักโหมครับ",
            "nutrition": "ดื่มน้ำตามปกติ เพลินๆ ได้เลย ร่างกายพร้อมรับทุกอย่างแล้ว"
        }
    elif label == "Caution":
        return {
            "header": "อากาศหน่อยๆ หนักใจ ซ้อมเบาๆ พอแล้วกันนะ",
            "program": "แนะนำให้วิ่งเบาๆ หายใจทำได้สบายๆ อย่าฝืนซ้อมหนักหรือทำสถิติใหม่วันนี้ครับ",
            "nutrition": "จิบน้ำบ่อยๆ ป้องกันคอแห้ง และสังเกตตัวเองดีๆ ว่าเหนื่อยเกินไปไหม"
        }
    elif label == "Extreme":
        if pm25 > 75:
            return {
                "header": "ฝุ่นเยอะแบบนี้ เลี่ยงวิ่งข้างนอกดีกว่าครับ",
                "program": "ปอดจะทนไม่ไหวแน่ๆ แนะนำให้วิ่งบนลู่วิ่งในร่ม หรือออกกำลังกายแบบอื่นที่ปลอดภัยกว่า",
                "nutrition": "ดื่มน้ำเยอะๆ ช่วยล้างพิษในร่างกาย และควรใส่หน้ากากถ้าต้องออกไปข้างนอก"
            }
        elif hi_c > 32:
            return {
                "header": "อากาศร้อนจัด ระวังหน้ามืดเป็นลม",
                "program": "ซ้อมหนักๆ พักไว้ก่อนครับ ถ้าจะวิ่งก็แค่เบาๆ น้อยกว่า 45 นาที แล้วกลับบ้านเลย",
                "nutrition": "อย่าลืมดื่มน้ำเกลือแร่เพื่อชดเชยเหงื่อที่เสียไป และหาที่ร่มพักบ่อยๆ"
            }
        else:
            return {
                "header": "สภาพอากาศมีความท้าทายเหลือเกิน",
                "program": "ลดระยะทางและความเร็วลงครับ ถ้ารู้สึกผิดปกติให้หยุดทันที",
                "nutrition": "เพิ่มการดื่มน้ำให้มากกว่าปกติ"
            }
    elif label == "Danger":
        return {
            "header": "อันตรายมาก! หยุดทุกอย่างเลย อย่าออกไปวิ่งข้างนอกเด็ดขาด",
            "program": "เปลี่ยนเป็นออกกำลังกายในร่มที่แอร์เย็นๆ ดีกว่าครับ ปลอดภัยกว่าเยอะ",
            "nutrition": "พักผ่อนให้เพียงพอ รักษาสุขภาพกันก่อน วันพรุ่งนี้ค่อยวิ่งใหม่"
        }


# --- UI Layout ---
st.markdown('<div class="main-header">Running Decision Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ระบบวิเคราะห์ความเหมาะสมในการวิ่งกลางแจ้งด้วยวิทยาศาสตร์การกีฬา</div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("ข้อมูลสภาพอากาศ")
    st.caption("ปรับค่าตามสภาพอากาศจริงในพื้นที่ของคุณ")

    # ปรับปรุง UI: เปลี่ยนเป็น Slider ตาม Feedback
    temp = st.slider("อุณหภูมิ (°C)", min_value=10.0, max_value=50.0, value=30.0, step=0.5)
    hum = st.slider("ความชื้นสัมพัทธ์ (%)", min_value=0, max_value=100, value=60, step=1)
    pm25 = st.slider("ค่า PM2.5 (µg/m³)", min_value=0, max_value=500, value=25, step=1)

    analyze_btn = st.button("วิเคราะห์ความเหมาะสม", type="primary")

if analyze_btn:
    # 1. Scientific Calculations
    feels_like = calculate_heat_index_celsius(temp, hum)

    # 2. Model Prediction
    input_data = pd.DataFrame([[temp, hum, pm25]], columns=['temperature', 'humidity', 'pm25'])
    input_scaled = scaler.transform(input_data)
    prediction_idx = model.predict(input_scaled)[0]
    label = le.inverse_transform([prediction_idx])[0]

    # 3. Get Specific Advice
    advice = get_sports_science_advice(label, feels_like, pm25)

    # 4. Display Results

    # Determine CSS class
    card_class = ""
    if label == "Safe":
        card_class = "safe-card"
    elif label == "Caution":
        card_class = "caution-card"
    elif label == "Extreme":
        card_class = "extreme-card"
    elif label == "Danger":
        card_class = "danger-card"

    # Display Decision Card
    st.markdown(f"""
    <div class="result-card {card_class}">
        <h1 style="margin:0; font-size: 2rem;">{label}</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">{advice['header']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Detailed Metrics & Advice
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ดัชนีความร้อน (Feels Like)", f"{feels_like:.1f} °C")
    with col2:
        st.metric("ค่าฝุ่นละออง (PM2.5)", f"{pm25:.1f} µg/m³")

    st.markdown("### คำแนะนำการซ้อม (Training Guidance)")
    st.markdown(f"<div class='advice-title'>โปรแกรมการซ้อม:</div> {advice['program']}", unsafe_allow_html=True)
    st.markdown(f"<div class='advice-title'>การดูแลตัวเอง:</div> {advice['nutrition']}", unsafe_allow_html=True)

else:
    st.info("กรุณาเลื่อนแถบค่าพารามิเตอร์ทางซ้ายมือ และกดปุ่มวิเคราะห์เพื่อดูผลลัพธ์")

# Footer
st.markdown("---")
st.caption("Developed by Senior Data Scientist & Sports Science Expert | System v2.1")