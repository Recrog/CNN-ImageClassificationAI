# updated_streamlit_app_fixed.py
# 🚦 Trafik işareti tanıma (robust + crop)
# - Model: robust_cnn32_best.keras
# - Preprocess: channel_stats.json
# - Crop: kırmızı/mavi alanlardan işareti bulup kırpma

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os, json
from datetime import datetime

st.set_page_config(page_title="Robust Trafik İşareti Tanıma", page_icon="🚦", layout="wide")

CLASS_NAMES = [
    'Hız limiti (20km/h)', 'Hız limiti (30km/h)', 'Hız limiti (50km/h)', 'Hız limiti (60km/h)',
    'Hız limiti (70km/h)', 'Hız limiti (80km/h)', 'Hız limiti (80km/h) sonu', 'Hız limiti (100km/h)',
    'Hız limiti (120km/h)', 'Geçme yasağı', 'Kamyonlara geçme yasağı', 'Kavşak öncelik yolu',
    'Öncelikli yol', 'Yol ver', 'Dur', 'Araç trafiğine kapalı', 'Kamyonlara kapalı', 'Giriş yasak',
    'Genel tehlike', 'Tehlikeli sol viraj', 'Tehlikeli sağ viraj', 'Çifte viraj', 'Engebeli yol',
    'Kaygan yol', 'Sağda daralan yol', 'Yol çalışması', 'Trafik ışıkları', 'Yaya geçidi',
    'Çocuklar', 'Bisiklet geçidi', 'Buzlu/karlı yol', 'Vahşi hayvan geçidi', 
    'Hız ve geçme yasağı sonu', 'Sağa dönüş zorunlu', 'Sola dönüş zorunlu', 'Düz gitme zorunlu',
    'Düz git veya sağa dön', 'Düz git veya sola dön', 'Sağdan geçme zorunlu', 'Soldan geçme zorunlu',
    'Ada sağdan', 'Ada soldan', 'Zorunlu yön'
]

# -------------------------
# Model yükleme
# -------------------------
@st.cache_resource
def load_model():
    model_files = [f for f in os.listdir('.') if f.endswith('.keras') or f.endswith('.h5')]
    if not model_files:
        st.error("❌ Model dosyası bulunamadı.")
        return None, None
    preferred = [f for f in model_files if "robust" in f.lower() or "best" in f.lower()]
    chosen = max(preferred or model_files, key=os.path.getctime)
    model = tf.keras.models.load_model(chosen)
    return model, chosen

def get_channel_stats():
    if os.path.exists("channel_stats.json"):
        try:
            with open("channel_stats.json","r", encoding="utf-8") as f:
                js = json.load(f)
            return js.get("means",[0.3403,0.3121,0.3214]), js.get("stds",[0.2724,0.2608,0.2669])
        except Exception:
            pass
    return [0.3403,0.3121,0.3214], [0.2724,0.2608,0.2669]

# -------------------------
# Detection + Crop
# -------------------------
def find_and_crop_sign(img_bgr, target_size=32):
    """Renk maskesi ile en büyük kırmızı/mavi işareti bul ve crop et"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # kırmızı maske (2 aralık)
    lower_red1, upper_red1 = np.array([0,70,50]), np.array([10,255,255])
    lower_red2, upper_red2 = np.array([160,70,50]), np.array([180,255,255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # mavi maske
    lower_blue, upper_blue = np.array([100,70,50]), np.array([130,255,255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.bitwise_or(mask_red, mask_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        if w*h > 200:  # çok küçükse alma
            cropped = img_bgr[y:y+h, x:x+w]
            if cropped.size > 0:
                return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # fallback: merkezden kare crop
    h,w,_ = img_bgr.shape
    m = min(h,w)
    sx, sy = (w-m)//2, (h-m)//2
    cropped = img_bgr[sy:sy+m, sx:sx+m]
    return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

# -------------------------
# Preprocess
# -------------------------
def robust_preprocess_image(pil_image, img_size=32):
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    sign = find_and_crop_sign(img_bgr, img_size)

    img = sign.astype("float32")/255.0
    means, stds = get_channel_stats()
    for c in range(3):
        img[:,:,c] = (img[:,:,c] - means[c]) / (stds[c] + 1e-8)
    return np.expand_dims(img,0), img

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("🛡️ Robust Trafik İşareti Tanıma")
    model, model_name = load_model()
    if model is None: st.stop()

    uploaded = st.file_uploader("Bir görüntü yükleyin", type=["jpg","jpeg","png","bmp"])
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Yüklenen görüntü", use_column_width=True)

        if st.button("Tahmin Et"):
            x, vis = robust_preprocess_image(image, 32)
            pred = model.predict(x, verbose=0)[0]
            cls = int(np.argmax(pred))
            conf = float(np.max(pred))
            st.subheader(f"Tahmin: {CLASS_NAMES[cls]} ({conf*100:.1f}%)")
            st.image((vis*255).astype(np.uint8), caption="İşlenmiş crop (32x32)", width=160)

    st.caption(f"Güncelleme: {datetime.now().strftime('%Y-%m-%d')} | Model: {model_name}")

# streamlit run updated_streamlit_app_fixed.py
if __name__ == "__main__":
    main()
