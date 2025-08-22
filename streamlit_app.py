# Trafik İşareti Tanıma Web Arayüzü
# streamlit_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Sayfa yapılandırması
st.set_page_config(
    page_title="Trafik İşareti Tanıma",
    page_icon="🚦",
    layout="wide"
)

# CSS stil ekleyin
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Trafik işareti sınıf isimleri (43 sınıf)
CLASS_NAMES = [
    'Hız limiti (20km/h)',
    'Hız limiti (30km/h)', 
    'Hız limiti (50km/h)',
    'Hız limiti (60km/h)',
    'Hız limiti (70km/h)',
    'Hız limiti (80km/h)',
    'Hız limiti (80km/h) sonu',
    'Hız limiti (100km/h)',
    'Hız limiti (120km/h)',
    'Geçme yasağı',
    'Kamyonlara geçme yasağı',
    'Kavşak öncelik yolu',
    'Öncelikli yol',
    'Yol ver',
    'Dur',
    'Araç trafiğine kapalı',
    'Kamyonlara kapalı',
    'Giriş yasak',
    'Genel tehlike',
    'Tehlikeli sol viraj',
    'Tehlikeli sağ viraj',
    'Çifte viraj',
    'Engebeli yol',
    'Kaygan yol',
    'Sağda daralan yol',
    'Yol çalışması',
    'Trafik ışıkları',
    'Yaya geçidi',
    'Çocuklar',
    'Bisiklet geçidi',
    'Buzlu/karlı yol',
    'Vahşi hayvan geçidi',
    'Hız ve geçme yasağı sonu',
    'Sağa dönüş zorunlu',
    'Sola dönüş zorunlu',
    'Düz gitme zorunlu',
    'Düz git veya sağa dön', 
    'Düz git veya sola dön',
    'Sağdan geçme zorunlu',
    'Soldan geçme zorunlu',
    'Ada sağdan',
    'Ada soldan',
    'Zorunlu yön'
]

@st.cache_resource
def load_model():
    """Eğitilmiş modeli yükle"""
    try:
        # En son kaydedilen modeli bul
        model_files = [f for f in os.listdir('.') if f.endswith('.h5')  or f.endswith('.keras')]
        if model_files:
            # En yeni dosyayı seç
            latest_model = max(model_files, key=os.path.getctime)
            model = tf.keras.models.load_model(latest_model)
            return model, latest_model
        
        else:
            st.error("❌ Model dosyası bulunamadı! Önce modeli eğitmelisiniz.")
            return None, None
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata: {str(e)}")
        return None, None

def preprocess_image(image):
    """Yüklenen görüntüyü model için hazırla"""
    try:
        # --- 1. Görseli beyaz arka plana yerleştir (şeffaflık varsa) ---
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image.convert('RGBA')).convert('RGB')
        else:
            image = image.convert('RGB')

        # --- 2. Görseli merkeze yerleştir, kenarları beyaz doldur (kare yap) ---
        w, h = image.size
        max_side = max(w, h)
        new_img = Image.new('RGB', (max_side, max_side), (255, 255, 255))
        new_img.paste(image, ((max_side - w) // 2, (max_side - h) // 2))
        image = new_img

        # --- 3. Kontrastı artır (isteğe bağlı, daha net olması için) ---
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # 2.0 kontrastı artırır, gerekirse 1.5-3.0 arası dene

        # --- 4. Numpy array'e çevir ---
        img_array = np.array(image)

        # --- 5. 32x32'ye yeniden boyutlandır ---
        img_resized = cv2.resize(img_array, (32, 32), interpolation=cv2.INTER_AREA)

        # --- 6. Normalize et (0-1 arası) ---
        img_normalized = img_resized.astype('float32') / 255.0

        # --- 7. Batch dimension ekle ---
        img_batch = np.expand_dims(img_normalized, axis=0)

        # --- 8. Debug: Min-max ve işlenmiş görseli göster ---
        st.write(f"Min-Max: {img_normalized.min():.3f} - {img_normalized.max():.3f}")
        st.image(img_resized, caption="32x32 İşlenmiş Görsel", width=128)
        st.info(f"✅ Görüntü işlendi: {img_batch.shape}")

        return img_batch, img_resized
    except Exception as e:
        st.error(f"Görüntü işlenirken hata: {str(e)}")
        return None, None

def predict_sign(model, processed_image):
    """Trafik işaretini tahmin et"""
    try:
        # Tahmin yap
        predictions = model.predict(processed_image, verbose=0)
        
        # En yüksek olasılıklı sınıfı bul
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Tüm tahminleri döndür
        all_predictions = [(i, prob, CLASS_NAMES[i]) for i, prob in enumerate(predictions[0])]
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predicted_class, confidence, all_predictions
    except Exception as e:
        st.error(f"Tahmin yapılırken hata: {str(e)}")
        return None, None, None

def display_prediction_chart(all_predictions, top_n=5):
    """Tahmin grafiğini göster"""
    # En yüksek N tahmini al
    top_predictions = all_predictions[:top_n]
    
    # Grafik oluştur
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = [pred[2][:25] + "..." if len(pred[2]) > 25 else pred[2] for pred in top_predictions]
    probabilities = [pred[1] * 100 for pred in top_predictions]
    
    bars = ax.barh(classes, probabilities, color=['#ff7f0e' if i == 0 else '#1f77b4' for i in range(len(classes))])
    
    ax.set_xlabel('Olasılık (%)')
    ax.set_title(f'En Yüksek {top_n} Tahmin')
    ax.set_xlim(0, 100)
    
    # Değerleri çubuklarda göster
    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', ha='left', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)

def main():
    # Ana başlık
    st.markdown('<h1 class="main-header">🚦 Trafik İşareti Tanıma Sistemi</h1>', unsafe_allow_html=True)
    
    # Model yükleme
    with st.spinner('Model yükleniyor...'):
        model, model_name = load_model()
    
    if model is None:
        st.stop()
    
    # Model bilgisi
    st.markdown(f'<div class="success-box">✅ Model başarıyla yüklendi: <b>{model_name}</b></div>', 
                unsafe_allow_html=True)
    st.sidebar.success(f"Yüklenen model dosyası: {model_name}")
    # Kenar çubuğu - Bilgiler
    st.sidebar.markdown("## 📊 Model Bilgileri")
    st.sidebar.info(f"""
    **Model:** {model_name}
    **Sınıf Sayısı:** 43
    **Görüntü Boyutu:** 32x32 piksel
    **Eğitim Doğruluğu:** ~97%
    """)
    
    st.sidebar.markdown("## 🎯 Nasıl Kullanılır?")
    st.sidebar.markdown("""
    1. Trafik işareti fotoğrafı yükleyin
    2. 'Tahmin Et' butonuna tıklayın
    3. Sonuçları inceleyin
    
    **İpucu:** En iyi sonuç için net ve merkezi fotoğraflar kullanın!
    """)
    
    # Ana içerik
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Görüntü Yükleme")
        
        # Dosya yükleme
        uploaded_file = st.file_uploader(
            "Trafik işareti fotoğrafı seçin:",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="JPG, PNG veya BMP formatında fotoğraf yükleyebilirsiniz."
        )
        
        if uploaded_file is not None:
            # Orijinal görüntüyü göster
            image = Image.open(uploaded_file)
            st.image(image, caption='Yüklenen Fotoğraf', use_column_width=True)
            
            # Görüntü bilgileri
            st.info(f"**Görüntü Boyutu:** {image.size[0]} x {image.size[1]} piksel")
    
    with col2:
        st.markdown("### 🎯 Tahmin Sonuçları")   #solundakiyle aligned değil nasıl çözerim
        st.markdown("")
        st.markdown("")

        
        if uploaded_file is not None:
            if st.button("🔍 Tahmin Et", type="primary"):
                with st.spinner('Tahmin yapılıyor...'):
                    # Görüntüyü işle
                    processed_image, resized_image = preprocess_image(image)
                    
                    if processed_image is not None:
                        # Tahmin yap
                        predicted_class, confidence, all_predictions = predict_sign(model, processed_image)
                        
                        if predicted_class is not None:
                            # Ana sonuç
                            st.markdown(f"""
                            <div class="success-box">
                            <h3>🏆 Tahmin Sonucu</h3>
                            <h2 style="color: #155724;">{CLASS_NAMES[predicted_class]}</h2>
                            <h3>Güven: {confidence*100:.1f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # İşlenmiş görüntüyü göster
                            st.markdown("**İşlenmiş Görüntü (32x32):**")
                            st.image(resized_image, width=200)
                            
                            # Detaylı sonuçlar
                            with st.expander("📈 Detaylı Tahmin Sonuçları"):
                                # Grafik göster
                                display_prediction_chart(all_predictions)
                                
                                # Tablo halinde göster
                                st.markdown("**Top 10 Tahmin:**")
                                df_results = pd.DataFrame({
                                    'Sıra': range(1, 11),
                                    'Sınıf': [pred[2] for pred in all_predictions[:10]],
                                    'Olasılık (%)': [f"{pred[1]*100:.2f}%" for pred in all_predictions[:10]]
                                })
                                st.dataframe(df_results, use_container_width=True)
                            
                            # Güven seviyesine göre uyarı
                            if confidence < 0.5:
                                st.warning("⚠️ Düşük güven seviyesi! Fotoğrafın daha net ve merkezi olduğundan emin olun.")
                            elif confidence < 0.7:
                                st.info("💡 Orta güven seviyesi. Sonuç doğru olabilir ama dikkatli olun.")
                            else:
                                st.success("✅ Yüksek güven seviyesi! Sonuç büyük olasılıkla doğrudur.")
        else:
            st.info("👆 Önce bir fotoğraf yükleyin")
    
    # Alt bilgi
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>🤖 Bu sistem CNN (Convolutional Neural Network) teknolojisi kullanarak Alman Trafik İşaretleri veri seti ile eğitilmiştir.</p>
    <p>Geliştiren: Stajyer Projesi | Tarih: """ + datetime.now().strftime("%Y-%m-%d") + """</p>
    </div>
    """, unsafe_allow_html=True)


# streamlit run streamlit_app.py
# Uygulamayı çalıştırır local host başlatır
if __name__ == "__main__":
    main()