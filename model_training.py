# CNN Model Eğitimi - Trafik İşaretleri
# Önceki data_analysis.py'dan sonra çalıştırın

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import cv2
import os
from datetime import datetime

# Veri setinin bulunduğu klasör
DATA_PATH = "Trafic Signs Preprocssed data/"

def load_preprocessed_data():
    """Önceden işlenmiş veri setini yükler"""
    print("Veri yükleniyor...")
    
    # Eğitim verisi
    with open(DATA_PATH + "train.pickle", "rb") as f:
        train_data = pickle.load(f)
    X_train = train_data['features']
    y_train = train_data['labels']
    
    # Doğrulama verisi  
    with open(DATA_PATH + "valid.pickle", "rb") as f:
        valid_data = pickle.load(f)
    X_valid = valid_data['features']
    y_valid = valid_data['labels']
    
    # Test verisi
    with open(DATA_PATH + "test.pickle", "rb") as f:
        test_data = pickle.load(f)
    X_test = test_data['features']
    y_test = test_data['labels']
    
    print(f"✅ Eğitim verisi: {X_train.shape}")
    print(f"✅ Doğrulama verisi: {X_valid.shape}")
    print(f"✅ Test verisi: {X_test.shape}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def normalize_data(X_train, X_valid, X_test):
    """Veriyi normalize et (0-1 arasına getir)"""
    print("Veri normalize ediliyor...")
    
    if X_train.max() > 1:
        X_train_norm = X_train.astype('float32') / 255.0
        X_valid_norm = X_valid.astype('float32') / 255.0
        X_test_norm = X_test.astype('float32') / 255.0
        print(f"✅ Normalizasyon tamamlandı: {X_train_norm.min():.3f} - {X_train_norm.max():.3f}")
    else:
        X_train_norm = X_train.astype('float32')
        X_valid_norm = X_valid.astype('float32')
        X_test_norm = X_test.astype('float32')
        print("✅ Veri zaten normalize edilmiş")
    
    return X_train_norm, X_valid_norm, X_test_norm

def create_data_augmentation():
    """Veri artırma katmanları"""
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.1),                    # ±10 derece döndürme
        layers.RandomZoom(0.1),                        # %10 zoom
        layers.RandomBrightness(0.1),                  # Parlaklık değişimi
        layers.RandomContrast(0.1),                    # Kontrast değişimi
    ])
    return data_augmentation

def create_cnn_model_v1(input_shape=(32, 32, 3), num_classes=43):
    """Basit CNN modeli (Versiyon 1)"""
    print("🔨 Basit CNN modeli oluşturuluyor...")
    
    model = keras.Sequential([
        # Giriş katmanı
        layers.Input(shape=input_shape),
        
        # İlk konvolüsyon bloğu
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # İkinci konvolüsyon bloğu
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Üçüncü konvolüsyon bloğu
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_cnn_model_v2(input_shape=(32, 32, 3), num_classes=43):
    """Gelişmiş CNN modeli (Versiyon 2) - Veri artırma ve Batch Normalization ile"""
    print("🚀 Gelişmiş CNN modeli oluşturuluyor...")
    
    # Veri artırma
    data_augmentation = create_data_augmentation()
    
    model = keras.Sequential([
        # Veri artırma (sadece eğitimde aktif)
        data_augmentation,
        
        # İlk blok
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # İkinci blok
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Üçüncü blok
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Global Average Pooling (Flatten yerine)
        layers.GlobalAveragePooling2D(),
        
        # Classifier
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.001): #LR ayarını ekledik ORT, daha küçük veya büyük olabilir
    """Modeli derle"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model derlendi")
    return model

def setup_callbacks(model_name="traffic_sign_model"):
    """Eğitim callback'lerini ayarla"""
    callbacks = [
        # Erken durma - 5 epoch boyunca iyileşme yoksa dur
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate azaltma - 3 epoch boyunca iyileşme yoksa LR'yi azalt
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        ),
        
        # En iyi modeli kaydet    # .h5 yerine Keras'ın yeni formatı olan .keras kullanabilirsiniz?
        ModelCheckpoint(
            f"{model_name}_best.keras" ,  # .h5 yerine .keras
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

def train_model(model, X_train, y_train, X_valid, y_valid, epochs=30, batch_size=32):
    """Modeli eğit"""
    print(f"🏋️‍♂️ Model eğitimi başlıyor... ({epochs} epoch)")
    
    # Callback'leri ayarla      #Olası sorun 2. model için .h5 yerine .keras kullanılması olabilir?? araştır
    callbacks = setup_callbacks()
    
    # Eğitimi başlat
    start_time = datetime.now()
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_valid, y_valid),
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print(f"✅ Eğitim tamamlandı! Süre: {training_time}")
    
    return history

def plot_training_history(history):
    """Eğitim geçmişini görselleştir"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy grafiği
    ax1.plot(history.history['accuracy'], label='Eğitim Doğruluğu', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu', marker='s')
    ax1.set_title('Model Doğruluğu')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss grafiği
    ax2.plot(history.history['loss'], label='Eğitim Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Doğrulama Loss', marker='s')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # En iyi sonuçları yazdır
    best_train_acc = max(history.history['accuracy'])
    best_val_acc = max(history.history['val_accuracy'])
    
    print(f"\n📊 EĞİTİM SONUÇLARI:")
    print(f"En iyi eğitim doğruluğu: {best_train_acc:.4f}")
    print(f"En iyi doğrulama doğruluğu: {best_val_acc:.4f}")
    
    return best_val_acc

def evaluate_model(model, X_test, y_test):
    """Modeli test verisiyle değerlendir"""
    print("\n🧪 Model test ediliyor...")
    
    # Test doğruluğu
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"📈 TEST SONUÇLARI:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return test_accuracy

# Ana eğitim fonksiyonu
def main_training():
    """Ana eğitim pipeline'ı"""
    print("=" * 60)
    print("🚦 TRAFİK İŞARETİ CNN MODEL EĞİTİMİ")
    print("=" * 60)
    
    # 1. Veriyi yükle
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_preprocessed_data() 
    
    # 2. Normalize et
    X_train_norm, X_valid_norm, X_test_norm = normalize_data(X_train, X_valid, X_test)
    
    print(f"\n📊 VERİ BİLGİLERİ:")
    print(f"Görüntü boyutu: {X_train_norm.shape[1:]}")
    print(f"Sınıf sayısı: {len(np.unique(y_train))}")
    print(f"Eğitim örnekleri: {len(X_train_norm):,}")
    print(f"Doğrulama örnekleri: {len(X_valid_norm):,}")
    print(f"Test örnekleri: {len(X_test_norm):,}")
    
    # 3. Model seç (1 veya 2)
    print("\n🔧 MODEL SEÇİMİ:")
    print("1️⃣  Basit CNN (Hızlı eğitim)")
    print("2️⃣  Gelişmiş CNN (Daha iyi performans)")
    
    choice = input("Hangi modeli kullanmak istiyorsunuz? (1/2): ")
    
    if choice == "1":
        model = create_cnn_model_v1()
        model_name = "basit_cnn"
        epochs = 20
    else:
        model = create_cnn_model_v2()
        model_name = "gelismis_cnn"
        epochs = 30
    
    # 4. Modeli derle
    model = compile_model(model)
    
    # 5. Model özetini göster
    print(f"\n🏗️  {model_name.upper()} MODEL ÖZETİ:")
    model.summary()
    
    # 6. Modeli eğit
    history = train_model(model, X_train_norm, y_train, X_valid_norm, y_valid, epochs=epochs)
    
    # 7. Sonuçları görselleştir
    best_val_acc = plot_training_history(history)
    
    # 8. Test et
    test_accuracy = evaluate_model(model, X_test_norm, y_test)
    
    # 9. Modeli kaydet
    model_filename = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.keras"
    model.save(model_filename)
    print(f"\n💾 Model kaydedildi: {model_filename}")
    
    print(f"\n🎉 EĞİTİM TAMAMLANDI!")
    print(f"Doğrulama Doğruluğu: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Test Doğruluğu: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return model, history

if __name__ == "__main__":
    # Eğitimi başlat
    model, history = main_training()