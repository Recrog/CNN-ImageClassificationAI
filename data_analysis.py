# Traffic Signs Classification Project
# Hazır veri seti ile başlangıç kodu

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# Veri setinin bulunduğu klasör (Gerçek klasör adınız)
DATA_PATH = "Trafic Signs Preprocssed data/"

def load_preprocessed_data():
    """Önceden işlenmiş veri setini yükler"""
    print("Veri yükleniyor...")
    
    # Önce dosyaları kontrol et
    import os
    print(f"Mevcut dizin: {os.getcwd()}")
    
    # Dosya yollarını kontrol et
    files_to_check = ["train.pickle", "valid.pickle", "test.pickle"]
    for file in files_to_check:
        full_path = DATA_PATH + file
        if os.path.exists(full_path):
            print(f"✅ {file} bulundu")
        else:
            print(f"❌ {file} bulunamadı: {full_path}")
    
    # Alternatif yolları dene
    possible_paths = [
        "Trafic Signs Preprocssed data/",
        "./Trafic Signs Preprocssed data/", 
        "Trafic Signs Preprocssed data\\",
        "Traffic Signs Preprocessed data/",
        "./Traffic Signs Preprocessed data/", 
        "Traffic Signs Preprocessed data\\",
        ""  # Doğrudan ana dizinde
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path + "train.pickle"):
            found_path = path
            print(f"✅ Doğru yol bulundu: {path}")
            break
    
    if found_path is None:
        print("❌ Hiçbir yolda train.pickle bulunamadı!")
        print("Mevcut dosyalar:")
        for item in os.listdir("."):
            print(f"  - {item}")
        return None, None, None, None, None, None
    
    # Doğru yolu kullan
    DATA_PATH_CORRECT = found_path
    
    # Eğitim verisi
    with open(DATA_PATH_CORRECT + "train.pickle", "rb") as f:
        train_data = pickle.load(f)
    X_train = train_data['features']
    y_train = train_data['labels']
    
    # Doğrulama verisi  
    with open(DATA_PATH_CORRECT + "valid.pickle", "rb") as f:
        valid_data = pickle.load(f)
    X_valid = valid_data['features']
    y_valid = valid_data['labels']
    
    # Test verisi
    with open(DATA_PATH_CORRECT + "test.pickle", "rb") as f:
        test_data = pickle.load(f)
    X_test = test_data['features']
    y_test = test_data['labels']
    
    print(f"Eğitim verisi: {X_train.shape} - {len(np.unique(y_train))} sınıf")
    print(f"Doğrulama verisi: {X_valid.shape}")
    print(f"Test verisi: {X_test.shape}")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_label_names():
    """Sınıf isimlerini yükler"""
    import os
    
    # Dosya yollarını dene
    possible_paths = [
        "Trafic Signs Preprocssed data/",
        "./Trafic Signs Preprocssed data/", 
        "Trafic Signs Preprocssed data\\",
        "Traffic Signs Preprocessed data/",
        "./Traffic Signs Preprocessed data/", 
        "Traffic Signs Preprocessed data\\",
        ""
    ]
    
    for path in possible_paths:
        try:
            labels_df = pd.read_csv(path + "label_names.csv")
            print(f"✅ label_names.csv bulundu: {path}")
            return labels_df['SignName'].values
        except:
            continue
    
    print("⚠️  label_names.csv bulunamadı, varsayılan isimler kullanılıyor...")
    return [f"Class_{i}" for i in range(43)]

def explore_dataset(X_train, y_train, class_names):
    """Veri setini keşfet ve görselleştir"""
    print("\n=== VERİ SETİ ANALİZİ ===")
    
    # Temel bilgiler
    print(f"Görüntü boyutu: {X_train.shape[1:]}")
    print(f"Görüntü sayısı: {X_train.shape[0]}")
    print(f"Sınıf sayısı: {len(np.unique(y_train))}")
    print(f"Piksel değer aralığı: {X_train.min():.2f} - {X_train.max():.2f}")
    
    # Sınıf dağılımı
    unique, counts = np.unique(y_train, return_counts=True)
    
    plt.figure(figsize=(15, 5))
    plt.bar(unique, counts)
    plt.title('Sınıf Dağılımı')
    plt.xlabel('Sınıf ID')
    plt.ylabel('Görüntü Sayısı')
    plt.xticks(range(0, len(unique), 5))
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # En fazla ve en az örnekli sınıflar
    class_distribution = pd.DataFrame({
        'Class_ID': unique,
        'Class_Name': [class_names[i] if i < len(class_names) else f"Class_{i}" for i in unique],
        'Count': counts
    })
    
    print("\nEn fazla örnekli 5 sınıf:")
    print(class_distribution.nlargest(5, 'Count'))
    
    print("\nEn az örnekli 5 sınıf:")
    print(class_distribution.nsmallest(5, 'Count'))
    
    return class_distribution

def show_sample_images(X_train, y_train, class_names, samples_per_class=5):
    """Her sınıftan örnek görüntüler göster"""
    unique_classes = np.unique(y_train)
    
    # İlk 10 sınıftan örnekler göster
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Örnek Trafik İşaretleri (İlk 10 Sınıf)', fontsize=16)
    
    for i, class_id in enumerate(unique_classes[:10]):
        row = i // 5
        col = i % 5
        
        # Bu sınıfa ait görüntülerden birini seç
        class_indices = np.where(y_train == class_id)[0]
        sample_idx = class_indices[0]
        
        # Görüntüyü göster
        axes[row, col].imshow(X_train[sample_idx])
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        axes[row, col].set_title(f"ID:{class_id}\n{class_name[:20]}...", fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def check_data_quality(X_train, y_train):
    """Veri kalitesini kontrol et"""
    print("\n=== VERİ KALİTESİ KONTROLÜ ===")
    
    # Eksik değerler
    print(f"Eksik değer var mı? {np.isnan(X_train).any()}")
    
    # Sıfır değerli görüntüler
    zero_images = np.sum(np.all(X_train == 0, axis=(1,2,3)))
    print(f"Tamamen siyah görüntü sayısı: {zero_images}")
    
    # Histogram analizi (parlaklik dağılımı)
    mean_brightness = np.mean(X_train, axis=(1,2,3))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(mean_brightness, bins=50, alpha=0.7)
    plt.title('Ortalama Parlaklık Dağılımı')
    plt.xlabel('Ortalama Piksel Değeri')
    plt.ylabel('Frekans')
    
    # Renk kanalları analizi
    plt.subplot(1, 2, 2)
    for channel, color in enumerate(['red', 'green', 'blue']):
        channel_mean = np.mean(X_train[:, :, :, channel])
        plt.bar(channel, channel_mean, color=color, alpha=0.7, label=f'{color.capitalize()}')
    
    plt.title('Ortalama Renk Kanalı Değerleri')
    plt.xlabel('Renk Kanalı')
    plt.ylabel('Ortalama Değer')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def normalize_data(X_train, X_valid, X_test):
    """Veriyi normalize et (0-1 arasına getir)"""
    print("Veri normalize ediliyor...")
    
    # Eğer veri 0-255 arasındaysa normalize et
    if X_train.max() > 1:
        X_train_norm = X_train.astype('float32') / 255.0
        X_valid_norm = X_valid.astype('float32') / 255.0
        X_test_norm = X_test.astype('float32') / 255.0
        print(f"Normalizasyon tamamlandı. Yeni aralık: {X_train_norm.min():.3f} - {X_train_norm.max():.3f}")
    else:
        X_train_norm = X_train.astype('float32')
        X_valid_norm = X_valid.astype('float32')
        X_test_norm = X_test.astype('float32')
        print("Veri zaten normalize edilmiş.")
    
    return X_train_norm, X_valid_norm, X_test_norm

# Ana kod
if __name__ == "__main__":
    print("🚦 TRAFİK İŞARETİ SINIFLANDIRMA PROJESİ")
    print("=" * 50)
    
    # 1. Veriyi yükle
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_preprocessed_data()
    
    # 2. Sınıf isimlerini yükle
    class_names = load_label_names()
    print(f"Yüklenen sınıf sayısı: {len(class_names)}")
    
    # 3. Veriyi keşfet
    class_distribution = explore_dataset(X_train, y_train, class_names)
    
    # 4. Örnek görüntüleri göster
    show_sample_images(X_train, y_train, class_names)
    
    # 5. Veri kalitesini kontrol et
    check_data_quality(X_train, y_train)
    
    # 6. Veriyi normalize et
    X_train_norm, X_valid_norm, X_test_norm = normalize_data(X_train, X_valid, X_test)
    
    print("\n✅ Veri analizi tamamlandı!")
    print("Sonraki adım: Model oluşturma ve eğitim")
    print("\nKullanım için:")
    print("X_train_norm, y_train -> Eğitim verisi")
    print("X_valid_norm, y_valid -> Doğrulama verisi") 
    print("X_test_norm, y_test -> Test verisi")
    print(f"class_names -> {len(class_names)} sınıf ismi")