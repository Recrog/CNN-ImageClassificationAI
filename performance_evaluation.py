# Model Performans Analizi ve Raporu
# performance_evaluation.py

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Türkçe font ayarları
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Trafik işareti sınıf isimleri
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

def load_data_and_model():
    """Veri seti ve modeli yükle"""
    print("📊 Veri seti ve model yükleniyor...")
    
    # Veri setini yükle
    DATA_PATH = "Trafic Signs Preprocssed data/"
    
    # Test verisi
    with open(DATA_PATH + "test.pickle", "rb") as f:
        test_data = pickle.load(f)
    X_test = test_data['features']
    y_test = test_data['labels']
    
    # Veriyi normalize et
    if X_test.max() > 1:
        X_test = X_test.astype('float32') / 255.0
    else:
        X_test = X_test.astype('float32')
    
    # En son kaydedilen modeli bul
    model_files = [f for f in os.listdir('.') if f.endswith('.keras') or f.endswith('.h5')]
    if model_files:
        latest_model = max(model_files, key=os.path.getctime)
        model = tf.keras.models.load_model(latest_model)
        print(f"✅ Model yüklendi: {latest_model}")
    else:
        raise FileNotFoundError("❌ Model dosyası bulunamadı!")
    
    print(f"✅ Test verisi: {X_test.shape}")
    print(f"✅ Sınıf sayısı: {len(np.unique(y_test))}")
    
    return X_test, y_test, model, latest_model

def generate_predictions(model, X_test, y_test):
    """Model tahminlerini oluştur"""
    print("\n🔮 Tahminler oluşturuluyor...")
    
    # Tahmin yap
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print(f"✅ {len(y_pred)} tahmin tamamlandı")
    
    return y_pred, y_pred_proba

def calculate_basic_metrics(y_test, y_pred):
    """Temel metrikleri hesapla"""
    print("\n📈 Temel metrikler hesaplanıyor...")
    
    # Temel metrikler
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision (Macro)': precision_macro,
        'Recall (Macro)': recall_macro,
        'F1-Score (Macro)': f1_macro,
        'Precision (Weighted)': precision_weighted,
        'Recall (Weighted)': recall_weighted,
        'F1-Score (Weighted)': f1_weighted
    }
    
    return metrics_dict

def plot_confusion_matrix(y_test, y_pred, class_names, save_path=None):
    """Confusion Matrix'i görselleştir"""
    print("\n🎯 Confusion Matrix oluşturuluyor...")
    
    # Confusion matrix hesapla
    cm = confusion_matrix(y_test, y_pred)
    
    # Yüzdelik confusion matrix
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 2 ayrı grafik: Sayı ve yüzde
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # 1. Sayı bazında
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(43), yticklabels=range(43), ax=ax1)
    ax1.set_title('Confusion Matrix (Sayı)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Tahmin Edilen Sınıf', fontsize=12)
    ax1.set_ylabel('Gerçek Sınıf', fontsize=12)
    
    # 2. Yüzde bazında
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Oranges', 
                xticklabels=range(43), yticklabels=range(43), ax=ax2)
    ax2.set_title('Confusion Matrix (Yüzde %)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Tahmin Edilen Sınıf', fontsize=12)
    ax2.set_ylabel('Gerçek Sınıf', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrix kaydedildi: {save_path}")
    
    plt.show()
    
    return cm

def analyze_class_performance(y_test, y_pred, class_names):
    """Sınıf bazında performans analizi"""
    print("\n🔍 Sınıf bazında analiz yapılıyor...")
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                 target_names=[f"Class_{i}" for i in range(43)],
                                 output_dict=True, zero_division=0)
    
    # DataFrame'e dönüştür
    df_report = pd.DataFrame(report).transpose()
    
    # Sınıf isimlerini ekle
    class_data = []
    for i in range(43):
        if f"Class_{i}" in df_report.index:
            class_data.append({
                'Class_ID': i,
                'Class_Name': class_names[i] if i < len(class_names) else f"Class_{i}",
                'Precision': df_report.loc[f"Class_{i}", 'precision'],
                'Recall': df_report.loc[f"Class_{i}", 'recall'],
                'F1-Score': df_report.loc[f"Class_{i}", 'f1-score'],
                'Support': int(df_report.loc[f"Class_{i}", 'support'])
            })
    
    df_class_performance = pd.DataFrame(class_data)
    
    return df_class_performance, report

def plot_class_performance(df_class_performance, save_path=None):
    """Sınıf performansını görselleştir"""
    print("\n📊 Sınıf performans grafikleri oluşturuluyor...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. F1-Score dağılımı
    axes[0,0].bar(df_class_performance['Class_ID'], df_class_performance['F1-Score'], 
                  color='skyblue', alpha=0.7)
    axes[0,0].set_title('Sınıf Bazında F1-Score', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Sınıf ID')
    axes[0,0].set_ylabel('F1-Score')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 1)
    
    # 2. Precision vs Recall
    scatter = axes[0,1].scatter(df_class_performance['Precision'], df_class_performance['Recall'], 
                               c=df_class_performance['F1-Score'], cmap='viridis', alpha=0.7, s=60)
    axes[0,1].set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Precision')
    axes[0,1].set_ylabel('Recall')
    axes[0,1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0,1], label='F1-Score')
    
    # 3. Support (örnek sayısı) dağılımı
    axes[1,0].bar(df_class_performance['Class_ID'], df_class_performance['Support'], 
                  color='lightcoral', alpha=0.7)
    axes[1,0].set_title('Sınıf Bazında Test Örneği Sayısı', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Sınıf ID')
    axes[1,0].set_ylabel('Örnek Sayısı')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. En iyi ve en kötü performans gösteren sınıflar
    top_5 = df_class_performance.nlargest(5, 'F1-Score')
    bottom_5 = df_class_performance.nsmallest(5, 'F1-Score')
    
    combined = pd.concat([top_5, bottom_5])
    colors = ['green']*5 + ['red']*5
    
    y_pos = range(len(combined))
    axes[1,1].barh(y_pos, combined['F1-Score'], color=colors, alpha=0.7)
    axes[1,1].set_yticks(y_pos)
    axes[1,1].set_yticklabels([f"{row['Class_ID']}: {row['Class_Name'][:20]}..." 
                               for _, row in combined.iterrows()], fontsize=8)
    axes[1,1].set_title('En İyi ve En Kötü Performans (F1-Score)', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('F1-Score')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Performans grafikleri kaydedildi: {save_path}")
    
    plt.show()

def create_performance_summary(metrics_dict, df_class_performance, model_name):
    """Performans özet raporu oluştur"""
    print("\n📋 Performans özet raporu oluşturuluyor...")
    
    # Genel statistikler
    avg_f1 = df_class_performance['F1-Score'].mean()
    std_f1 = df_class_performance['F1-Score'].std()
    min_f1 = df_class_performance['F1-Score'].min()
    max_f1 = df_class_performance['F1-Score'].max()
    
    # En iyi ve en kötü sınıflar
    best_class = df_class_performance.loc[df_class_performance['F1-Score'].idxmax()]
    worst_class = df_class_performance.loc[df_class_performance['F1-Score'].idxmin()]
    
    # Rapor metni
    summary_report = f"""
🎯 TRAFIK İŞARETİ SINIFLAMA MODEL PERFORMANS RAPORU
{'='*70}

📊 MODEL BİLGİLERİ:
• Model Dosyası: {model_name}
• Test Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Toplam Sınıf Sayısı: 43
• Test Örnek Sayısı: {df_class_performance['Support'].sum()}

📈 GENEL PERFORMANS METRİKLERİ:
• Accuracy (Doğruluk): {metrics_dict['Accuracy']:.4f} ({metrics_dict['Accuracy']*100:.2f}%)
• Precision (Macro): {metrics_dict['Precision (Macro)']:.4f} ({metrics_dict['Precision (Macro)']*100:.2f}%)
• Recall (Macro): {metrics_dict['Recall (Macro)']:.4f} ({metrics_dict['Recall (Macro)']*100:.2f}%)
• F1-Score (Macro): {metrics_dict['F1-Score (Macro)']:.4f} ({metrics_dict['F1-Score (Macro)']*100:.2f}%)
• F1-Score (Weighted): {metrics_dict['F1-Score (Weighted)']:.4f} ({metrics_dict['F1-Score (Weighted)']*100:.2f}%)

📊 SINIF BAZINDA İSTATİSTİKLER:
• Ortalama F1-Score: {avg_f1:.4f} ± {std_f1:.4f}
• En Yüksek F1-Score: {max_f1:.4f}
• En Düşük F1-Score: {min_f1:.4f}
• F1-Score Aralığı: {max_f1-min_f1:.4f}

🏆 EN İYİ PERFORMANS GÖSTEREN SINIF:
• Sınıf ID: {best_class['Class_ID']}
• Sınıf Adı: {best_class['Class_Name']}
• F1-Score: {best_class['F1-Score']:.4f}
• Precision: {best_class['Precision']:.4f}
• Recall: {best_class['Recall']:.4f}

⚠️  EN DÜŞÜK PERFORMANS GÖSTEREN SINIF:
• Sınıf ID: {worst_class['Class_ID']}
• Sınıf Adı: {worst_class['Class_Name']}
• F1-Score: {worst_class['F1-Score']:.4f}
• Precision: {worst_class['Precision']:.4f}
• Recall: {worst_class['Recall']:.4f}

🎯 PERFORMANS DEĞERLENDİRMESİ:
"""
    
    # Performans değerlendirmesi
    if metrics_dict['Accuracy'] >= 0.95:
        summary_report += "• ✅ ÇOK İYİ: Model çok yüksek doğruluk gösteriyor (≥95%)\n"
    elif metrics_dict['Accuracy'] >= 0.90:
        summary_report += "• ✅ İYİ: Model yüksek doğruluk gösteriyor (90-95%)\n"
    elif metrics_dict['Accuracy'] >= 0.80:
        summary_report += "• ⚡ ORTA: Model kabul edilebilir doğruluk gösteriyor (80-90%)\n"
    else:
        summary_report += "• ❌ DÜŞÜK: Model düşük doğruluk gösteriyor (<80%)\n"
    
    if std_f1 < 0.1:
        summary_report += "• ✅ KARARLILILIK: Sınıflar arası performans tutarlı (düşük std)\n"
    else:
        summary_report += "• ⚠️ DAĞINIKLIK: Bazı sınıflar düşük performans gösteriyor\n"
    
    summary_report += f"\n📊 RAPOR ÖZETİ:\n"
    summary_report += f"Bu model, 43 trafik işareti sınıfını %{metrics_dict['Accuracy']*100:.1f} doğrulukla\n"
    summary_report += f"sınıflandırabilir. Genel F1-Score %{metrics_dict['F1-Score (Weighted)']*100:.1f} olup,\n"
    summary_report += f"pratik kullanım için {'uygun' if metrics_dict['Accuracy'] >= 0.85 else 'iyileştirme gerektirir'}.\n"
    
    return summary_report

def save_detailed_report(df_class_performance, summary_report, model_name):
    """Detaylı raporu dosyaya kaydet"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # CSV dosyası
    csv_filename = f"performance_report_{timestamp}.csv"
    df_class_performance.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    # Text raporu
    txt_filename = f"summary_report_{timestamp}.txt"
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    
    print(f"💾 Detaylı rapor kaydedildi: {csv_filename}")
    print(f"💾 Özet rapor kaydedildi: {txt_filename}")
    
    return csv_filename, txt_filename

def main_evaluation():
    """Ana değerlendirme fonksiyonu"""
    print("🚦 TRAFİK İŞARETİ MODEL PERFORMANS ANALİZİ")
    print("="*60)
    
    try:
        # 1. Veri ve modeli yükle
        X_test, y_test, model, model_name = load_data_and_model()
        
        # 2. Tahminleri oluştur
        y_pred, y_pred_proba = generate_predictions(model, X_test, y_test)
        
        # 3. Temel metrikleri hesapla
        metrics_dict = calculate_basic_metrics(y_test, y_pred)
        
        # 4. Sonuçları yazdır
        print("\n🎯 GENEL PERFORMANS METRİKLERİ:")
        print("-" * 40)
        for metric, value in metrics_dict.items():
            print(f"{metric:25}: {value:.4f} ({value*100:.2f}%)")
        
        # 5. Confusion Matrix
        cm = plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, 
                                 save_path=f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        # 6. Sınıf bazında analiz
        df_class_performance, report = analyze_class_performance(y_test, y_pred, CLASS_NAMES)
        
        # 7. Performans grafiklerini göster
        plot_class_performance(df_class_performance, 
                             save_path=f"class_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        # 8. En iyi ve en kötü sınıfları göster
        print("\n🏆 EN İYİ PERFORMANS GÖSTEREN 5 SINIF (F1-Score):")
        print(df_class_performance.nlargest(5, 'F1-Score')[['Class_ID', 'Class_Name', 'F1-Score', 'Precision', 'Recall']])
        
        print("\n⚠️ EN DÜŞÜK PERFORMANS GÖSTEREN 5 SINIF (F1-Score):")
        print(df_class_performance.nsmallest(5, 'F1-Score')[['Class_ID', 'Class_Name', 'F1-Score', 'Precision', 'Recall']])
        
        # 9. Özet rapor oluştur
        summary_report = create_performance_summary(metrics_dict, df_class_performance, model_name)
        
        # 10. Raporu göster
        print(summary_report)
        
        # 11. Raporları kaydet
        csv_file, txt_file = save_detailed_report(df_class_performance, summary_report, model_name)
        
        print(f"\n✅ PERFORMANS ANALİZİ TAMAMLANDI!")
        print(f"📊 Detaylı veriler: {csv_file}")
        print(f"📋 Özet rapor: {txt_file}")
        print(f"🎯 Confusion Matrix ve grafikler: PNG dosyaları")
        
        return df_class_performance, metrics_dict, summary_report
        
    except Exception as e:
        print(f"❌ Hata oluştu: {str(e)}")
        print("Lütfen model ve veri dosyalarının doğru konumda olduğundan emin olun.")

if __name__ == "__main__":
    # Performans analizini çalıştır
    df_results, metrics, summary = main_evaluation()