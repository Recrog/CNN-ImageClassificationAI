Trafik İşareti Tanıma Sistemi
Bu proje, Alman Trafik İşaretleri veri seti kullanılarak Convolutional Neural Network (CNN) tabanlı bir trafik işareti tanıma sistemi geliştirmek için tasarlanmıştır. Sistem, trafik işaretlerini sınıflandırmak için eğitilmiş bir derin öğrenme modeli ve Streamlit tabanlı bir web arayüzü içerir. Kullanıcılar, bir trafik işareti fotoğrafı yükleyerek modelin tahminlerini görebilir ve sonuçları ayrıntılı bir şekilde inceleyebilir.
İçindekiler

Özellikler
Kurulum
Kullanılan Teknolojiler
Veri Seti
Proje Yapısı
Kullanım
Model Eğitimi
Web Arayüzü
Geliştirme ve Katkı
Lisans

Özellikler

Veri Analizi: Veri setinin dağılımını, görüntü kalitesini ve sınıf istatistiklerini görselleştirme.
Model Eğitimi: Basit ve gelişmiş iki farklı CNN modeli seçeneği ile eğitim.
Web Arayüzü: Streamlit ile kullanıcı dostu bir arayüz üzerinden trafik işareti tahmini.
Detaylı Tahminler: Tahmin sonuçlarını grafik ve tablo formatında görüntüleme.
Veri Artırma: Eğitim sırasında veri artırma teknikleri ile model performansını artırma.
Erken Durdurma ve Optimizasyon: Model eğitiminde erken durdurma ve öğrenme oranı azaltma gibi teknikler.

Kurulum

Gerekli Kütüphaneleri Yükleyin:Projenin çalışması için aşağıdaki Python kütüphanelerine ihtiyaç vardır:
pip install streamlit tensorflow numpy pandas matplotlib seaborn opencv-python pillow scikit-learn

Not: TensorFlow GPU desteği için uygun CUDA ve cuDNN sürümlerini yüklemeniz gerekebilir. 
Not: Microsoft Visiual C++ kurulu ve güncel veya yakın olmalı

Veri Setini İndirin:

Alman Trafik İşaretleri veri setinin önceden işlenmiş hali olan Trafic Signs Preprocssed data klasörünü indirin.
Veri setini proje dizinine yerleştirin (örneğin, ./Trafic Signs Preprocssed data/).


Model Dosyasını Hazırlayın:

Model eğitimi için model_training.py dosyasını çalıştırarak model_best.h5 dosyasını oluşturun.
Oluşturulan model dosyasını proje dizinine yerleştirin.



Kullanılan Teknolojiler

Python: 3.13 veya üstü
TensorFlow: CNN model eğitimi ve tahmini
Streamlit: Web arayüzü
OpenCV: Görüntü işleme
NumPy, Pandas: Veri manipülasyonu
Matplotlib, Seaborn: Veri görselleştirme
Pillow: Görüntü ön işleme
Scikit-learn: Sınıflandırma metrikleri

Veri Seti
Proje, Alman Trafik İşaretleri veri setinin önceden işlenmiş versiyonunu kullanır (Trafic Signs Preprocssed data). Bu veri seti:

43 farklı trafik işareti sınıfı içerir.
Eğitim (train.pickle), doğrulama (valid.pickle) ve test (test.pickle) veri setlerini içerir.
Görüntüler 32x32 piksel boyutunda ve RGB formatındadır.
Sınıf isimleri label_names.csv dosyasında bulunur

Proje Yapısı
.
├── Trafic Signs Preprocssed data/  # Veri seti klasörü
│   ├── train.pickle                # Eğitim verisi
│   ├── valid.pickle                # Doğrulama verisi
│   ├── test.pickle                 # Test verisi
│   └── label_names.csv             # Sınıf isimleri (opsiyonel)
├── data_analysis.py                # Veri seti analizi ve görselleştirme
├── model_training.py               # CNN model eğitimi
├── streamlit_app.py                # Web arayüzü
├── model_best.h5                   # Eğitilmiş model dosyası
└── README.md                       # Bu dosya

Kullanım

Veri Analizi:
python data_analysis.py


Veri setini yükler, sınıf dağılımlarını analiz eder ve örnek görüntüleri görselleştirir.
Veri kalitesini kontrol eder ve normalizasyon işlemini gerçekleştirir.


Model Eğitimi:
python model_training.py


Kullanıcıya basit (cnn_v1) veya gelişmiş (cnn_v2) model seçeneği sunar. 
Modeli eğitir, sonuçları görselleştirir ve en iyi modeli model_best.h5 olarak kaydeder.(Gerekirse .keras)
Eğitim süresi ve performansı konsolda görüntülenir.


Web Arayüzünü Başlatma:
streamlit run streamlit_app.py


Yerel bir web sunucusu başlatır (genellikle http://localhost:8501).
Kullanıcılar bir trafik işareti fotoğrafı yükleyebilir ve tahmin sonuçlarını görebilir.



Model Eğitimi

Basit CNN Modeli (V1):
Daha hızlı eğitim için hafif bir yapı.
3 konvolüsyon katmanı ve 2 tam bağlantılı katman içerir.
Yaklaşık %95 doğruluk sağlar.(97 civarı geçen)


Gelişmiş CNN Modeli (V2):
Veri artırma ve Batch Normalization içerir.
Daha iyi genelleştirme için Global Average Pooling kullanır.
Yaklaşık %97 doğruluk sağlar. (99 civarı tahmini geçen)


Eğitim Süreci:
Erken durdurma ve öğrenme oranı azaltma ile optimize edilir.
En iyi model model_best.h5 olarak kaydedilir.
Eğitim geçmişi grafiksel olarak görselleştirilir.



Web Arayüzü

Streamlit tabanlı arayüz, kullanıcıların trafik işareti görüntülerini yüklemesine olanak tanır.
Özellikler:
Görüntü yükleme ve ön işleme (32x32 piksele yeniden boyutlandırma, normalizasyon).
Tahmin sonuçlarını grafik ve tablo formatında görüntüleme.
Güven seviyesine göre uyarılar (düşük, orta, yüksek).
İşlenmiş görüntünün önizlemesi.


Kullanım İpuçları:
En iyi sonuçlar için net ve iyi aydınlatılmış merkezi görüntüler kullanın. 
JPG, PNG veya BMP formatları desteklenir.



Geliştirme ve Katkı (İleride github paylaşımı )

Yeni özellikler eklemek için bir dal oluşturun ve pull request gönderin.
Hata raporları veya öneriler için lütfen bir issue açın.
Model performansını artırmak için:
Daha fazla veri artırma tekniği deneyin.
Model mimarisini özelleştirin (örneğin, ek katmanlar ekleyin).
Hiperparametre optimizasyonu yapın (öğrenme oranı, batch boyutu vb.).





Geliştirici: Recep Özenç Stajyer Projesi