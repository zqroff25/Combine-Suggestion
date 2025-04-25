from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import requests
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/kombin-olustur', methods=['POST'])
def kombin_olustur():
    if 'files[]' not in request.files:
        return jsonify({"error": "Dosya bulunamadı"}), 400

    files = request.files.getlist('files[]')
    if len(files) < 2:
        return jsonify({"error": "En az 2 kıyafet resmi yüklemelisiniz."}), 400

    tarz = request.form.get('tarz', 'Casual')
    mevsim = request.form.get('mevsim', 'İlkbahar')
    image_descriptions = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # BLIP çağrısı
            blip_api = os.environ.get("BLIP_API_URL")
            resp = requests.post(blip_api, files={'image': open(filepath, 'rb')})
            desc = resp.json().get("description", "")
            image_descriptions.append(desc)

    gemma_api = os.environ.get("GEMMA_API_URL")
    gemma_prompt = f"""
Sen moda uzmanısın. Şu kıyafetleri "{tarz}" tarzda ve "{mevsim}" mevsimine uygun analiz et:
{', '.join(image_descriptions)}

Lütfen aşağıdaki başlıklara 1-2 cümlelik profesyonel ve sade cevaplar ver. Başlıkları belirt:
Kıyafet Analizi:
Renk Uyumu:
Kombin Önerisi:
Kombin Yorumu:
Alternatif Kombin:
"""

    response = requests.post(gemma_api, json={"prompt": gemma_prompt})
    if response.status_code == 200:
        answer = response.json().get("response", "")
    else:
        answer = "Model yanıt vermedi."

    return jsonify({
        "blip_aciklamalari": image_descriptions,
        "gemma_cevabi": answer
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))




'''# Gerekli kütüphaneleri içe aktarıyoruz
from flask import Flask, render_template, request, jsonify  # Flask ve ilgili modülleri içe aktarıyoruz
import os  # İşletim sistemi ile ilgili işlemler için os modülünü içe aktarıyoruz
from PIL import Image  # Görüntü işleme için PIL kütüphanesini içe aktarıyoruz
from transformers import BlipProcessor, BlipForConditionalGeneration  # BLIP modeli için gerekli kütüphaneleri içe aktarıyoruz
import torch  # PyTorch kütüphanesini içe aktarıyoruz
from werkzeug.utils import secure_filename  # Güvenli dosya isimlendirme için werkzeug modülünü içe aktarıyoruz
import requests  # HTTP istekleri yapmak için requests kütüphanesini içe aktarıyoruz
import re  # Düzenli ifadeler için re modülünü içe aktarıyoruz
import uuid  # Benzersiz kimlikler oluşturmak için uuid modülünü içe aktarıyoruz

# Flask uygulamasını başlatıyoruz
app = Flask(__name__)  # Flask uygulama nesnesi oluşturuyoruz

# Yükleme klasörünü ayarlıyoruz
UPLOAD_FOLDER = 'static/uploads'  # Yükleme klasörünün yolunu belirliyoruz
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Klasör yoksa oluşturuyoruz
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Flask uygulama konfigürasyonuna yükleme klasörünü ekliyoruz

# BLIP Modelini yüklüyoruz ve cihazı ayarlıyoruz
# Eğer CUDA destekli bir GPU varsa onu kullanıyoruz, yoksa CPU kullanıyoruz
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Cihazı ayarlıyoruz
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')  # BLIP işlemcisini yüklüyoruz
blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)  # BLIP modelini yüklüyoruz ve cihaza gönderiyoruz

# Dosya uzantısını kontrol eden yardımcı fonksiyon
def allowed_file(filename):  # Dosya uzantısını kontrol eden fonksiyon
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}  # Dosya uzantısının izin verilen türlerden biri olup olmadığını kontrol ediyoruz

# BLIP modeli ile görselden açıklama üreten fonksiyon
def blip_generate(image_path):  # Görselden açıklama üreten fonksiyon
    image = Image.open(image_path).convert("RGB")  # Görseli açıp RGB formatına çeviriyoruz
    inputs = processor(image, return_tensors='pt').to(device)  # Görseli işleyip tensör formatına çeviriyoruz
    out = blip_model.generate(**inputs)  # BLIP modeli ile açıklama üretiyoruz
    description = processor.decode(out[0], skip_special_tokens=True)  # Üretilen açıklamayı metne çeviriyoruz
    return description  # Açıklamayı döndürüyoruz

# GEMMA API'sinden dönen metni başlıklara ayıran fonksiyon
def parse_gemma_sections(text):  # Metni başlıklara ayıran fonksiyon
    text = text.replace("*", "").replace("**", "").strip()  # Metindeki gereksiz karakterleri temizliyoruz
    sections = {  # Başlıklar için boş bir sözlük oluşturuyoruz
        "Kıyafet Analizi": "",
        "Renk Uyumu": "",
        "Kombin Önerisi": "",
        "Kombin Yorumu": "",
        "Alternatif Kombin": "",
    }
    pattern = r"(Kıyafet Analizi|Renk Uyumu|Kombin Önerisi|Kombin Yorumu|Alternatif Kombin)\s*:\s*(.*?)\s*(?=(Kıyafet Analizi|Renk Uyumu|Kombin Önerisi|Kombin Yorumu|Alternatif Kombin|$))"  # Başlıkları ayırmak için düzenli ifade deseni
    matches = re.findall(pattern, text, re.DOTALL)  # Metni başlıklara göre ayırıyoruz
    for (heading, content, _) in matches:  # Her başlık ve içeriği için
        sections[heading] = content.strip()  # İçeriği sözlüğe ekliyoruz
    return sections  # Başlıklar ve içeriklerini döndürüyoruz

# Yorum metnini başlıklara ayıran fonksiyon
def extract_yorum(answer, title="Kıyafet Yorumu"):  # Yorum metnini başlıklara ayıran fonksiyon
    answer = answer.replace("*", "").replace("**", "").strip()  # Metindeki gereksiz karakterleri temizliyoruz
    try:
        yorum_text = answer.split(title + ":")[1].strip()  # Başlığa göre metni ayırıyoruz
    except:
        yorum_text = answer.strip()  # Hata durumunda tüm metni alıyoruz
    lines = [line.strip("•-– ") for line in yorum_text.splitlines() if line.strip()]  # Her satırı temizliyoruz
    return "\n".join(f"- {line}" for line in lines)  # Satırları birleştirip döndürüyoruz

# Anasayfa için route tanımlıyoruz
@app.route('/')  # Anasayfa rotası
def index():  # Anasayfa fonksiyonu
    return render_template('index.html')  # index.html dosyasını render ediyoruz

# Kombin oluşturma endpointi
@app.route('/kombin-olustur', methods=['POST'])  # Kombin oluşturma rotası
def kombin_olustur():  # Kombin oluşturma fonksiyonu
    # Dosyaların varlığını kontrol ediyoruz
    if 'files[]' not in request.files:  # Dosyaların varlığını kontrol ediyoruz
        return jsonify({"error": "Dosya bulunamadı"}), 400  # Hata mesajı döndürüyoruz

    files = request.files.getlist('files[]')  # Dosyaları alıyoruz
    if len(files) < 2:  # En az iki dosya kontrolü
        return jsonify({"error": "En az 2 kıyafet resmi yüklemelisiniz."}), 400  # Hata mesajı döndürüyoruz

    # Tarz ve mevsim bilgilerini alıyoruz
    tarz = request.form.get('tarz', 'Casual')  # Tarz bilgisini alıyoruz
    mevsim = request.form.get('mevsim', 'İlkbahar')  # Mevsim bilgisini alıyoruz
    image_descriptions = []  # Görsel açıklamaları için liste oluşturuyoruz

    # Her bir dosya için açıklama üretiyoruz
    for file in files:  # Her dosya için
        if file and allowed_file(file.filename):  # Dosya uzantısını kontrol ediyoruz
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"  # Dosya ismini güvenli hale getiriyoruz
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Dosya yolunu oluşturuyoruz
            file.save(filepath)  # Dosyayı kaydediyoruz
            description = blip_generate(filepath)  # Açıklama üretiyoruz
            image_descriptions.append(description)  # Açıklamayı listeye ekliyoruz

    # GEMMA API'sine gönderilecek promptu hazırlıyoruz
    gemma_prompt = f"""
Sen moda uzmanısın. Şu kıyafetleri "{tarz}" tarzda ve "{mevsim}" mevsimine uygun analiz et:
{', '.join(image_descriptions)}

Lütfen aşağıdaki başlıklara 1-2 cümlelik profesyonel ve sade cevaplar ver. Başlıkları belirt:
Kıyafet Analizi:
Renk Uyumu:
Kombin Önerisi:
Kombin Yorumu:
Alternatif Kombin:
"""  # GEMMA API'sine gönderilecek metni hazırlıyoruz

    # GEMMA API'sine istek gönderiyoruz
    GEMMA_API_URL = os.environ.get("GEMMA_API_URL", "http://localhost:11434/api/generate")  # GEMMA API URL'sini alıyoruz
    response = requests.post(  # API'ye POST isteği gönderiyoruz
        GEMMA_API_URL,
        json={
            "model": "gemma3:12b",
            "prompt": gemma_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 200
            }
        }
    )

    # API yanıtını işliyoruz
    if response.status_code == 200:  # Yanıt başarılıysa
        gemma_answer = response.json().get('response', '')  # Yanıtı alıyoruz
        parsed_sections = parse_gemma_sections(gemma_answer)  # Yanıtı başlıklara ayırıyoruz
        gemma2_response = {  # Yanıtı sözlük formatına çeviriyoruz
            "kombin_analizi": parsed_sections["Kıyafet Analizi"],
            "renk_uyumu": parsed_sections["Renk Uyumu"],
            "kombin_onerisi": parsed_sections["Kombin Önerisi"],
            "kombin_yorumu": parsed_sections["Kombin Yorumu"],
            "alternatif_kombin": parsed_sections["Alternatif Kombin"]
        }
    else:  # Yanıt başarısızsa
        gemma2_response = {key: "Model yanıt vermedi" for key in [  # Hata mesajı oluşturuyoruz
            "kombin_analizi", "renk_uyumu", "kombin_onerisi", "kombin_yorumu", "alternatif_kombin"
        ]}

    return jsonify({  # Yanıtı JSON formatında döndürüyoruz
        "blip_aciklamalari": image_descriptions,
        "gemma2_cevabi": gemma2_response
    })

# Kıyafet Yorumla endpointi
@app.route('/yorum-olustur', methods=['POST'])  # Yorum oluşturma rotası
def yorum_olustur():  # Yorum oluşturma fonksiyonu
    yorum_text = request.form.get('yorum', '')  # Yorum metnini alıyoruz
    file = request.files.get('yorumFile')  # Yorum dosyasını alıyoruz
    description = ""  # Açıklama için boş bir değişken oluşturuyoruz
    if file and allowed_file(file.filename):  # Dosya uzantısını kontrol ediyoruz
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"  # Dosya ismini güvenli hale getiriyoruz
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Dosya yolunu oluşturuyoruz
        file.save(filepath)  # Dosyayı kaydediyoruz
        description = blip_generate(filepath)  # Açıklama üretiyoruz

    # Tarz, mevsim, cinsiyet ve yaş aralığı bilgilerini alıyoruz
    tarz = request.form.get('yorumTarz', 'Casual')  # Tarz bilgisini alıyoruz
    mevsim = request.form.get('yorumMevsim', 'İlkbahar')  # Mevsim bilgisini alıyoruz
    cinsiyet = request.form.get('cinsiyet', 'Diğer')  # Cinsiyet bilgisini alıyoruz
    yas_araligi = request.form.get('yasAraligi', '0-5')  # Yaş aralığı bilgisini alıyoruz

    full_description = " ".join([yorum_text, description]).strip()  # Yorum ve açıklamayı birleştiriyoruz
    if not full_description:  # Eğer açıklama yoksa
        return jsonify({"error": "Yorum veya görsel gerekli"}), 400  # Hata mesajı döndürüyoruz

    # GEMMA API'sine gönderilecek yorum promptunu hazırlıyoruz
    gemma_yorum_prompt = f"""
Sen deneyimli bir moda stil danışmanısın. Kullanıcının tanımladığı veya görsel olarak paylaştığı kıyafeti analiz et. 
Yorumun hem kıyafetin genel şıklığını hem de giyilebilecek ortamı ve küçük önerileri içersin.

Kullanıcı bilgileri:
- Tarz: {tarz}
- Mevsim: {mevsim}
- Cinsiyet: {cinsiyet}
- Yaş Aralığı: {yas_araligi}

seçilen cinsiyet ve mevsime göre önerilerde bulun:
- Eğer cinsiyet 'Erkek' ise, daha maskülen, erkeklik gösteren, bluz gibi şeyler olmayan ve şık bir kombin öner.
- Eğer cinsiyet 'Kadın' ise, daha feminen ve kadınsı ve şık bir kombin öner.
- Mevsime uygun kumaş ve renk seçimleri yap.

Yalnızca şu formatta cevap ver:
Kıyafet Yorumu:
- Kıyafeti kısa ve profesyonel bir dille değerlendir.
- Uygun ortam/tören için görüş belirt.
- Dilersen 1 küçük dokunuş öner.

Maddeleri açık ve anlaşılır yaz. Süs karakterleri kullanma.
"""  # Yorum promptunu hazırlıyoruz

    # GEMMA API'sine istek gönderiyoruz
    GEMMA_API_URL = os.environ.get("GEMMA_API_URL", "http://localhost:11434/api/generate")  # GEMMA API URL'sini alıyoruz
    response = requests.post(  # API'ye POST isteği gönderiyoruz
        GEMMA_API_URL,
        json={
            "model": "gemma2",
            "prompt": gemma_yorum_prompt + "\n" + full_description,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 150
            }
        }
    )

    # API yanıtını işliyoruz
    if response.status_code == 200:  # Yanıt başarılıysa
        gemma_yorum_answer = response.json().get('response', '')  # Yanıtı alıyoruz
        yorum_result = extract_yorum(gemma_yorum_answer)  # Yanıtı işliyoruz
    else:  # Yanıt başarısızsa
        yorum_result = "Model yanıt vermedi."  # Hata mesajı oluşturuyoruz

    return jsonify({  # Yanıtı JSON formatında döndürüyoruz
        "yorum": yorum_result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
# Flask uygulamasını başlatıyoruz
if __name__ == '__main__':  # Ana modül olarak çalıştırıldığında
    app.run(debug=True, port=8501) # Flask uygulamasını başlatıyoruz

'''