# DSP Image ↔ Audio Project
## ماده DSP - تحويل الصور لأصوات

### الفكرة
كل pixel في الصورة بيتحول لـ 3 نغمات جيبية مخلوطة:
- R → 440 Hz  (amplitude = R/255)
- G → 880 Hz  (amplitude = G/255)  
- B → 1760 Hz (amplitude = B/255)

كل pixel بياخد 10ms في الملف الصوتي.
عند الاسترجاع، بنحلل كل segment بـ DFT لنعرف amplitude كل frequency.

### التشغيل
```bash
pip install -r requirements.txt
python manage.py runserver
```
ثم افتح http://127.0.0.1:8000

### الـ Endpoints
- `GET  /`                  → واجهة المستخدم
- `POST /api/encode/`       → ارفع صورة، تاخد ملف صوت
- `POST /api/decode/`       → ارفع صوت، تاخد صورة (كامل)
- `POST /api/decode/stream/`→ ارفع صوت، يرجعلك pixels تدريجياً (SSE)

### البارامترات
- MAX_IMAGE_DIM = 48px (أي صورة بتتحجم لـ 48x48 أو أصغر)
- PIXEL_DURATION = 10ms
- SAMPLE_RATE = 44100 Hz
- Header = 5 × 100ms + 50ms = ~550ms (بيحفظ الأبعاد)

### ملاحظات DSP
- الـ frequencies اختُيرت بعيدة عن بعض (倍 harmonics) لتقليل الـ crosstalk
- الـ amplitude detection بيستخدم dot product مع sin و cos (Goertzel-like)
- متوسط الخطأ في الـ pixel ~3% (7/255)