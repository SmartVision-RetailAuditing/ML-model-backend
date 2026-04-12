import easyocr
from utils import turkce_karakter_temizle

class TextValidator:
    def __init__(self, katalog):
        # Sadece ihtiyaç anında belleğe yüklenir
        self.reader = easyocr.Reader(['tr', 'en'], gpu=False)
        # Katalogdaki markaları bir listeye alalım
        self.markalar = set([k[1].get("brand").upper() for k in katalog.items()])

    def validate(self, crop_rgb):
        # detail=1 ile güven skoru da alırız
        results = self.reader.readtext(crop_rgb, detail=1)
        for (bbox, text, prob) in results:
            txt = text.upper()
            if len(txt) < 3: continue # "1L", "%3" gibi verileri ele
            
            for m in self.markalar:
                if m in txt and prob > 0.60: # Okuma güveni %60+ ise
                    return turkce_karakter_temizle(m.title()), 0.80 # OCR güvenini 0.80 say
        return None, None