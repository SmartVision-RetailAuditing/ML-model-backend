<<<<<<< HEAD
=======
# utils/helpers.py
>>>>>>> origin/development
import re


def turkce_karakter_temizle(metin):
    if not metin: return metin
    degisim = {'ı': 'i', 'İ': 'I', 'I': 'I', 'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U', 'ş': 's', 'Ş': 'S', 'ö': 'o',
               'Ö': 'O', 'ç': 'c', 'Ç': 'C'}
    for tr, eng in degisim.items(): metin = metin.replace(tr, eng)
    return metin


<<<<<<< HEAD
def urun_detay_bul(aranan_isim, katalog):
    if aranan_isim in katalog:
        d = katalog[aranan_isim]
        return turkce_karakter_temizle(d.get("brand")), turkce_karakter_temizle(d.get("product_name"))

    temiz = re.sub(r'(_v?\d+)$', '', aranan_isim, flags=re.IGNORECASE)
    if temiz in katalog:
        d = katalog[temiz]
        return turkce_karakter_temizle(d.get("brand")), turkce_karakter_temizle(d.get("product_name"))

    print(f"⚠️ [HARİTALAMA HATASI]: Model '{aranan_isim}' sınıfını buldu ama JSON kataloğunda yok!")
    return "Bilinmiyor", aranan_isim
=======
def standartlastir(metin):
    if not metin: return ""
    metin = metin.replace('I', 'ı').replace('İ', 'i').lower()
    degisimler = {'ş': 's', 'ç': 'c', 'ğ': 'g', 'ü': 'u', 'ö': 'o', 'ı': 'i'}
    for tr, eng in degisimler.items():
        metin = metin.replace(tr, eng)
    return metin.strip()


def urun_detay_bul(aranan_isim, katalog):
    # 1. Sadece dosya uzantısını temizle (.png falan kaldıysa diye)
    aranan = re.sub(r'\.(png|jpg|jpeg)$', '', aranan_isim, flags=re.IGNORECASE)
    aranan_std = standartlastir(aranan)

    # --- 1. AŞAMA: TAM İSABET (KENDİ TOPUĞUMUZA SIKMADAN) ---
    # DINO'dan gelen isimle (Örn: 153104761_ICIM_2) JSON'daki Key BİREBİR aynı mı?
    for key, details in katalog.items():
        if standartlastir(key) == aranan_std:
            return details

    # --- 2. AŞAMA: EKSİZ YEDEK ARAMA ---
    # Eğer birebir bulamadıysa, belki sonundaki "_1" fazlalıktır diye silip şansımızı deniyoruz.
    aranan_eksiz = re.sub(r'(_v?\d+)$', '', aranan, flags=re.IGNORECASE)
    aranan_eksiz_std = standartlastir(aranan_eksiz)

    for key, details in katalog.items():
        katalogdaki_key = standartlastir(key)
        katalogdaki_marka = standartlastir(details.get("brand", ""))

        # Eksiz haliyle key eşleşiyor mu?
        if katalogdaki_key == aranan_eksiz_std:
            return details

        # --- 3. AŞAMA: SON ÇARE MARKA ---
        if katalogdaki_marka == aranan_eksiz_std or katalogdaki_marka == aranan_std:
            return details

    return None
>>>>>>> origin/development
