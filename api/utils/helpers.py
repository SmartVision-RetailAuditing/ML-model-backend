# utils/helpers.py
import re


def turkce_karakter_temizle(metin):
    if not metin: return metin
    degisim = {'ı': 'i', 'İ': 'I', 'I': 'I', 'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U', 'ş': 's', 'Ş': 'S', 'ö': 'o',
               'Ö': 'O', 'ç': 'c', 'Ç': 'C'}
    for tr, eng in degisim.items(): metin = metin.replace(tr, eng)
    return metin


def standartlastir(metin):
    if not metin: return ""
    metin = metin.replace('I', 'ı').replace('İ', 'i').lower()
    degisimler = {'ş': 's', 'ç': 'c', 'ğ': 'g', 'ü': 'u', 'ö': 'o', 'ı': 'i'}
    for tr, eng in degisimler.items():
        metin = metin.replace(tr, eng)
    return metin.strip()


def urun_detay_bul(aranan_isim, katalog):
    # Temizleme
    clean_name = re.sub(r'(_v?\d+)$', '', aranan_isim, flags=re.IGNORECASE)
    aranan_marka = standartlastir(clean_name)

    for key, details in katalog.items():
        katalogdaki_marka = standartlastir(details.get("brand", ""))
        katalogdaki_key = standartlastir(key)

        if katalogdaki_marka == aranan_marka or katalogdaki_key == aranan_marka:
            return details

    return None