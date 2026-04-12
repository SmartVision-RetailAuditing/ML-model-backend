import re


def turkce_karakter_temizle(metin):
    if not metin: return metin
    degisim = {'ı': 'i', 'İ': 'I', 'I': 'I', 'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U', 'ş': 's', 'Ş': 'S', 'ö': 'o',
               'Ö': 'O', 'ç': 'c', 'Ç': 'C'}
    for tr, eng in degisim.items(): metin = metin.replace(tr, eng)
    return metin


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