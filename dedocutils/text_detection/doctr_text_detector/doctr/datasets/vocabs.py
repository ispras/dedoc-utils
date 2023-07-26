import string
from typing import Dict

__all__ = ['VOCABS']


VOCABS: Dict[str, str] = {
    'digits': string.digits,
    'ascii_letters': string.ascii_letters,
    'punctuation': string.punctuation,
    'currency': '£€¥¢฿',
    'ancient_greek': 'αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ',
    'arabic_letters': 'ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىي',
    'persian_letters': 'پچڢڤگ',
    'hindi_digits': '٠١٢٣٤٥٦٧٨٩',
    'arabic_diacritics': 'ًٌٍَُِّْ',
    'arabic_punctuation': '؟؛«»—'
}

VOCABS['latin'] = VOCABS['digits'] + VOCABS['ascii_letters'] + VOCABS['punctuation']
VOCABS['english'] = VOCABS['latin'] + '°' + VOCABS['currency']
VOCABS['legacy_french'] = VOCABS['latin'] + '°' + 'àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ' + VOCABS['currency']
VOCABS['french'] = VOCABS['english'] + 'àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ'
VOCABS['portuguese'] = VOCABS['english'] + 'áàâãéêíïóôõúüçÁÀÂÃÉÊÍÏÓÔÕÚÜÇ'
VOCABS['spanish'] = VOCABS['english'] + 'áéíóúüñÁÉÍÓÚÜÑ' + '¡¿'
VOCABS['german'] = VOCABS['english'] + 'äöüßÄÖÜẞ'
VOCABS['arabic'] = (VOCABS['digits'] + VOCABS['hindi_digits'] + VOCABS['arabic_letters'] + VOCABS['persian_letters'] +
                    VOCABS['arabic_diacritics'] + VOCABS['arabic_punctuation'] + VOCABS['punctuation'])
