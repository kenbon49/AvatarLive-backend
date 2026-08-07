punctuation = ["!", "?", "\u2026", ",", ".", "'", "-", "\u00bf", "\u00a1"]

zh_symbols = [
    "E", "En", "a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e",
    "ei", "en", "eng", "er", "f", "g", "h", "i", "i0", "ia", "ian",
    "iang", "iao", "ie", "in", "ing", "iong", "ir", "iu", "j", "k", "l",
    "m", "n", "o", "ong", "ou", "p", "q", "r", "s", "sh", "t", "u", "ua",
    "uai", "uan", "uang", "ui", "un", "uo", "v", "van", "ve", "vn", "w",
    "x", "y", "z", "zh", "AA", "EE", "OO",
]
en_symbols = [
    "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "er",
    "ey", "f", "g", "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng", "ow",
    "oy", "p", "r", "s", "sh", "t", "th", "uh", "uw", "V", "w", "y", "z", "zh",
]

symbols = ["_"] + sorted(set(zh_symbols + en_symbols)) + punctuation + ["SP", "UNK"]
language_id_map = {"EN": 2, "ZH_MIX_EN": 3}
language_tone_start_map = {"ZH_MIX_EN": 0, "EN": 7}
