"""Gesture -> spoken phrase in English / Hindi / Marathi."""
import os

PHRASES = {
    "hello":      {"en": "Hello",                "hi": "नमस्ते",              "mr": "नमस्कार"},
    "help":       {"en": "I need help",          "hi": "मुझे मदद चाहिए",       "mr": "मला मदत हवी आहे"},
    "need_water": {"en": "I need water",         "hi": "मुझे पानी चाहिए",      "mr": "मला पाणी हवे आहे"},
    "emergency":  {"en": "Emergency! Please call for help immediately",
                   "hi": "आपातकाल! कृपया तुरंत मदद बुलाइए",
                   "mr": "आणीबाणी! कृपया ताबडतोब मदत बोलवा"},
    "need_food":  {"en": "I need food",          "hi": "मुझे खाना चाहिए",      "mr": "मला जेवण हवे आहे"},
        "yes":         {"en": "Yes",                      "hi": "हाँ",                          "mr": "होय"},
    "no":          {"en": "No",                       "hi": "नहीं",                         "mr": "नाही"},
    "thank_you":   {"en": "Thank you",                "hi": "धन्यवाद",                      "mr": "धन्यवाद"},
    "sorry":       {"en": "Sorry",                    "hi": "माफ़ कीजिए",                   "mr": "माफ करा"},
    "please":      {"en": "Please",                   "hi": "कृपया",                        "mr": "कृपया"},
    "stop":        {"en": "Stop",                     "hi": "रुकिए",                        "mr": "थांबा"},
    "toilet":      {"en": "I need the toilet",        "hi": "मुझे शौचालय जाना है",          "mr": "मला शौचालयाला जायचे आहे"},
    "pain":        {"en": "I am in pain",             "hi": "मुझे दर्द हो रहा है",          "mr": "मला वेदना होत आहेत"},
    "call_family": {"en": "Please call my family",    "hi": "कृपया मेरे परिवार को फ़ोन करें", "mr": "कृपया माझ्या कुटुंबाला फोन करा"},
    "good_ok":     {"en": "I am fine",                "hi": "मैं ठीक हूँ",                   "mr": "मी ठीक आहे"},
}

LANGS = {"en": "English", "hi": "Hindi", "mr": "Marathi"}


def phrase(gesture, lang="en"):
    return PHRASES[gesture][lang]


def speak(gesture, lang="en", out_dir="outputs/audio", play=False):
    """Create an mp3 for the phrase (gTTS, needs internet). Returns the file path,
    or None if TTS is unavailable (the text is still shown by the caller)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{gesture}_{lang}.mp3")
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        try:
            from gtts import gTTS
            tmp = path + ".part"
            gTTS(text=phrase(gesture, lang), lang=lang).save(tmp)
            os.replace(tmp, path)                   # only keep complete files
        except Exception as e:                      # offline / blocked
            print(f"[tts] unavailable ({e.__class__.__name__}); text only")
            if os.path.exists(path + ".part"): os.remove(path + ".part")
            return None
    if play:
        _play(path)
    return path


def _play(path):
    import shutil, subprocess
    for player in ("mpg123", "ffplay", "afplay", "play"):
        if shutil.which(player):
            args = [player] + (["-nodisp", "-autoexit", "-loglevel", "quiet"] if player == "ffplay" else []) + [path]
            subprocess.run(args, check=False)
            return
