"""
AI Video Analyzer — wersja pod Streamlit Community Cloud
=========================================================

Co robi aplikacja:
  1. Przyjmuje plik wideo/audio.
  2. Wyciąga z niego samo audio (FFmpeg) i transkrybuje je za pomoca Whisper-1 od OpenAI.
  3. Generuje podsumowanie, punkty i tagi.
  4. Eksportuje napisy SRT i pokazuje szacunkowe koszty (w dolarach).

"""

import copy  # deepcopy: bezpieczne kopiowanie domyślnych wartości stanu sesji
import os  # operacje na ścieżkach i usuwanie plików tymczasowych
import re  # wyszukiwanie nazwy kodeka na liście kodeków ffmpeg
import shutil  # which() sprawdza, czy ffmpeg jest w PATH; copyfileobj() kopiuje strumieniowo
import subprocess  # uruchamianie ffmpeg/ffprobe jako osobnych procesów
import tempfile  # pliki i katalogi tymczasowe (dysk serwera, nie RAM)
import streamlit as st
from openai import OpenAI

import json
from functools import lru_cache  # cache: kodeki sprawdzamy raz, nie przy każdym klipie


# KONFIGURACJA STRONY
# musi być na pierwszym miejscu inaczej Streamlit rzuci wyjątek.
st.set_page_config(page_title="AI Video Analyzer", page_icon="🎬", layout="wide")


# STAŁE

# UWAGA: ceny poniżej są orientacyjne — sprawdź aktualny cennik na openai.com.
WHISPER_MODEL = "whisper-1"
GPT_MODEL = "gpt-4o-mini"
WHISPER_USD_PER_MINUTE = 0.006
GPT_USD_PER_1M_INPUT = 0.15
GPT_USD_PER_1M_OUTPUT = 0.60

# Whisper API przyjmuje pliki do 25 MB. Zamiast wysyłać cały film, wyciągamy
# audio i tniemy je na kawałki po 10 minut. Mono, 16 kHz, 48 kb/s (AAC) daje ok. 3,6 MB
# na kawałek — wielki zapas poniżej limitu, a jakość rozpoznawania mowy
# praktycznie się nie pogarsza.

CHUNK_SECONDS = 600

# Rozszerzenia, które traktujemy jako "tylko audio" (bez obrazu).
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav"}

# Etykiety wyboru języka -> kod ISO przekazywany do Whispera.
# Podanie języka wprost poprawia jakość i skraca czas względem autodetekcji.
LANGUAGES = {
    "Wykryj automatycznie": None,
    "Polski": "pl",
    "Angielski": "en",
    "Niemiecki": "de",
}


# Ile ostatnich wiadomości z czatu wysyłamy do modelu. Cała transkrypcja i tak
# jest w system prompcie, więc bez limitu historia rosłaby w nieskończoność
# i każde pytanie kosztowałoby coraz więcej tokenów.
MAX_HISTORY_MESSAGES = 10


# KLUCZ API I KLIENT OPENAI

def get_api_key():
    """Zwraca klucz API z pierwszego dostępnego źródła.

    Kolejność: pole w sidebarze -> st.secrets (Community Cloud) -> zmienna środowiskowa.

     Dlaczego tak, a nie samo `OpenAI()` na górze pliku?
        `OpenAI()` rzuca wyjątek już przy starcie, gdy nie ma klucza — aplikacja
        wywala się, zanim użytkownik zdąży cokolwiek wpisać. Tutaj klienta tworzymy
        dopiero, gdy klucz faktycznie mamy.
        """
    typed = st.session_state.get("api_key_input", "").strip()
    if typed:
        return typed
    try:
        # st.secrets rzuca wyjątek, gdy nie ma pliku secrets (np. lokalnie)
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.environ.get("OPENAI_API_KEY")


# SIDEBAR

with st.sidebar:
    st.header("Ustawienia")
    # Pole jest typu password, żeby klucz nie był widoczny na ekranie.
    st.text_input(
        "Klucz API OpenAI (opcjonalnie)",
        type="password",
        key="api_key_input",
        help="Jeśli puste, użyję klucza z aplikacji.",
    )
    language_label = st.selectbox("Język nagrania", list(LANGUAGES.keys()))
    language_code = LANGUAGES[language_label]
    st.markdown("---")
    st.info(
        f"Modele: **{WHISPER_MODEL}** (transkrypcja) i **{GPT_MODEL}** (analiza). "
        "Obróbka audio/wideo odbywa się lokalnie przez **FFmpeg**."
    )

api_key = get_api_key()
if not api_key:
    st.title("AI Video Analyzer")
    st.warning(
        "Brak klucza API. Wpisz go w panelu bocznym"
    )
    # st.stop() przerywa wykonanie skryptu w tym miejscu. 
        # Lepsze to niż zagnieżdżanie całej aplikacji w jednym wielkim `if api_key:`.
    st.stop()

client = OpenAI(api_key=api_key)

# Sprawdzamy ffmpeg od razu, żeby błąd był czytelny. Na Community Cloud ffmpeg
# pojawia się tylko wtedy, gdy w repozytorium jest plik packages.txt z linią "ffmpeg".
if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    st.error(
        "Nie znaleziono FFmpeg. Dodaj do repozytorium plik `packages.txt` z jedną "
        "linią: `ffmpeg` i uruchom aplikację ponownie (lokalnie: zainstaluj ffmpeg)."
    )
    st.stop()


# STAN SESJI


DEFAULTS = {
    "transcript": None,  # dict: {"text", "segments", "duration"}
    "summary": None,  # tekst podsumowania od GPT
    "tokens": {"input": 0, "output": 0},  # zużycie tokenów GPT
    "whisper_seconds": 0.0,  # ile sekund audio przetworzył Whisper (do kosztów)
    "media_path": None,  # ścieżka do wgranego pliku na dysku serwera
    "media_sig": None,  # "podpis" pliku (nazwa:rozmiar) do wykrycia nowego uploadu
}


DEFAULTS.update({"chat_history": [], "clips": []})


for _key, _value in DEFAULTS.items():
    # deepcopy, bo wartości domyślne to słowniki/listy — bez kopii wszystkie
    # sesje współdzieliłyby ten sam obiekt.
    st.session_state.setdefault(_key, copy.deepcopy(_value))


# FUNKCJE POMOCNICZE: PLIKI I FFMPEG

def safe_remove(path):
    """Usuwa plik, a jeśli go nie ma — ignoruje.

    Pliki tymczasowe czyścimy, bo dysk na Community Cloud jest ograniczony
    i nie sprząta automatycznie, dopóki aplikacja żyje.
    """
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


def reset_session_for_new_file():
    """Czyści wyniki poprzedniego pliku (i jego pliki tymczasowe).

    Bez tego po wgraniu drugiego filmu użytkownik widziałby transkrypcję
    pierwszego — błąd.
    """
    safe_remove(st.session_state.media_path)
    #sprzątanie wyciętych klipów
    for clip in st.session_state.clips:
        safe_remove(clip["path"])
    for key, value in DEFAULTS.items():
        st.session_state[key] = copy.deepcopy(value)


def save_upload_to_disk(uploaded):
    """Zapisuje wgrany plik na dysk i zwraca ścieżkę.

    Dlaczego na dysk, skoro Streamlit trzyma plik w RAM?
    FFmpeg to osobny program i potrzebuje prawdziwej ścieżki do pliku.
    Używamy shutil.copyfileobj (kopiowanie strumieniowe), a nie
    `uploaded.getvalue()`, bo getvalue() tworzy w pamięci DRUGĄ pełną kopię
    pliku — przy kilkusetmegabajtowym filmie na darmowym hostingu to ryzyko.
    """
    suffix = os.path.splitext(uploaded.name)[1].lower() or ".mp4"
    uploaded.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(uploaded, tmp)
        return tmp.name


def run_cmd(cmd):
    """Uruchamia polecenie i zwraca jego stdout; przy błędzie rzuca czytelny wyjątek.

    Używam listy argumentów (a nie jednego stringa z shell=True), bo wtedy
    nazwy plików ze spacjami lub dziwnymi znakami nie psują polecenia i nie ma
    ryzyka wstrzyknięcia komend.
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Ogon stderr zawiera zwykle faktyczną przyczynę; początek to baner ffmpeg.
        raise RuntimeError(f"{cmd[0]} zakończył się błędem:\n{result.stderr[-1500:]}")
    return result.stdout


def get_duration(path):
    """Zwraca długość pliku audio/wideo w sekundach (przez ffprobe)."""
    out = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    return float(out.strip())


def extract_audio_chunks(media_path, workdir):
    """Wyciąga audio z pliku i tnie je na kawałki po CHUNK_SECONDS sekund.

    Robi to jednym wywołaniem ffmpeg (muxer "segment"), więc plik źródłowy
    czyta raz. Zwraca posortowaną listę ścieżek do kawałków .m4a.
    """
    pattern = os.path.join(workdir, "chunk_%03d.m4a")
    run_cmd([
        "ffmpeg", "-y", "-i", media_path,
        "-vn",                    # odrzuć obraz — do transkrypcji zbędny
        "-ac", "1",               # mono
        "-ar", "16000",           # 16 kHz
        # Kodek AAC (natywny koder FFmpeg) zamiast mp3/libmp3lame: libmp3lame jest
        # biblioteką opcjonalną i część buildów (np. na Windowsie) jest bez niej
        # kompilowana -> błąd "Unknown encoder". Koder aac jest wbudowany w każdy
        # build, a Whisper przyjmuje m4a, więc działa wszędzie.
        "-c:a", "aac",
        "-b:a", "48k",            # niski bitrate = mały plik (ok. 3,6 MB / 10 min)
        "-segment_format", "mp4",  # każdy kawałek to osobny, samodzielny plik m4a
        "-f", "segment",          # dziel wyjście na części...
        "-segment_time", str(CHUNK_SECONDS),  # co tyle sekund
        "-reset_timestamps", "1",  # każdy kawałek zaczyna się od 0
        pattern,
    ])
    return sorted(
        os.path.join(workdir, f) for f in os.listdir(workdir) if f.startswith("chunk_")
    )


#FORMATY CZASU

def format_srt_time(seconds):
    """Sekundy -> format SRT `HH:MM:SS,mmm`.

    Liczymy na milisekundach (int), bo działania na floatach potrafią dać
    np. 59.9999 -> "00:00:59,999" zamiast "00:01:00,000".
    """
    total_ms = int(round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


#czytelny znacznik czasu dla czatu i klipów)
def format_mmss(seconds):
    """Sekundy -> `MM:SS` (albo `H:MM:SS` dla materiałów powyżej godziny).

    Ten format jest krótszy niż SRT, a więc oszczędza tokeny w prompcie i jest
    czytelny dla użytkownika w odpowiedziach czatu.
    """
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"




#OPENAI

def ask_gpt(messages, **kwargs):
    """Jedno miejsce, przez które idą wszystkie zapytania do GPT.

    Po co osobna funkcja zamiast bezpośrednich wywołań `client.chat.completions`?
          * zliczanie tokenów robimy raz, a nie kopiujemy 3-4 razy
    """
    response = client.chat.completions.create(model=GPT_MODEL, messages=messages, **kwargs)
    st.session_state.tokens["input"] += response.usage.prompt_tokens
    st.session_state.tokens["output"] += response.usage.completion_tokens
    return response.choices[0].message.content


def _get(obj, name):
    """Czyta pole z obiektu SDK albo ze słownika.

    W różnych wersjach biblioteki `openai` segmenty Whispera bywają raz
    obiektami (segment.start), raz słownikami (segment["start"]). Ta funkcja
    sprawia, że kod działa w obu przypadkach.
    """
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def transcribe_media(media_path, language):
    """Transkrybuje plik: wyciąga audio, tnie na kawałki, wysyła do Whispera.
    
    Kluczowy szczegół: timestampy w kolejnych kawałkach liczone są od zera,
    więc do każdego dodajemy `offset` = suma długości poprzednich kawałków.
    Offset liczymy z faktycznej długości (ffprobe), a nie z CHUNK_SECONDS.
    """
    workdir = tempfile.mkdtemp(prefix="chunks_")
    try:
        chunks = extract_audio_chunks(media_path, workdir)
        if not chunks:
            raise RuntimeError("Nie udało się wyciągnąć ścieżki audio z pliku.")

        segments, texts, offset = [], [], 0.0
        granularities = ["segment"]  # segmenty = czasy zdań/fraz; wystarczą do napisów SRT
        #słowa z czasami do precyzyjnego czatu
        # Dla czatu potrzebujemy dokładniejszych czasów: segment to kilka-kilkanaście
        # sekund pod jednym znacznikiem, więc "kiedy pada słowo X" wychodziło z błędem
        # nawet o 5-10 s. Whisper zwraca czasy każdego słowa w tym samym zapytaniu
        # (bez dodatkowego kosztu), więc prosimy o oba poziomy naraz.
        granularities.append("word")
        words = []
        
        progress = st.progress(0.0, text="Transkrypcja...")

        for i, chunk_path in enumerate(chunks):
            # language podajemy tylko, gdy użytkownik go wybrał (None = autodetekcja).
            extra = {"language": language} if language else {}
            with open(chunk_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=f,
                    response_format="verbose_json",  # tylko ten format zwraca segmenty z czasami
                    timestamp_granularities=granularities,
                    **extra,
                )
            texts.append(resp.text.strip())
            for seg in getattr(resp, "segments", None) or []:
                segments.append({
                    "start": float(_get(seg, "start")) + offset,
                    "end": float(_get(seg, "end")) + offset,
                    "text": _get(seg, "text").strip(),
                })
            #zbieranie słów z przesunięciem o offset kawałka
            for w in getattr(resp, "words", None) or []:
                words.append({
                    "start": float(_get(w, "start")) + offset,
                    "end": float(_get(w, "end")) + offset,
                    "word": _get(w, "word").strip(),
                })
            
            offset += get_duration(chunk_path)
            progress.progress((i + 1) / len(chunks), text=f"Transkrypcja: {i + 1}/{len(chunks)}")

        progress.empty()
        result = {"text": " ".join(texts), "segments": segments, "duration": offset}
        
        result["words"] = words
        return result
    finally:
        # finally = sprzątamy kawałki audio także wtedy, gdy coś poszło nie tak.
        shutil.rmtree(workdir, ignore_errors=True)


def generate_summary(text):
    """Prosi GPT o podsumowanie, punkty i tagi na podstawie pełnego tekstu."""
    prompt = (
        "Na podstawie poniższej transkrypcji przygotuj:\n"
        "1. Krótkie podsumowanie (3-5 zdań).\n"
        "2. Główne punkty / wnioski w punktach.\n"
        "3. Propozycję 3-5 tagów.\n\nTranskrypcja:\n" + text
    )
    return ask_gpt([
        {"role": "system", "content": "Jesteś pomocnym asystentem analitycznym. Odpowiadaj po polsku."},
        {"role": "user", "content": prompt},
    ])


def build_srt(segments):
    """Składa zawartość pliku .srt z listy segmentów."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n{seg['text']}\n"
        )
    return "\n".join(blocks)


#FUNKCJE CZATU I KLIPÓW

def group_words(words, max_span=5.0, max_words=15):
    """Grupuje słowa w krótkie linie (ok. 5 s) i zwraca listę (start, tekst).

    Dlaczego grupujemy, a nie dajemy znacznika przy każdym słowie? Znacznik
    na każde słowo kilkukrotnie zwiększyłby liczbę tokenów (i koszt każdego
    pytania). Linie po ok. 5 s to kompromis: dokładność rzędu kilku sekund
    przy umiarkowanym koszcie.
    """
    lines, current = [], []
    for w in words:
        if current and (w["end"] - current[0]["start"] > max_span or len(current) >= max_words):
            lines.append(current)
            current = []
        current.append(w)
    if current:
        lines.append(current)
    return [(ln[0]["start"], " ".join(x["word"] for x in ln)) for ln in lines]


def build_timestamped_transcript(transcript):
    """Transkrypcja w postaci `[MM:SS] tekst` — jedna linia na ok. 5 s mowy.

    Gdy mamy czasy słów, używamy ich (dokładniej). Gdy ich nie ma (np. starszy
    wynik w sesji), wracamy do segmentów, żeby nic się nie wysypało.
    Model widzi prawdziwe czasy i może je tylko przepisywać, nie zgadywać/wymyślać.
    """
    words = transcript.get("words")
    if words:
        pairs = group_words(words)
    else:
        pairs = [(s["start"], s["text"]) for s in transcript["segments"]]
    return "\n".join(f"[{format_mmss(start)}] {text}" for start, text in pairs)


def answer_chat_question(question):
    """Odpowiada na pytanie o film, uwzględniając historię rozmowy.

    Transkrypcję umieszczamy w wiadomości systemowej (raz), a historię rozmowy
    dokładamy jako zwykłe wiadomości — dzięki temu pytania typu
    "a co mówi dalej na ten temat?" - działają.
    """
    transcript = st.session_state.transcript
    system = (
        "Jesteś ekspertem analizującym film wyłącznie na podstawie poniższej transkrypcji. "
        "Każda linia obejmuje ok. 5 sekund mowy i zaczyna się od znacznika czasu [MM:SS]. "
        "Gdy użytkownik pyta, kiedy zaczyna się temat, podaj znacznik linii, w której "
        "temat faktycznie pada po raz pierwszy (nie linii wcześniejszej, z wprowadzeniem). "
        "Gdy wskazujesz moment w filmie, podawaj znaczniki czasu wyłącznie takie, "
        "jakie występują w transkrypcji — niczego nie zgaduj ani nie wyliczaj. "
        "Jeśli odpowiedzi nie ma w transkrypcji, napisz to wprost.\n\n"
        "TRANSKRYPCJA:\n" + build_timestamped_transcript(transcript)
    )
    history = st.session_state.chat_history[-MAX_HISTORY_MESSAGES:]
    messages = [{"role": "system", "content": system}] + history + [
        {"role": "user", "content": question}
    ]
    return ask_gpt(messages)


def suggest_clips(n_clips, max_len):
    """Prosi GPT o wskazanie najciekawszych fragmentów i zwraca zwalidowaną listę.

    Dlaczego response_format={"type": "json_object"}? Gwarantuje poprawny JSON,
    więc nie musimy wyłuskiwać go regexem z tekstu. Ten tryb wymaga, by w
    prompcie padło słowo "JSON" i zwraca obiekt (nie listę), dlatego prosimy
    o strukturę {"clips": [...]}.
    """
    transcript = st.session_state.transcript
    duration = transcript["duration"]
    # Tu czasy podajemy w sekundach z ułamkiem (a nie MM:SS), bo od razu
    # trafiają do ffmpeg i model nie musi niczego przeliczać.
    lines = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in transcript["segments"]
    )
    prompt = (
        f"Wybierz maksymalnie {n_clips} najciekawszych, samodzielnie zrozumiałych fragmentów "
        f"do klipów na social media. Każdy ma trwać od 10 do {max_len} sekund. "
        "Czasy start i end (w sekundach) bierz z transkrypcji poniżej. "
        'Odpowiedz wyłącznie obiektem JSON w formacie: '
        '{"clips": [{"title": "krótki tytuł", "start": 12.5, "end": 40.0}]}\n\n'
        "TRANSKRYPCJA:\n" + lines
    )
    raw = ask_gpt([{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    data = json.loads(raw)

    valid = []
    for item in data.get("clips", []):
        try:
            start = max(0.0, float(item["start"]))
            end = min(float(item["end"]), duration)
            title = str(item.get("title", "Klip"))
        except (KeyError, TypeError, ValueError):
            continue  # pomijamy zniekształcony wpis zamiast wywalać całą operację
        if end - start < 3:
            continue
        end = min(end, start + max_len)  # twardy limit długości
        valid.append({"title": title, "start": start, "end": end})
    return valid[:n_clips]

#Ta funkcja uruchamia ffmpeg -encoders i sprawdza, który kodek jest dostępny. 
#Bez cache’u robiłaby to za każdym razem, gdy ją wywołujemy, 
#czyli przy każdym wycinanym klipie i przy każdym odświeżeniu zakładki Klipy.
@lru_cache(maxsize=1)
def pick_video_encoder():
    """Wybiera kodek wideo dostępny w buildzie FFmpeg.

    Pytamy ffmpeg o listę kodeków i bierzemy pierwszy działający
    z listy preferencji. Wynik cache'ujemy (lru_cache), bo lista się nie zmienia.

    Kolejność (od najlepszego):
      1. libx264     - najlepsza jakość i szybkość, domyślny na Linuksie (Community Cloud)
      2. libopenh264 - H.264 na licencji BSD, często dostępny w buildach bez GPL
      3. h264_mf     - H.264 z Windows Media Foundation (tylko Windows)
      4. copy        - bez przekodowania; działa zawsze, ale tnie tylko na klatkach
                       kluczowych (początek klipu może się przesunąć o kilka sekund)
    """
    try:
        listing = run_cmd(["ffmpeg", "-hide_banner", "-encoders"])
    except RuntimeError:
        listing = ""
    options = [
        ("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac"]),
        ("libopenh264", ["-c:v", "libopenh264", "-b:v", "2M", "-c:a", "aac"]),
        ("h264_mf", ["-c:v", "h264_mf", "-b:v", "2M", "-c:a", "aac"]),
    ]
    for name, args in options:
        # \b = granica słowa, żeby "libx264" nie dopasowało się do "libx264rgb".
        if re.search(rf"\b{name}\b", listing):
            return {"name": name, "args": args}
    return {"name": "copy", "args": ["-c", "copy"]}


def cut_clip(media_path, start, end):
    """Wycina fragment [start, end] i zapisuje go jako nowy plik mp4. Zwraca ścieżkę.

    Preferujemy przekodowanie (patrz pick_video_encoder), bo daje cięcie co do
    klatki. Tryb "copy" jest awaryjny: szybki, ale tnie tylko na klatkach kluczowych.
    """
    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)  # mkstemp otwiera deskryptor; zamykamy, bo ffmpeg sam zapisze plik
    run_cmd([
        "ffmpeg", "-y",
        "-ss", f"{start:.2f}",
        "-i", media_path,
        "-t", f"{end - start:.2f}",
        *pick_video_encoder()["args"],
        "-avoid_negative_ts", "make_zero",  # poprawia znaczniki czasu, zwłaszcza w trybie copy
        "-movflags", "+faststart",
        out_path,
    ])
    return out_path




# INTERFEJS: UPLOAD I ANALIZA

st.title("AI Video Analyzer")
st.write("Wgraj plik, aby wygenerować transkrypcję, podsumowanie i napisy.")

uploaded = st.file_uploader(
    "Wgraj plik wideo lub audio",
    type=["mp4", "mov", "mkv", "webm", "mp3", "m4a", "wav"],
    help="Limit rozmiaru ustawiony w config.toml (server.maxUploadSize).",
)

if uploaded is None:
    st.info("Wgraj plik, aby zacząć.")
    st.stop()

# Wykrywamy nowy plik po "sygnaturze" (nazwa + rozmiar). Samo `uploaded is not None`
# nie wystarczy, bo skrypt biegnie od nowa przy każdym kliknięciu — bez podpisu
# zapisywalibyśmy ten sam film na dysk w kółko.
signature = f"{uploaded.name}:{uploaded.size}"
if st.session_state.media_sig != signature:
    reset_session_for_new_file()
    st.session_state.media_path = save_upload_to_disk(uploaded)
    st.session_state.media_sig = signature

media_path = st.session_state.media_path
is_audio = os.path.splitext(media_path)[1].lower() in AUDIO_EXTENSIONS

# Odtwarzamy ze ścieżki na dysku (nie z RAM). st.video nie obsługuje audio, stąd rozgałęzienie.
if is_audio:
    st.audio(media_path)
else:
    st.video(media_path)

if st.button("Rozpocznij analizę", type="primary"):
    st.session_state.summary = None
    try:
        with st.spinner("Transkrypcja audio (Whisper)..."):
            result = transcribe_media(media_path, language_code)
        # Transkrypcję zapisujemy zanim wywołamy GPT — gdyby podsumowanie się
        # nie udało, użytkownik nie traci już zapłaconej transkrypcji.
        st.session_state.transcript = result
        st.session_state.whisper_seconds += result["duration"]
    except Exception as exc:
        st.error(f"Błąd transkrypcji: {exc}")
    else:
        # Osobny try/except dla podsumowania: błąd GPT nie unieważnia transkrypcji.
        try:
            with st.spinner("Generowanie podsumowania (GPT)..."):
                st.session_state.summary = generate_summary(result["text"])
        except Exception as exc:
            st.warning(f"Transkrypcja gotowa, ale podsumowanie się nie udało: {exc}")

# INTERFEJS: ZAKŁADKI Z WYNIKAMI

if not st.session_state.transcript:
    st.stop()

# Zakładki budujemy ze słownika klucz -> etykieta. Dzięki temu dodanie lub
# usunięcie zakładki to jedna linijka, a reszta kodu odwołuje się
# po stabilnym kluczu (tabs["chat"]), nie po numerze, który by się przesuwał.
TAB_LABELS = {"summary": "Podsumowanie"}
# >>> PR-1 START (dodatkowe zakładki)
TAB_LABELS["chat"] = "Czat z filmem"
TAB_LABELS["clips"] = "Klipy"
# <<< PR-1 END
TAB_LABELS["srt"] = "Napisy (SRT)"
TAB_LABELS["costs"] = "Koszty"
tabs = dict(zip(TAB_LABELS, st.tabs(list(TAB_LABELS.values()))))

#Podsumowanie
with tabs["summary"]:
    st.subheader("Streszczenie i analiza")
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        # Bez tego st.markdown(None) wywalałby aplikację, gdy GPT zawiódł.
        st.info("Podsumowanie niedostępne. Uruchom analizę ponownie.")

#Zakładka: czat z filmem
#Czat
with tabs["chat"]:
    st.subheader("Zadaj pytanie o treść filmu")
    st.caption("Np. „W której minucie mowa o cenach?” albo „Jakie główne tezy padły?”")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Zadaj pytanie dotyczące filmu..."):
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Szukam odpowiedzi w transkrypcji..."):
                    answer = answer_chat_question(question)
                st.markdown(answer)
                # Do historii dopisujemy parę dopiero po udanej odpowiedzi,
                # żeby po błędzie nie zostało "wiszące" pytanie bez odpowiedzi.
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error(f"Błąd czatu: {exc}")

#Klipy
with tabs["clips"]:
    st.subheader("Generator najciekawszych momentów")
    if is_audio:
        st.info("Klipy wideo są dostępne tylko dla plików wideo.")
    else:
        if pick_video_encoder()["name"] == "copy":
            st.caption(
                "Ten FFmpeg nie ma kodeka H.264, więc klipy są cięte bez przekodowania "
                "(początek może się przesunąć o kilka sekund)."
            )
        col_a, col_b = st.columns(2)
        n_clips = col_a.slider("Liczba klipów", 1, 5, 3)
        max_len = col_b.slider("Maks. długość klipu (s)", 15, 120, 60)

        if st.button("🔍 Znajdź ciekawe momenty i wytnij klipy"):
            try:
                # Stare klipy usuwamy z dysku, zanim zrobimy nowe.
                for old in st.session_state.clips:
                    safe_remove(old["path"])
                st.session_state.clips = []

                with st.spinner("AI wybiera fragmenty..."):
                    suggestions = suggest_clips(n_clips, max_len)
                if not suggestions:
                    st.warning("Model nie wskazał żadnych poprawnych fragmentów. Spróbuj ponownie.")
                for item in suggestions:
                    with st.spinner(f"Wycinam: {item['title']}..."):
                        item["path"] = cut_clip(media_path, item["start"], item["end"])
                    st.session_state.clips.append(item)
            except Exception as exc:
                st.error(f"Nie udało się wygenerować klipów: {exc}")

        # Klipy wyświetlamy ze stanu sesji, więc nie znikają po innym kliknięciu.
        for idx, clip in enumerate(st.session_state.clips):
            st.markdown(
                f"**Klip {idx + 1}: {clip['title']}** "
                f"({format_mmss(clip['start'])} – {format_mmss(clip['end'])})"
            )
            st.video(clip["path"])
            with open(clip["path"], "rb") as f:
                st.download_button(
                    label=f"Pobierz klip {idx + 1}",
                    data=f.read(),
                    file_name=f"klip_{idx + 1}.mp4",
                    mime="video/mp4",
                    # key musi być unikalny, inaczej Streamlit zgłosi duplikat widgetu.
                    key=f"download_clip_{idx}",
                )


#Napisy
with tabs["srt"]:
    st.subheader("Napisy")
    srt_content = build_srt(st.session_state.transcript["segments"])
    st.text_area("Podgląd SRT", srt_content, height=300)
    st.download_button(
        "Pobierz napisy (.srt)", data=srt_content, file_name="napisy.srt", mime="text/plain"
    )

#Koszty
with tabs["costs"]:
    st.subheader("Zużycie i szacunkowy koszt")
    tokens_in = st.session_state.tokens["input"]
    tokens_out = st.session_state.tokens["output"]
    # Cena za 1M tokenów -> dzielimy przez 1_000_000; Whisper liczony jest za minuty audio.
    gpt_cost = tokens_in / 1e6 * GPT_USD_PER_1M_INPUT + tokens_out / 1e6 * GPT_USD_PER_1M_OUTPUT
    whisper_cost = st.session_state.whisper_seconds / 60 * WHISPER_USD_PER_MINUTE

    c1, c2, c3 = st.columns(3)
    c1.metric("Tokeny wejściowe", f"{tokens_in:,}")
    c2.metric("Tokeny wyjściowe", f"{tokens_out:,}")
    c3.metric("Audio (min)", f"{st.session_state.whisper_seconds / 60:.1f}")

    st.info(
        f"Whisper: **${whisper_cost:.4f}** | GPT: **${gpt_cost:.5f}** | "
        f"Razem: **${whisper_cost + gpt_cost:.4f}**"
    )
    st.caption("Kwoty orientacyjne, liczone ze stałych na górze pliku — zweryfikuj z cennikiem OpenAI.")
