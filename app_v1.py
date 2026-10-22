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


# KONFIGURACJA STRONY
# musi być na pierwszym miejscu inaczej Streamlit rzuci wyjątek.
st.set_page_config(page_title="AI Video Analyzer", layout="wide")

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
    pierwszego —  błąd.
    """
    safe_remove(st.session_state.media_path)
    for key, value in DEFAULTS.items():
        st.session_state[key] = copy.deepcopy(value)


def save_upload_to_disk(uploaded):
    """Zapisuje wgrany plik na dysk i zwraca ścieżkę.

    Dlaczego na dysk, skoro Streamlit trzyma plik w RAM?
    FFmpeg to osobny program i potrzebuje prawdziwej ścieżki do pliku.
    Używamy shutil.copyfileobj (kopiowanie strumieniowe), a nie
    `uploaded.getvalue()`, bo getvalue() tworzy w pamięci drugą pełną kopię
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
        "-f", "segment",          # dziel wyjście na części
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
        progress = st.progress(0.0, text="Transkrypcja...")

        for i, chunk_path in enumerate(chunks):
            # language podajemy tylko, gdy użytkownik go wybrał (None to autodetekcja).
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
            offset += get_duration(chunk_path)
            progress.progress((i + 1) / len(chunks), text=f"Transkrypcja: {i + 1}/{len(chunks)}")

        progress.empty()
        result = {"text": " ".join(texts), "segments": segments, "duration": offset}
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
        # Transkrypcję zapisujemy znaim wywołamy GPT — gdyby podsumowanie się
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


# Napisy
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
