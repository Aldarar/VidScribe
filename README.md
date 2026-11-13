# AI Video Analyzer

An end-to-end Python web application built with **Streamlit**, **FFmpeg**, and **OpenAI APIs** (`whisper-1` and `gpt-4o-mini`) designed to transcribe, summarize, query, and repurpose video and audio content.

## Key Features
- **Smart Audio Processing:** Automatically extracts, converts (mono, 16 kHz), and splits large media files into manageable chunks using `FFmpeg` to respect API limits.
- **Advanced Transcription:** Leverages OpenAI **Whisper-1** with word-level timestamps (`timestamp_granularities`) for precise temporal alignment.
- **AI-Powered Summaries & Structured Data:** Uses **GPT** to generate summaries, key takeaways, tags, and JSON-validated social media clip suggestions.
- **Interactive Chat with Video:** Context-aware Q&A system allowing users to ask questions about the video content and retrieve exact timestamps (`[MM:SS]`).
- **Automated Video Clipping:** Dynamically cuts and exports highlight clips based on AI suggestions, checking available hardware encoders (`libx264`, `libopenh264`, etc.).
- **SRT & Cost Tracking:** Generates standard subtitle (`.srt`) files and calculates real-time API token usage and estimated expenses in USD.

## Tech Stack
- **Python 3**
- **Streamlit** (Interactive dashboard & session state management)
- **OpenAI API** (`whisper-1`, `gpt-4o-mini`)
- **FFmpeg / FFprobe** (Local media handling & processing via `subprocess`)
- **Pandas / Plotly** (Data display)

## Local Installation & Running

1. Ensure **FFmpeg** is installed on your system and available in your `PATH`.
2. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt