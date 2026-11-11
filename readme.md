# Youtube Video Clipper with Python

## Tools pemotongan video YouTube berbasis AI menggunakan Python, `yt-dlp`, dan Gemini API.

## Persiapan:

1. **Python 3.x**
   Unduh dan pasang Python melalui [python.org](https://www.python.org/downloads/).
2. **FFmpeg**
   Dibutuhkan untuk pemrosesan video dan audio. Pasang via Command Prompt/Terminal:
   ```bash
   winget install ffmpeg
   ```
3. Install library : pip install yt-dlp google-genai
4. Set GOOGLE API KEY : setx GEMINI_API_KEY "isi_api_key_kamu"

## Cara menggunakan:

1. Jalankan command prompt (terminal) lalu ketik python clipper.py
2. Masukkan link video youtube yang ingin di clip
3. Masukkan total jumlah clip yang ingin dibuat
4. Tunggu proses pemotongan

## Hasil luaran:

1. File video dengan format MP4
2. File audio dengan format MP3
3. File subtitle dengan format srt
