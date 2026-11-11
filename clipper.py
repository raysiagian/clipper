"""
Auto Clipper: YouTube -> Gemini pilih bagian menarik -> konfirmasi -> download per klip.

Alur:
    1. Masukkan link video
    2. Masukkan jumlah klip
    3. Gemini memberi saran bagian menarik
    4. Pilih: lanjutkan, atau cari saran lain
    5. Proses pemotongan

Hasil per klip:
    "01 - Judul.mp4"   (video)
    "01 - Judul.srt"   (subtitle)
    "01 - Judul.mp3"   (audio)

Cara pakai:
    python clipper.py
    (atau: python clipper.py "https://www.youtube.com/watch?v=XXXX" --jumlah 5)

Butuh:
    pip install yt-dlp google-genai
    ffmpeg terpasang & bisa dipanggil dari terminal
    Environment variable GEMINI_API_KEY berisi API key dari Google AI Studio
"""

import argparse
import glob
import json
import math
import re
import subprocess
import threading
import time
import logging
import shutil
from pathlib import Path

import yt_dlp
from yt_dlp.utils import download_range_func
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)

# Dicoba berurutan; kalau yang pertama gagal, lanjut ke berikutnya
MODELS = ["gemini-flash-latest", "gemini-3.1-flash-lite", "gemini-3-flash-preview"]
OUTPUT_DIR = Path(r"D:\Depkomifo Desain\ternak uang\clipper\output")
SUB_LANGS = ["id", "en"]  # urutan bahasa subtitle YouTube yang dicoba
MAX_TINGGI = 1080  # resolusi maksimal video (turunkan ke 720 kalau mau lebih cepat)
PAD_SEBELUM = 20  # detik konteks tambahan sebelum bagian yang dipilih Gemini
PAD_SESUDAH = 20  # detik konteks tambahan sesudah bagian yang dipilih Gemini
SRT_CADANGAN_GEMINI = True  # kalau subtitle YouTube tidak ada, buat SRT dari transkrip Gemini
BUAT_MP3 = True  # ekstrak audio mp3 dari tiap klip
MP3_KUALITAS = "2"  # kualitas VBR libmp3lame: 0 (terbaik) - 9 (terkecil); 2 = ~190 kbps


def js_runtime_opts() -> dict:
    """Pakai runtime JavaScript yang terpasang (deno/node/bun).
    YouTube sekarang butuh ini agar daftar format video lengkap."""
    for nama in ("deno", "node", "bun"):
        if shutil.which(nama):
            return {
                "js_runtimes": {nama: {}},
                # izinkan yt-dlp mengunduh skrip penyelesai tantangan YouTube
                "remote_components": ["ejs:github"],
            }
    return {}


# ---------- Util ----------
def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r'[\\/:*?"<>|]', "", text)
    text = re.sub(r"[\x00-\x1f]", "", text)  # karakter kontrol
    text = re.sub(r"\s+", " ", text).strip()
    # Windows tidak mengizinkan nama file/folder berakhir dengan spasi atau titik
    text = text[:max_len].rstrip(" .")
    return text or "tanpa_judul"


def to_seconds(ts) -> float:
    """'MM:SS' atau 'HH:MM:SS' (boleh ada koma/titik desimal) atau angka -> detik."""
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = str(ts).strip().replace(",", ".").split(":")
    parts = [float(p) for p in parts]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def fmt_srt(sec: float) -> str:
    sec = max(sec, 0)
    ms = int(round((sec - int(sec)) * 1000))
    s = int(sec) % 60
    m = (int(sec) // 60) % 60
    h = int(sec) // 3600
    if ms == 1000:
        ms, s = 0, s + 1
        if s == 60:
            s, m = 0, m + 1
            if m == 60:
                m, h = 0, h + 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_hms(sec: float) -> str:
    return fmt_srt(sec)[:8]


def normalisasi_url(url: str) -> str:
    """Ubah berbagai bentuk link YouTube (youtu.be, shorts, live, ada &t=, &list=, dll)
    menjadi https://www.youtube.com/watch?v=ID. Gemini hanya mengenali bentuk bersih ini;
    parameter tambahan membuatnya dibaca sebagai halaman web (error 'text/html')."""
    m = re.search(
        r"(?:v=|youtu\.be/|/shorts/|/live/|/embed/)([A-Za-z0-9_-]{11})", url.strip()
    )
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    return url.strip()


# ---------- Gemini ----------
def make_client():
    # Timeout 5 menit per request, dan jangan mengulang request diam-diam
    # api_version v1beta: model terbaru (flash-latest, dll) hanya ada di versi ini
    try:
        http_options = types.HttpOptions(
            api_version="v1beta",
            timeout=300_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
    except Exception:  # library versi lama tidak punya retry_options
        http_options = types.HttpOptions(api_version="v1beta", timeout=300_000)
    return genai.Client(http_options=http_options)


def parse_json_text(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    return json.loads(text)


def generate_with_retry(client, contents, config, ronde_maks: int, tunggu: int):
    """Coba semua model berurutan, ulangi beberapa putaran kalau server sibuk."""

    def heartbeat(ev):
        t0 = time.time()
        while not ev.wait(15):
            print(f"  ...masih menunggu Gemini ({int(time.time() - t0)} detik)", flush=True)

    errors = []
    for ronde in range(1, ronde_maks + 1):
        for model in MODELS:
            print(f"  Mencoba model: {model} (putaran {ronde}/{ronde_maks})", flush=True)
            stop = threading.Event()
            threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
            try:
                return client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except Exception as e:
                msg = f"{model} -> {type(e).__name__}: {str(e)[:300]}"
                print(f"  GAGAL: {msg}", flush=True)
                errors.append(msg)
                # 400 = permintaan salah (misalnya link tidak valid), bukan server sibuk.
                # Mengulang atau ganti model tidak akan membantu.
                if getattr(e, "code", None) == 400:
                    raise RuntimeError(
                        "Gemini menolak permintaan (400). Periksa link video "
                        "(harus video YouTube publik).\n" + msg
                    )
            finally:
                stop.set()
        if ronde < ronde_maks:
            print(f"  Semua model sibuk/gagal. Menunggu {tunggu} detik lalu coba lagi...", flush=True)
            time.sleep(tunggu)
    raise RuntimeError("Semua model gagal:\n" + "\n".join(errors))


PROMPT = """Kamu editor video profesional. Tonton video ini lalu pilih {n} bagian
paling menarik untuk dijadikan klip pendek (misalnya untuk Shorts/Reels/TikTok).

Aturan:
- Durasi tiap klip 30-90 detik, mulai dan berakhir di batas kalimat yang natural.
- Klip tidak boleh saling tumpang tindih.
- Pilih momen yang kuat: hook menarik, insight, lucu, emosional, atau punchline.
- Timestamp harus akurat sesuai video, format "MM:SS" atau "HH:MM:SS".
- Judul dalam bahasa yang sama dengan video, singkat dan menarik.
{tambahan}
Balas HANYA JSON array, setiap item berisi:
  "start" (string), "end" (string), "title" (string), "reason" (string)
"""


def ask_gemini(url: str, n: int, hindari: list[dict] | None = None) -> list[dict]:
    """Minta Gemini memilih n bagian menarik.
    hindari: daftar klip yang sudah pernah disarankan (supaya saran baru berbeda)."""
    tambahan = ""
    if hindari:
        daftar = "\n".join(
            f'  - {fmt_hms(c["start"])} - {fmt_hms(c["end"])} ({c["title"]})' for c in hindari
        )
        tambahan = (
            "- Bagian berikut SUDAH pernah disarankan dan ditolak pengguna. "
            "Jangan pilih bagian yang sama atau tumpang tindih dengannya, "
            f"cari momen lain yang berbeda:\n{daftar}\n"
        )

    client = make_client()
    contents = types.Content(
        parts=[
            types.Part(file_data=types.FileData(file_uri=url)),
            types.Part(text=PROMPT.format(n=n, tambahan=tambahan)),
        ]
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        # resolusi video rendah = jauh lebih sedikit token (lebih cepat, hemat kuota)
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
    )
    # server Gemini sering sibuk (503); ulangi seluruh daftar model beberapa kali
    response = generate_with_retry(client, contents, config, ronde_maks=4, tunggu=45)
    clips = parse_json_text(response.text)
    result = []
    for c in clips:
        start, end = to_seconds(c["start"]), to_seconds(c["end"])
        if end > start:
            result.append(
                {
                    "start": start,
                    "end": end,
                    "title": c.get("title", "Klip"),
                    "reason": c.get("reason", ""),
                }
            )
    return result


SRT_PROMPT = """Transkripsikan semua ucapan pada video ini menjadi subtitle.

Aturan:
- Tulis persis apa yang diucapkan, dalam bahasa yang digunakan di video.
- Pecah menjadi baris pendek (maksimal sekitar 10 kata atau 2 detik-6 detik per baris).
- Timestamp RELATIF terhadap awal video ini (detik pertama video = "00:00"),
  format "MM:SS.mmm".
- Lewati bagian tanpa ucapan.

Balas HANYA JSON array, setiap item berisi:
  "start" (string), "end" (string), "text" (string)
"""


def transcribe_clip(url: str, mulai: float, selesai: float, out: Path) -> bool:
    """Buat SRT untuk satu klip dari transkrip Gemini (cadangan)."""
    client = make_client()
    durasi_klip = selesai - mulai
    contents = types.Content(
        parts=[
            types.Part(
                file_data=types.FileData(file_uri=url),
                video_metadata=types.VideoMetadata(
                    start_offset=f"{int(mulai)}s",
                    end_offset=f"{int(math.ceil(selesai))}s",
                ),
            ),
            types.Part(text=SRT_PROMPT),
        ]
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
    )
    response = generate_with_retry(client, contents, config, ronde_maks=2, tunggu=20)
    items = parse_json_text(response.text)

    cues = []
    for it in items:
        try:
            s, e = to_seconds(it["start"]), to_seconds(it["end"])
        except Exception:
            continue
        text = str(it.get("text", "")).strip()
        if text and e > s:
            cues.append([s, e, text])
    if not cues:
        return False

    # Kalau timestamp ternyata absolut (melebihi durasi klip), geser ke relatif
    if max(c[1] for c in cues) > durasi_klip + 5:
        cues = [[s - mulai, e - mulai, t] for s, e, t in cues if e - mulai > 0]
    cues = [[max(s, 0), min(e, durasi_klip), t] for s, e, t in cues if s < durasi_klip]
    if not cues:
        return False

    with out.open("w", encoding="utf-8") as f:
        for i, (s, e, text) in enumerate(cues, 1):
            f.write(f"{i}\n{fmt_srt(s)} --> {fmt_srt(e)}\n{text}\n\n")
    return True


# ---------- Info & subtitle YouTube ----------
def get_info_and_subs(url: str, workdir: Path) -> tuple[Path | None, str, float | None]:
    """Ambil judul + durasi + subtitle YouTube (kecil). Video TIDAK diunduh di sini."""
    # Bersihkan sisa subtitle dari proses sebelumnya
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)

    # 1) Judul & durasi, tanpa download
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, **js_runtime_opts()}) as ydl:
        info = ydl.extract_info(url, download=False)
    title = info.get("title", "video")
    durasi = info.get("duration")  # detik; agar padding tidak melewati akhir video

    # 2) Subtitle satu per satu (id dulu, lalu en). Error ditampilkan, bukan disembunyikan.
    sub = None
    for lang in SUB_LANGS:
        sub_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [lang],
            "subtitlesformat": "vtt",
            "outtmpl": str(workdir / "source.%(ext)s"),
            "quiet": True,
            "no_warnings": False,
            **js_runtime_opts(),
        }
        try:
            with yt_dlp.YoutubeDL(sub_opts) as ydl:
                ydl.extract_info(url, download=True)
        except Exception as e:
            print(f"  (Subtitle '{lang}' gagal diunduh: {str(e)[:200]})")
        found = sorted(glob.glob(str(workdir / f"source.{lang}*.vtt")))
        if found:
            sub = Path(found[0])
            print(f"  Subtitle YouTube '{lang}' berhasil diunduh: {sub.name}")
            break
        time.sleep(2)

    if sub is None:
        isi = [p.name for p in workdir.iterdir()]
        print(f"  Subtitle YouTube tidak ditemukan. Isi folder sementara: {isi}")
    return sub, title, durasi


# ---------- Parse subtitle YouTube (VTT/SRT) ----------
TIME_RE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)
TIME_RE_SHORT = re.compile(  # format MM:SS.mmm tanpa jam
    r"^(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+)[,.](\d+)"
)


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    cues = []
    prev_lines: list[str] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.strip().splitlines()
        idx = next(
            (i for i, l in enumerate(lines) if TIME_RE.search(l) or TIME_RE_SHORT.search(l)),
            None,
        )
        if idx is None:
            continue
        m = TIME_RE.search(lines[idx])
        if m:
            g = [int(x) for x in m.groups()]
            start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
            end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        else:
            g = [int(x) for x in TIME_RE_SHORT.search(lines[idx]).groups()]
            start = g[0] * 60 + g[1] + g[2] / 1000
            end = g[3] * 60 + g[4] + g[5] / 1000
        text_lines = [re.sub(r"<[^>]+>", "", l).strip() for l in lines[idx + 1 :]]
        text_lines = [l for l in text_lines if l]
        # subtitle otomatis YouTube sering mengulang baris sebelumnya
        new_lines = [l for l in text_lines if l not in prev_lines]
        prev_lines = text_lines
        if new_lines:
            cues.append((start, end, "\n".join(new_lines)))
    return cues


def write_clip_srt(cues, start: float, end: float, out: Path) -> bool:
    out_cues = []
    for cs, ce, text in cues:
        if ce <= start or cs >= end:
            continue
        out_cues.append((max(cs, start) - start, min(ce, end) - start, text))
    if not out_cues:
        return False
    with out.open("w", encoding="utf-8") as f:
        for i, (s, e, text) in enumerate(out_cues, 1):
            f.write(f"{i}\n{fmt_srt(s)} --> {fmt_srt(e)}\n{text}\n\n")
    return True


# ---------- Download per klip ----------
def download_clip(url: str, start: float, end: float, out: Path, tmp: Path) -> Path:
    """Unduh hanya rentang start-end, simpan ke nama sementara yang aman
    (tanpa karakter aneh dari judul), lalu pindahkan ke nama akhir."""
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("clip.*"):  # bersihkan sisa klip sebelumnya
        f.unlink(missing_ok=True)

    opts = {
        "format": f"bv*[height<={MAX_TINGGI}]+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(tmp / "clip.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "overwrites": True,
        "quiet": True,
        "no_warnings": True,
        **js_runtime_opts(),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        kode = ydl.download([url])

    # Cari file hasil (abaikan potongan sementara seperti clip.f401.mp4 / .part)
    kandidat = [
        f for f in tmp.glob("clip.*")
        if f.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")
        and not re.search(r"\.f\d+$", f.stem)
    ]
    if not kandidat:
        sisa = [f.name for f in tmp.iterdir()]
        raise FileNotFoundError(f"Klip tidak terbentuk (kode {kode}). Isi folder sementara: {sisa}")
    hasil = max(kandidat, key=lambda f: f.stat().st_size)

    tujuan = out.with_suffix(hasil.suffix.lower())
    if tujuan.exists():
        tujuan.unlink()
    shutil.move(str(hasil), str(tujuan))
    return tujuan


# ---------- Ekstrak audio MP3 ----------
def extract_mp3(video: Path, out: Path) -> bool:
    """Ekstrak audio dari klip video menjadi mp3 memakai ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video),
        "-vn",
        "-codec:a", "libmp3lame",
        "-q:a", MP3_KUALITAS,
        str(out),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
    except FileNotFoundError:
        print("  (ffmpeg tidak ditemukan, mp3 dilewati)")
        return False
    if r.returncode != 0:
        print(f"  (Ekstrak mp3 gagal: {r.stderr.strip()[:200]})")
        return False
    return out.exists() and out.stat().st_size > 0


# ---------- Interaksi pengguna ----------
def tanya_jumlah(default: int | None = None) -> int:
    while True:
        teks = f" (default {default})" if default else ""
        s = input(f"Masukkan jumlah klip yang diinginkan{teks}: ").strip()
        if not s and default:
            return default
        if s.isdigit() and int(s) > 0:
            return int(s)
        print("  Masukkan angka bulat lebih dari 0.")


def tampilkan_saran(clips: list[dict]) -> None:
    print("\n=== SARAN KLIP DARI GEMINI ===")
    for i, c in enumerate(clips, 1):
        durasi = c["end"] - c["start"]
        print(f"\n {i}. {c['title']}")
        print(f"    Waktu : {fmt_hms(c['start'])} - {fmt_hms(c['end'])}  ({int(durasi)} detik)")
        if c["reason"]:
            print(f"    Alasan: {c['reason']}")
    print(
        f"\n(Saat dipotong, tiap klip ditambah konteks {PAD_SEBELUM} detik "
        f"sebelum dan {PAD_SESUDAH} detik sesudah.)"
    )


def pilih_klip(url: str, n: int) -> list[dict] | None:
    """Minta saran Gemini, lalu tanya pengguna: lanjutkan, cari lagi, atau batal.
    Mengembalikan daftar klip yang disetujui, atau None kalau dibatalkan."""
    sudah_disarankan: list[dict] = []
    clips: list[dict] = []

    while True:
        print("\nMeminta Gemini memilih bagian menarik...")
        clips = ask_gemini(url, n, hindari=sudah_disarankan)
        if not clips:
            print("Gemini tidak mengembalikan klip yang valid.")
        else:
            tampilkan_saran(clips)

        while True:
            print("\nPilih tindakan:")
            if clips:
                print("  [1] Lanjutkan potong klip")
            print("  [2] Cari saran lain")
            print("  [3] Batal / keluar")
            pilihan = input("Pilihan: ").strip()
            if pilihan == "1" and clips:
                return clips
            if pilihan == "2":
                sudah_disarankan.extend(clips)
                break  # ulangi permintaan ke Gemini
            if pilihan == "3":
                return None
            print("  Pilihan tidak valid.")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Auto clip video YouTube dengan Gemini")
    ap.add_argument("url", nargs="?", help="Link video YouTube (opsional, kalau kosong akan ditanya)")
    ap.add_argument("--jumlah", type=int, default=None, help="Jumlah klip (opsional)")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("PERINGATAN: ffmpeg tidak ditemukan di PATH. Pemotongan/mp3 bisa gagal.")

    runtime = js_runtime_opts()
    if runtime:
        print(f"Runtime JavaScript untuk yt-dlp: {list(runtime['js_runtimes'])[0]}")
    else:
        print(
            "PERINGATAN: deno/node tidak ditemukan. YouTube bisa menyembunyikan sebagian format,\n"
            "sehingga unduhan klip gagal. Pasang Node.js (https://nodejs.org) lalu jalankan:\n"
            '  pip install -U "yt-dlp[default]"'
        )

    # [1] Link video
    url = args.url
    while not url:
        url = input("\nMasukkan link YouTube: ").strip()
    url_asli = url
    url = normalisasi_url(url)
    if url != url_asli:
        print(f"  Link dirapikan menjadi: {url}")

    # [2] Jumlah klip
    jumlah = args.jumlah if args.jumlah and args.jumlah > 0 else tanya_jumlah()

    # [3] Saran Gemini + konfirmasi (lanjutkan / cari lagi)
    clips = pilih_klip(url, jumlah)
    if clips is None:
        print("Dibatalkan oleh pengguna.")
        return

    # [4] Info video & subtitle YouTube
    print("\nAmbil judul & subtitle...")
    tmp = OUTPUT_DIR / "_tmp"
    srt, video_title, durasi = get_info_and_subs(url, tmp)

    folder = OUTPUT_DIR / safe_name(video_title)
    folder.mkdir(parents=True, exist_ok=True)

    cues = parse_srt(srt) if srt else []
    if cues:
        print(
            f"  Subtitle YouTube terbaca: {len(cues)} baris, "
            f"mencakup {fmt_hms(cues[0][0])} - {fmt_hms(cues[-1][1])}."
        )
    elif SRT_CADANGAN_GEMINI:
        print("  Subtitle YouTube tidak tersedia. SRT akan dibuat dari transkrip Gemini.")
    else:
        print("  Subtitle YouTube tidak tersedia, file .srt dilewati.")

    # [5] Potong: mp4 + srt + mp3
    print("\nDownload klip + subtitle + audio...")
    summary = []
    gagal = 0
    for i, c in enumerate(clips, 1):
        base = f"{i:02d} - {safe_name(c['title'])}"
        mp4 = folder / f"{base}.mp4"
        srt_path = folder / f"{base}.srt"
        mp3_path = folder / f"{base}.mp3"

        # Tambah konteks sebelum & sesudah, dibulatkan ke detik utuh agar video & SRT sejajar
        mulai = float(int(max(c["start"] - PAD_SEBELUM, 0)))
        selesai = float(math.ceil(c["end"] + PAD_SESUDAH))
        if durasi:
            selesai = min(selesai, float(durasi))

        print(f"\n  Klip {i}/{len(clips)}: {c['title']}")
        try:
            mp4 = download_clip(url, mulai, selesai, mp4, tmp)
        except Exception as e:
            gagal += 1
            print(f"  ✘ Klip {i} gagal: {type(e).__name__}: {str(e)[:200]}")
            continue

        # SRT: utamakan subtitle YouTube, cadangannya transkrip Gemini
        has_srt, sumber_srt = False, None
        if cues:
            has_srt = write_clip_srt(cues, mulai, selesai, srt_path)
            sumber_srt = "youtube" if has_srt else None
            if not has_srt:
                print(
                    f"  Klip {i}: subtitle YouTube tidak mencakup "
                    f"{fmt_hms(mulai)} - {fmt_hms(selesai)}."
                )
        if not has_srt and SRT_CADANGAN_GEMINI:
            print(f"  Klip {i}: minta Gemini membuat transkrip untuk SRT...")
            try:
                has_srt = transcribe_clip(url, mulai, selesai, srt_path)
                sumber_srt = "gemini" if has_srt else None
            except Exception as e:
                print(f"  (SRT Gemini gagal: {type(e).__name__}: {str(e)[:200]})")

        # MP3: ekstrak dari klip mp4 yang sudah jadi
        has_mp3 = extract_mp3(mp4, mp3_path) if BUAT_MP3 else False

        summary.append(
            {
                "judul": c["title"],
                "mulai": fmt_srt(mulai),
                "selesai": fmt_srt(selesai),
                "bagian_inti_gemini": f"{fmt_srt(c['start'])} - {fmt_srt(c['end'])}",
                "alasan": c["reason"],
                "video": mp4.name,
                "srt": srt_path.name if has_srt else None,
                "sumber_srt": sumber_srt,
                "mp3": mp3_path.name if has_mp3 else None,
            }
        )
        print(f"  ✔ {mp4.name}")
        print(f"    srt: {srt_path.name + ' (' + sumber_srt + ')' if has_srt else '(tanpa SRT)'}")
        print(f"    mp3: {mp3_path.name if has_mp3 else '(tanpa MP3)'}")

    print("\nMenyimpan ringkasan...")
    (folder / "klip.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.rmtree(tmp, ignore_errors=True)
    jumlah_srt = sum(1 for s in summary if s["srt"])
    jumlah_mp3 = sum(1 for s in summary if s["mp3"])
    print(
        f"\nSelesai! {len(summary)} klip berhasil "
        f"({jumlah_srt} dengan SRT, {jumlah_mp3} dengan MP3), {gagal} gagal."
    )
    print(f"Hasil ada di: {folder.resolve()}")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except KeyboardInterrupt:
        print("\nDihentikan oleh pengguna.")
    except Exception:
        err = traceback.format_exc()
        print("\n=== TERJADI ERROR ===")
        print(err)
        Path("error_log.txt").write_text(err, encoding="utf-8")
        print("(Error juga disimpan di error_log.txt)")
    # Tahan jendela supaya tidak langsung menutup (bisa dibaca dulu)
    input("\nTekan Enter untuk menutup jendela...")