# 🎛 AudioSplit

**Music Source Separation Tool** — Separate vocals, drums, bass, guitar, piano and more from any song.

Built from scratch using [Demucs](https://github.com/facebookresearch/demucs) as the AI separation engine.

## Features

- **3 separation modes**: 2-stem (vocals/instrumental), 4-stem, 6-stem
- **4 AI models**: HT Demucs, HT Demucs Fine-tuned, MDX Extra, MDX Extra Quantized
- **Multiple output formats**: WAV, MP3 (320kbps), FLAC
- **Web UI**: Modern dark interface with drag & drop, audio preview, and batch download
- **Background processing**: Non-blocking separation with real-time progress
- **Self-hostable**: Deploy on any server with Python

## Quick Start

### 1. Install dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install requirements
pip install -r requirements.txt
```

### 2. Run the app

```bash
python app.py
# or with options:
python app.py --host 0.0.0.0 --port 5555 --debug
```

### 3. Open in browser

Navigate to `http://127.0.0.1:5555`

## macOS Apple Silicon (M1/M2/M3/M4)

Demucs supports MPS acceleration on Apple Silicon:

```bash
# PyTorch with MPS support should install automatically
pip install torch torchaudio
```

The separation will automatically use the Metal Performance Shaders (MPS) backend if available.

## Deploy on VPS

```bash
# Install on your server
git clone <your-repo> audiosplit
cd audiosplit
pip install -r requirements.txt

# Run with gunicorn for production
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5555 --timeout 600 app:app
```

**Nginx config example:**

```nginx
server {
    listen 80;
    server_name audiosplit.yourdomain.com;

    client_max_body_size 500M;

    location / {
        proxy_pass http://127.0.0.1:5555;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600;
        proxy_send_timeout 600;
    }
}
```

## Project Structure

```
audiosplit/
├── app.py              # Flask backend + API routes
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web UI (single file, self-contained)
└── static/
    ├── uploads/        # Temporary uploaded files
    └── outputs/        # Separated stems output
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/api/info` | Available modes, models, formats |
| POST | `/api/separate` | Upload file & start separation |
| GET | `/api/status/<job_id>` | Check job progress |
| GET | `/api/download/<job_id>/<stem>` | Download a stem |
| GET | `/api/download-all/<job_id>` | Download all stems as ZIP |
| DELETE | `/api/cleanup/<job_id>` | Clean up job files |

## Models

| Model | Quality | Speed | Notes |
|-------|---------|-------|-------|
| `htdemucs` | ★★★★ | Fast | Default, best balance |
| `htdemucs_ft` | ★★★★★ | Slow | Fine-tuned, highest quality |
| `mdx_extra` | ★★★★ | Medium | Good for vocals |
| `mdx_extra_q` | ★★★ | Fastest | Quantized, lower quality |

## License

MIT — Use it, modify it, ship it.
