"""
AudioSplit - Music Source Separation Tool
Built from scratch using Demucs as the separation engine.
Author: Alejandro
"""

import os
import uuid
import json
import shutil
import threading
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Track job status in memory
jobs = {}

ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.wma', '.aac', '.opus'}

# Available separation modes
SEPARATION_MODES = {
    "2stems": {
        "name": "Vocals + Instrumental",
        "description": "Separate vocals from instrumental",
        "args": ["--two-stems", "vocals"],
        "stems": ["vocals", "no_vocals"]
    },
    "4stems": {
        "name": "4 Stems (Vocals, Drums, Bass, Other)",
        "description": "Full separation into 4 tracks",
        "args": [],
        "stems": ["vocals", "drums", "bass", "other"]
    },
    "6stems": {
        "name": "6 Stems (+ Guitar, Piano)",
        "description": "Extended separation with guitar and piano",
        "args": ["-n", "htdemucs_6s"],
        "stems": ["vocals", "drums", "bass", "guitar", "piano", "other"]
    }
}

# Available models
MODELS = {
    "htdemucs": {
        "name": "Hybrid Transformer Demucs (Default)",
        "description": "Best overall quality"
    },
    "htdemucs_ft": {
        "name": "Hybrid Transformer Demucs Fine-tuned",
        "description": "Highest quality, slower"
    },
    "mdx_extra": {
        "name": "MDX Extra",
        "description": "Good for vocals extraction"
    },
    "mdx_extra_q": {
        "name": "MDX Extra Quantized",
        "description": "Faster, slightly lower quality"
    },
}


def run_separation(job_id: str, filepath: str, mode: str, model: str, output_format: str):
    """Run Demucs separation in a background thread."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 10

        job_output = OUTPUT_DIR / job_id
        job_output.mkdir(parents=True, exist_ok=True)

        mode_config = SEPARATION_MODES.get(mode, SEPARATION_MODES["2stems"])

        # Build the demucs command
        cmd = ["python3", "-m", "demucs"]

        # Model selection (6stems forces htdemucs_6s)
        if mode != "6stems":
            cmd.extend(["-n", model])

        # Mode args
        cmd.extend(mode_config["args"])

        # Memory management: use smaller segments for heavy models
        # 6stems and fine-tuned models need more RAM, segment limits peak usage
        if mode == "6stems":
            cmd.extend(["--segment", "7"])
        elif model == "htdemucs_ft":
            cmd.extend(["--segment", "7"])

        # Output format
        if output_format == "mp3":
            cmd.extend(["--mp3", "--mp3-bitrate", "320"])
        elif output_format == "flac":
            cmd.append("--flac")

        # Output directory
        cmd.extend(["-o", str(job_output)])

        # Input file
        cmd.append(filepath)

        jobs[job_id]["progress"] = 20
        jobs[job_id]["cmd"] = " ".join(cmd)
        jobs[job_id]["detail"] = "Running Demucs separation..."

        # Run the process — merge stderr into stdout so we capture everything
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Collect all output lines for error reporting
        all_output = []

        for line in process.stdout:
            line = line.strip()
            if line:
                all_output.append(line)
                # Keep last 20 lines max for error reporting
                if len(all_output) > 50:
                    all_output = all_output[-50:]

                jobs[job_id]["detail"] = line

                # Try to parse progress from demucs output
                if "%" in line:
                    try:
                        pct = int(line.split("%")[0].split()[-1])
                        jobs[job_id]["progress"] = 20 + int(pct * 0.7)
                    except (ValueError, IndexError):
                        pass

        process.wait()

        if process.returncode != 0:
            # Show the last meaningful lines as the error
            error_lines = [l for l in all_output if l.strip()]
            # Filter to show traceback and error, skip download/info lines
            error_tail = "\n".join(error_lines[-15:])
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = f"Demucs failed (code {process.returncode}):\n{error_tail}"
            return

        jobs[job_id]["progress"] = 95
        jobs[job_id]["detail"] = "Collecting output files..."

        # Find the output stems
        ext = output_format if output_format in ("mp3", "flac") else "wav"
        stems_found = []

        # Demucs outputs to: output_dir/model_name/track_name/stem.wav
        for stem_file in job_output.rglob(f"*.{ext}"):
            stem_name = stem_file.stem
            stems_found.append({
                "name": stem_name,
                "filename": stem_file.name,
                "path": str(stem_file.relative_to(BASE_DIR)),
                "size": stem_file.stat().st_size
            })

        # Also check for wav if format was wav (demucs default)
        if not stems_found:
            for stem_file in job_output.rglob("*.wav"):
                stem_name = stem_file.stem
                stems_found.append({
                    "name": stem_name,
                    "filename": stem_file.name,
                    "path": str(stem_file.relative_to(BASE_DIR)),
                    "size": stem_file.stat().st_size
                })

        jobs[job_id]["stems"] = stems_found
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["detail"] = f"Done! {len(stems_found)} stems separated."

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info")
def api_info():
    """Return available modes and models."""
    return jsonify({
        "modes": {k: {"name": v["name"], "description": v["description"]} for k, v in SEPARATION_MODES.items()},
        "models": {k: {"name": v["name"], "description": v["description"]} for k, v in MODELS.items()},
        "formats": ["wav", "mp3", "flac"],
        "max_upload_mb": 500
    })


@app.route("/api/separate", methods=["POST"])
def separate():
    """Upload a file and start separation."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format: {ext}. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    mode = request.form.get("mode", "2stems")
    model = request.form.get("model", "htdemucs")
    output_format = request.form.get("format", "wav")

    if mode not in SEPARATION_MODES:
        return jsonify({"error": f"Invalid mode: {mode}"}), 400
    if model not in MODELS:
        return jsonify({"error": f"Invalid model: {model}"}), 400

    # Save uploaded file
    job_id = str(uuid.uuid4())[:12]
    upload_path = UPLOAD_DIR / f"{job_id}{ext}"
    file.save(str(upload_path))

    file_size = upload_path.stat().st_size

    # Initialize job
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "detail": "Queued for processing...",
        "filename": file.filename,
        "file_size": file_size,
        "mode": mode,
        "model": model,
        "format": output_format,
        "stems": [],
        "error": None
    }

    # Start separation in background
    thread = threading.Thread(
        target=run_separation,
        args=(job_id, str(upload_path), mode, model, output_format),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/status/<job_id>")
def job_status(job_id):
    """Check job status."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(jobs[job_id])


@app.route("/api/download/<job_id>/<stem_name>")
def download_stem(job_id, stem_name):
    """Download a specific separated stem."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed"}), 400

    for stem in job["stems"]:
        if stem["name"] == stem_name:
            file_path = BASE_DIR / stem["path"]
            if file_path.exists():
                return send_file(
                    str(file_path),
                    as_attachment=True,
                    download_name=f"{Path(job['filename']).stem}_{stem_name}{file_path.suffix}"
                )

    return jsonify({"error": "Stem not found"}), 404


@app.route("/api/download-all/<job_id>")
def download_all(job_id):
    """Download all stems as a zip file."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed"}), 400

    # Create zip
    zip_name = f"{Path(job['filename']).stem}_stems"
    zip_path = OUTPUT_DIR / job_id / zip_name

    # Find the directory containing stems
    stem_dir = None
    for stem in job["stems"]:
        stem_file = BASE_DIR / stem["path"]
        stem_dir = stem_file.parent
        break

    if stem_dir and stem_dir.exists():
        shutil.make_archive(str(zip_path), 'zip', str(stem_dir))
        return send_file(
            f"{zip_path}.zip",
            as_attachment=True,
            download_name=f"{zip_name}.zip"
        )

    return jsonify({"error": "Output files not found"}), 404


@app.route("/api/cleanup/<job_id>", methods=["DELETE"])
def cleanup_job(job_id):
    """Clean up job files."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    # Remove upload
    for f in UPLOAD_DIR.glob(f"{job_id}.*"):
        f.unlink(missing_ok=True)

    # Remove output directory
    job_output = OUTPUT_DIR / job_id
    if job_output.exists():
        shutil.rmtree(job_output)

    del jobs[job_id]
    return jsonify({"status": "cleaned"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AudioSplit - Music Source Separation")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5555, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print(f"""
    ╔══════════════════════════════════════════╗
    ║         🎵  AudioSplit  🎵              ║
    ║    Music Source Separation Tool          ║
    ║    http://{args.host}:{args.port}               ║
    ╚══════════════════════════════════════════╝
    """)

    app.run(host=args.host, port=args.port, debug=args.debug)
