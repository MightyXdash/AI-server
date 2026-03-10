import base64
import io
import json
import os
import re
import subprocess
import time
from difflib import get_close_matches
from pathlib import Path

import psutil
import requests
from PIL import ImageGrab

SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".wma"}
APP_LIBRARY_FILE = Path("app_library.json")
MEMORY_FILE = Path("memories.jsonl")

DEFAULT_APP_LIBRARY = {
    "calculator": "calc.exe",
    "notepad": "notepad.exe",
    "settings": "ms-settings:",
    "paint": "mspaint.exe",
    "task manager": "taskmgr.exe",
}

VISION_MODEL = "ministral-3:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def _load_app_library() -> dict:
    library = dict(DEFAULT_APP_LIBRARY)

    try:
        if APP_LIBRARY_FILE.exists():
            with open(APP_LIBRARY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                        library[k.strip().lower()] = v.strip()
    except Exception:
        pass

    return library


def list_apps():
    library = _load_app_library()
    names = sorted(library.keys())
    return names, library


def open_app(app: str):
    name = (app or "").lower().strip()
    if not name:
        return False, "Tell me which app to open."

    names, library = list_apps()

    if name in library:
        target = library[name]
        matched_name = name
    else:
        matches = get_close_matches(name, names, n=1, cutoff=0.45)
        if not matches:
            return False, "That app is not in the library."
        matched_name = matches[0]
        target = library[matched_name]

    try:
        if target.startswith("ms-"):
            os.startfile(target)
        elif os.path.isabs(target):
            os.startfile(target)
        else:
            subprocess.Popen(target, shell=True)
        return True, f"Opening {matched_name}"
    except Exception as e:
        return False, f"Failed to open the app. ({str(e)})"


def _possible_music_dirs():
    home = Path.home()
    return [
        home / "Music",
        home / "OneDrive" / "Music",
        Path("C:/Users/Public/Music"),
    ]


def _find_music_dir():
    for p in _possible_music_dirs():
        if p.exists() and p.is_dir():
            return p
    return Path.home() / "Music"


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def list_music():
    music_dir = _find_music_dir()
    mapping = {}
    names = []

    if not music_dir.exists():
        return [], {}

    count = 0
    for path in music_dir.rglob("*"):
        if count >= 2000:
            break
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            display = path.name
            mapping[display] = str(path)
            names.append(display)
            count += 1

    names.sort(key=lambda x: x.lower())
    return names, mapping


def _best_match(query: str, names: list[str]) -> str | None:
    if not names:
        return None

    qn = _normalize(query)

    scored = []
    for n in names:
        nn = _normalize(n)
        if qn and qn in nn:
            scored.append((abs(len(nn) - len(qn)), n))
    if scored:
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    matches = get_close_matches(query, names, n=1, cutoff=0.35)
    if matches:
        return matches[0]

    stems = [Path(n).stem for n in names]
    stem_match = get_close_matches(query, stems, n=1, cutoff=0.35)
    if stem_match:
        target_stem = stem_match[0]
        for n in names:
            if Path(n).stem == target_stem:
                return n

    return None


def play_music(query: str):
    names, mapping = list_music()
    if not names:
        return False, "No music files found."

    chosen = _best_match(query, names)
    if not chosen:
        return False, "Couldn't find that song."

    full_path = mapping.get(chosen)
    if not full_path:
        return False, "Matched a song name but couldn't resolve the file path."

    try:
        os.startfile(full_path)
        return True, f"Playing: {chosen}"
    except Exception as e:
        return False, f"Failed to open the music file. ({str(e)})"


def list_memories():
    memories = []
    if not MEMORY_FILE.exists():
        return memories

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    text = str(row.get("text", "")).strip()
                    if text:
                        memories.append(row)
    except Exception:
        pass

    return memories


def save_memory(text: str):
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return False, "Memory text was empty."

    if len(cleaned) > 160:
        cleaned = cleaned[:157].rstrip() + "..."

    existing = [m.get("text", "").strip().lower() for m in list_memories()]
    if cleaned.lower() in existing:
        return True, f"Memory already saved: {cleaned}"

    row = {"text": cleaned}

    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True, f"Saved memory: {cleaned}"
    except Exception as e:
        return False, f"Failed to save memory. ({str(e)})"


def _safe_float(value, fallback=0.0):
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _read_gpu_stats_windows():
    """Returns GPU utilization and VRAM usage percentage if NVIDIA tools are available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {
                "available": False,
                "name": "GPU unavailable",
                "utilization": 0.0,
                "memory_percent": 0.0,
            }

        first = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) < 4:
            return {
                "available": False,
                "name": "GPU unavailable",
                "utilization": 0.0,
                "memory_percent": 0.0,
            }

        util = _safe_float(parts[0])
        mem_used = _safe_float(parts[1])
        mem_total = max(_safe_float(parts[2], 1.0), 1.0)
        name = parts[3]
        return {
            "available": True,
            "name": name,
            "utilization": round(util, 2),
            "memory_percent": round((mem_used / mem_total) * 100.0, 2),
        }
    except Exception:
        return {
            "available": False,
            "name": "GPU unavailable",
            "utilization": 0.0,
            "memory_percent": 0.0,
        }


def get_hardware_snapshot():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    gpu = _read_gpu_stats_windows()
    return {
        "timestamp": round(time.time(), 3),
        "cpu": round(cpu, 2),
        "ram": round(ram, 2),
        "gpu": round(gpu.get("utilization", 0.0), 2),
        "gpu_memory": round(gpu.get("memory_percent", 0.0), 2),
        "gpu_name": gpu.get("name", "GPU unavailable"),
        "gpu_available": bool(gpu.get("available", False)),
    }


def get_hardware_stats_series(duration_sec: float = 3.0, step_sec: float = 0.5):
    duration_sec = max(0.5, float(duration_sec))
    step_sec = max(0.2, float(step_sec))

    samples = []
    steps = max(1, int(round(duration_sec / step_sec)))

    psutil.cpu_percent(interval=None)
    start = time.time()
    for i in range(steps):
        samples.append(get_hardware_snapshot())
        if i < steps - 1:
            time.sleep(step_sec)

    elapsed = round(time.time() - start, 3)
    return {
        "duration_sec": duration_sec,
        "step_sec": step_sec,
        "elapsed_sec": elapsed,
        "samples": samples,
        "summary": summarize_hardware_series(samples),
    }


def summarize_hardware_series(samples: list[dict]):
    if not samples:
        return {
            "cpu_avg": 0.0,
            "ram_avg": 0.0,
            "gpu_avg": 0.0,
            "gpu_memory_avg": 0.0,
            "peak_cpu": 0.0,
            "peak_ram": 0.0,
            "peak_gpu": 0.0,
            "gpu_name": "GPU unavailable",
        }

    cpu_vals = [float(s.get("cpu", 0.0)) for s in samples]
    ram_vals = [float(s.get("ram", 0.0)) for s in samples]
    gpu_vals = [float(s.get("gpu", 0.0)) for s in samples]
    gpu_mem_vals = [float(s.get("gpu_memory", 0.0)) for s in samples]
    gpu_name = samples[0].get("gpu_name", "GPU unavailable")

    return {
        "cpu_avg": round(sum(cpu_vals) / len(cpu_vals), 2),
        "ram_avg": round(sum(ram_vals) / len(ram_vals), 2),
        "gpu_avg": round(sum(gpu_vals) / len(gpu_vals), 2),
        "gpu_memory_avg": round(sum(gpu_mem_vals) / len(gpu_mem_vals), 2),
        "peak_cpu": round(max(cpu_vals), 2),
        "peak_ram": round(max(ram_vals), 2),
        "peak_gpu": round(max(gpu_vals), 2),
        "gpu_name": gpu_name,
    }


def show_hardwareStats(duration_sec: float = 3.0, step_sec: float = 0.5):
    stats = get_hardware_stats_series(duration_sec=duration_sec, step_sec=step_sec)
    summary = stats["summary"]
    text = (
        f"Hardware over {stats['elapsed_sec']}s | "
        f"CPU avg {summary['cpu_avg']}% peak {summary['peak_cpu']}% | "
        f"RAM avg {summary['ram_avg']}% peak {summary['peak_ram']}% | "
        f"GPU avg {summary['gpu_avg']}% peak {summary['peak_gpu']}% | "
        f"GPU mem avg {summary['gpu_memory_avg']}% | GPU: {summary['gpu_name']}"
    )
    return True, text, stats


def screen_vision():
    try:
        img = ImageGrab.grab()

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "model": VISION_MODEL,
            "prompt": "Describe what is happening on this screen in plain text. Be concise but clear.",
            "images": [b64],
            "stream": False,
        }

        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return True, data.get("response", "").strip()

    except Exception as e:
        return False, f"Screen tool failed. ({str(e)})"
