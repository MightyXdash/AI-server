import json
import re
from pathlib import Path
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from tools import (
    get_hardware_stats_series,
    list_apps,
    list_memories,
    list_music,
    open_app,
    play_music,
    save_memory,
    screen_vision,
)

BASE_DIR = Path(__file__).resolve().parent
PORT = 6967
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL = "lfm2:24b-q4_K_M"
NUM_CTX = 16000
TOOL_PREFIX = "TOOLCALL"
MAX_HISTORY = 20
TOOLS_ENABLED_MODEL_ALLOWLIST: list[str] = []

conversation_history: list[dict[str, str]] = []

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskBody(BaseModel):
    message: str
    model: str | None = None
    hardware: dict[str, Any] | None = None


class HardwareBody(BaseModel):
    duration_sec: float = 3.0
    step_sec: float = 0.5



def json_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")



def build_memory_block() -> str:
    memories = list_memories()
    if not memories:
        return "- no saved memories yet"

    lines = []
    for item in memories[:200]:
        text = str(item.get("text", "")).strip()
        if text:
            lines.append(f"- {text}")

    return "\n".join(lines) if lines else "- no saved memories yet"



def hardware_context_block(hardware: dict[str, Any] | None) -> str:
    if not hardware:
        return "No hardware sample was provided for this turn."

    samples = hardware.get("samples") or []
    summary = hardware.get("summary") or {}
    if not samples:
        return "No hardware sample was provided for this turn."

    lines = [
        "Recent hardware telemetry captured on request:",
        f"- Duration: {hardware.get('elapsed_sec', hardware.get('duration_sec', 0))} sec",
        f"- Step: {hardware.get('step_sec', 0.5)} sec",
        (
            f"- Summary: CPU avg {summary.get('cpu_avg', 0)}% peak {summary.get('peak_cpu', 0)}% | "
            f"RAM avg {summary.get('ram_avg', 0)}% peak {summary.get('peak_ram', 0)}% | "
            f"GPU avg {summary.get('gpu_avg', 0)}% peak {summary.get('peak_gpu', 0)}% | "
            f"GPU mem avg {summary.get('gpu_memory_avg', 0)}% | GPU: {summary.get('gpu_name', 'GPU unavailable')}"
        ),
        "- Samples:",
    ]

    for idx, sample in enumerate(samples, start=1):
        lines.append(
            f"  {idx}. CPU {sample.get('cpu', 0)}% | RAM {sample.get('ram', 0)}% | "
            f"GPU {sample.get('gpu', 0)}% | GPU mem {sample.get('gpu_memory', 0)}%"
        )

    return "\n".join(lines)



def build_system_prompt(music_names: list[str], app_names: list[str], hardware: dict[str, Any] | None) -> str:
    music_preview_limit = 200
    preview = music_names[:music_preview_limit]
    extra = len(music_names) - len(preview)

    music_block = "\n".join(f"- {name}" for name in preview)
    if extra > 0:
        music_block += f"\n- ... and {extra} more"

    app_block = "\n".join(f"- {name}" for name in app_names) if app_names else "- no apps configured"
    memory_block = build_memory_block()
    hardware_block = hardware_context_block(hardware)

    return (
        "You are /AI, a local assistant running on a Windows PC. Your name is atlas.\n"
        "Tone and style:\n"
        "- Be slightly Gen Z casual, warm, and natural.\n"
        "- You can use words like bet, fr, ngl, and fair when they fit naturally.\n"
        "- Do not overdo slang in every line.\n"
        "- Use proper markdown when it helps clarity.\n"
        "- Be helpful, clear, and relaxed instead of strict or overly formal.\n"
        "- Call the user bro sometimes if it feels natural.\n"
        "- Use emojis lightly and naturally.\n"
        "\n"
        "Saved memories about the user(in user's perspective):\n"
        f"{memory_block}\n"
        "\n"
        f"{hardware_block}\n"
        "\n"
        "Rules (must follow):\n"
        "1) You may answer normally in markdown.\n"
        "2) Never reveal tool call syntax to the user.\n"
        "3) If you want to use a tool, output EXACTLY ONE LINE starting with TOOLCALL followed by strict JSON.\n"
        '   TOOLCALL{"name":"tool_name","args":{...}}\n'
        "4) If the user asks to play a song, you MUST choose from the provided music list.\n"
        "5) If the user misspells a song, pick the closest match from the music list.\n"
        "6) If the user asks what is on the screen or what is happening on the screen, you MUST call screen_vision.\n"
        "7) If the user asks to open an app, you MUST only use apps from the approved app library below.\n"
        "8) If the requested app is not in the approved app library below, reply exactly: That app is not in the library.\n"
        "9) If the user asks which apps are available, use list_apps.\n"
        "10) Never mention saving memory or memory tools.\n"
        "11) Memories must be short factual snippets, not essays.\n"
        "12) Never invent memories.\n"
        "13) Use show_hardwareStats when the user actually asks for hardware stats, system load, CPU, RAM, or GPU usage.\n"
        "Available tools:\n"
        "A) list_music args: {}\n"
        'B) play_music args: {"query": "<what the user typed>"}\n'
        "C) screen_vision args: {}\n"
        'D) open_app args: {"app": "<app name>"}\n'
        "E) list_apps args: {}\n"
        'F) save_memory args: {"text": "<very short memory>"}\n'
        'G) show_hardwareStats args: {"duration_sec": 3, "step_sec": 0.5}\n'
        "\n"
        "Approved app library:\n"
        f"{app_block}\n"
        "\n"
        "Music library (names):\n"
        f"{music_block}\n"
    )



def maybe_autosave_memory(user_message: str):
    msg = re.sub(r"\s+", " ", (user_message or "")).strip()
    if not msg:
        return

    patterns = [
        r"\bmy name is\s+(.+)",
        r"\bi am\s+\d{1,2}\b",
        r"\bi'm\s+\d{1,2}\b",
        r"\bi like\s+(.+)",
        r"\bi love\s+(.+)",
        r"\bi hate\s+(.+)",
        r"\bi prefer\s+(.+)",
        r"\bmy favorite\s+.+",
    ]
    lower = msg.lower()
    for pat in patterns:
        if re.search(pat, lower):
            text = msg[:160].rstrip()
            if len(msg) > 160:
                text += "..."
            save_memory(text)
            return



def parse_tool_call(text: str):
    t = (text or "").strip()
    if not t:
        return None

    if t.startswith(TOOL_PREFIX):
        raw = t[len(TOOL_PREFIX):].strip()
    else:
        m = re.search(r"TOOLCALL\s*(\{.*\})", t, flags=re.DOTALL)
        if not m:
            return None
        raw = m.group(1).strip()

    raw = raw.strip("` \n")
    try:
        obj = json.loads(raw)
        name = obj.get("name")
        args = obj.get("args", {})
        if not isinstance(args, dict):
            args = {}
        return {"name": name, "args": args}
    except Exception:
        return None



def try_local_tool_route(user_message: str):
    text = (user_message or "").strip()
    low = text.lower()

    if not text:
        return None

    if re.fullmatch(r"(list|show)( the)? music", low) or low in {"list music", "show music", "music list", "what music do i have"}:
        music_names, _ = list_music()
        if not music_names:
            return "No music files found in your Music folder."
        lines = [f"{i}) {s}" for i, s in enumerate(music_names[:300], start=1)]
        if len(music_names) > 300:
            lines.append(f"...and {len(music_names) - 300} more")
        return "\n".join(lines)

    if low in {"list apps", "show apps", "what apps are available", "which apps are available", "available apps"}:
        app_names, _ = list_apps()
        if not app_names:
            return "No apps are in the library yet."
        return "\n".join(f"{i}) {name}" for i, name in enumerate(app_names, start=1))

    open_match = re.match(r"^(?:open|launch|start)\s+(.+)$", low)
    if open_match:
        app_name = open_match.group(1).strip()
        ok, msg = open_app(app_name)
        return msg

    screen_patterns = [
        "what is on my screen",
        "what's on my screen",
        "whats on my screen",
        "what is happening on my screen",
        "what's happening on my screen",
        "describe my screen",
        "look at my screen",
        "see my screen",
    ]
    if any(p in low for p in screen_patterns):
        ok, msg = screen_vision()
        return msg

    play_match = re.match(r"^(?:play)\s+(.+)$", text, flags=re.IGNORECASE)
    if play_match:
        query = play_match.group(1).strip()
        ok, msg = play_music(query)
        return msg

    return None



def model_supports_reasoning(model_name: str) -> bool:
    lower = (model_name or "").lower()
    reasoning_markers = ["r1", "qwq", "reason", "thinking", "deepseek-r1"]
    return any(marker in lower for marker in reasoning_markers)



def is_tools_enabled_model(name: str) -> bool:
    lower = name.lower()
    blocked_markers = ["embed", "vision", "vl", "whisper", "tts", "sd", "diffusion"]
    if any(marker in lower for marker in blocked_markers):
        return False
    if TOOLS_ENABLED_MODEL_ALLOWLIST:
        return name in TOOLS_ENABLED_MODEL_ALLOWLIST
    return True



def get_installed_tool_models() -> list[str]:
    try:
        res = requests.get(OLLAMA_TAGS_URL, timeout=10)
        res.raise_for_status()
        data = res.json()
        raw_models = data.get("models", [])
        names = []
        for item in raw_models:
            name = item.get("name")
            if isinstance(name, str) and name.strip() and is_tools_enabled_model(name.strip()):
                names.append(name.strip())
        names = sorted(set(names), key=lambda x: x.lower())
        return names
    except Exception:
        return TOOLS_ENABLED_MODEL_ALLOWLIST[:] if TOOLS_ENABLED_MODEL_ALLOWLIST else [DEFAULT_MODEL]



def execute_non_hardware_tool(tool_call: dict[str, Any], music_names: list[str], app_names: list[str]) -> str:
    name = tool_call["name"]
    args = tool_call["args"]

    if name == "list_music":
        if not music_names:
            return "No music files found in your Music folder."
        lines = [f"{i}) {s}" for i, s in enumerate(music_names[:300], start=1)]
        if len(music_names) > 300:
            lines.append(f"...and {len(music_names) - 300} more")
        return "\n".join(lines)

    if name == "play_music":
        query = str(args.get("query", "")).strip()
        if not query:
            return "Tell me which song to play."
        ok, msg = play_music(query)
        return msg

    if name == "screen_vision":
        ok, msg = screen_vision()
        return msg

    if name == "open_app":
        app_name = str(args.get("app", "")).strip()
        if not app_name:
            return "Tell me which app to open."
        ok, msg = open_app(app_name)
        return msg

    if name == "list_apps":
        if not app_names:
            return "No apps are in the library yet."
        return "\n".join(f"{i}) {app_name}" for i, app_name in enumerate(app_names, start=1))

    if name == "save_memory":
        memory_text = str(args.get("text", "")).strip()
        if memory_text:
            save_memory(memory_text)
        return "Noted."

    return "Tool not recognized."



def call_ollama_stream(system_prompt: str, user_message: str, model_name: str):
    payload = {
        "model": model_name,
        "prompt": user_message,
        "system": system_prompt,
        "stream": True,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": 0.75,
        },
    }
    return requests.post(OLLAMA_GENERATE_URL, json=payload, stream=True, timeout=240)



def build_conversation_prompt(user_message: str, tool_result: str | None = None) -> str:
    parts = []
    for m in conversation_history[-MAX_HISTORY:]:
        role = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{role}: {m['content']}")
    parts.append(f"User: {user_message}")
    if tool_result:
        parts.append(f"Tool result: {tool_result}")
        parts.append("Assistant: Use the tool result to answer the user naturally.")
    else:
        parts.append("Assistant:")
    return "\n".join(parts)



def stream_model_once(system_prompt: str, prompt_text: str, selected_model: str):
    res = call_ollama_stream(system_prompt, prompt_text, selected_model)
    res.raise_for_status()
    accumulated = ""
    try:
        for raw_line in res.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
            except Exception:
                continue
            delta = data.get("response", "")
            if delta:
                accumulated += delta
                yield delta
            if data.get("done"):
                break
    finally:
        try:
            res.close()
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
def serve_mobile():
    try:
        return (BASE_DIR / "mobile.html").read_text(encoding="utf-8")
    except Exception:
        return "<h1>/AI</h1><p>mobile.html not found next to main.py</p>"


@app.get("/models")
def get_models():
    models = get_installed_tool_models()
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else DEFAULT_MODEL)
    return JSONResponse({"models": models, "default": default_model})


@app.post("/hardware-stats")
def hardware_stats(body: HardwareBody):
    stats = get_hardware_stats_series(duration_sec=body.duration_sec, step_sec=body.step_sec)
    return JSONResponse(stats)


@app.post("/ask")
def ask_once(body: AskBody):
    user_message = (body.message or "").strip()
    if not user_message:
        return JSONResponse({"response": "Type something."})
    routed = try_local_tool_route(user_message)
    if routed is not None:
        return JSONResponse({"response": routed})
    return JSONResponse({"response": "Use /ask-stream from the updated UI."})


@app.post("/ask-stream")
def ask_stream(body: AskBody):
    user_message = (body.message or "").strip()
    selected_model = (body.model or DEFAULT_MODEL).strip() or DEFAULT_MODEL

    def event_stream():
        global conversation_history

        if not user_message:
            yield json_line({"type": "error", "message": "Type something."})
            return

        maybe_autosave_memory(user_message)

        routed = try_local_tool_route(user_message)
        if routed is not None:
            yield json_line({"type": "final", "text": routed})
            return

        music_names, _ = list_music()
        app_names, _ = list_apps()
        conversation_history.append({"role": "user", "content": user_message})
        if len(conversation_history) > MAX_HISTORY:
            conversation_history[:] = conversation_history[-MAX_HISTORY:]

        hardware_payload = None
        tool_result = None

        for round_idx in range(2):
            system_prompt = build_system_prompt(music_names, app_names, hardware_payload)
            prompt_text = build_conversation_prompt(user_message, tool_result)

            try:
                chunks = []
                for delta in stream_model_once(system_prompt, prompt_text, selected_model):
                    chunks.append(delta)
                    yield json_line({"type": "token", "delta": delta, "round": round_idx})
                accumulated = "".join(chunks).strip()
            except Exception as e:
                yield json_line({"type": "error", "message": f"Server couldn't reach Ollama. Is it running? ({str(e)})"})
                return

            tool_call = parse_tool_call(accumulated)
            if not tool_call:
                conversation_history.append({"role": "assistant", "content": accumulated})
                if len(conversation_history) > MAX_HISTORY:
                    conversation_history[:] = conversation_history[-MAX_HISTORY:]
                yield json_line({"type": "done"})
                return

            if tool_call["name"] == "show_hardwareStats":
                duration_sec = float(tool_call["args"].get("duration_sec", 3) or 3)
                step_sec = float(tool_call["args"].get("step_sec", 0.5) or 0.5)
                hardware_payload = get_hardware_stats_series(duration_sec=duration_sec, step_sec=step_sec)
                tool_result = (
                    "Hardware telemetry captured successfully.\n" +
                    hardware_context_block(hardware_payload)
                )
                yield json_line({"type": "clear"})
                yield json_line({"type": "hardware", "payload": hardware_payload})
                continue

            tool_text = execute_non_hardware_tool(tool_call, music_names, app_names)
            conversation_history.append({"role": "assistant", "content": tool_text})
            if len(conversation_history) > MAX_HISTORY:
                conversation_history[:] = conversation_history[-MAX_HISTORY:]
            yield json_line({"type": "replace", "text": tool_text})
            yield json_line({"type": "done"})
            return

        fallback = "I captured the hardware stats, but the follow-up response loop hit its limit."
        conversation_history.append({"role": "assistant", "content": fallback})
        yield json_line({"type": "replace", "text": fallback})
        yield json_line({"type": "done"})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
