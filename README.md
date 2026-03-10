# AI Server

A local AI assistant server that runs on a personal gaming PC.

This project turns a gaming PC into a **local AI server** capable of running LLMs, executing tools, and providing a web-based interface.

---

## Features

- Local AI assistant
- Streaming responses
- Tool execution
- Hardware telemetry graphs
- Screen vision (AI can describe your screen)
- App launcher
- Music control
- Memory system
- Web UI

---

## Requirements

- Python 3.10+
- Ollama installed
- Windows (for app launching features)

Python libraries:

```
fastapi
uvicorn
requests
pydantic
psutil
pillow
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MightyXdash/AI-server.git
cd AI-server
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama:

```
https://ollama.com
```

Pull a model:

```bash
ollama pull llama3
```

---

## Run the server

```bash
python main.py
```

Open the interface:

```
http://localhost:6967
```

---

## Project Structure

```
AI-server
├── main.py
├── tools.py
├── mobile.html
├── app_library.json
├── memories.jsonl
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Tools

The assistant can use tools such as:

```
list_music
play_music
open_app
screen_vision
save_memory
show_hardwareStats
```


## License

MIT License
