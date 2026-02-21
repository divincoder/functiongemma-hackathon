"""
SOS Local: offline-first emergency assistant demo (terminal).

- Uses cactus_transcribe for voice input (offline).
- Routes intents to tools via generate_hybrid (local-first, cloud fallback).
- Offline mode: set --offline to block cloud calls (CACTUS_OFFLINE=1).
- How to use:
- Mic, offline: python sos_assistant.py --mic --mic-seconds 6 --target-lang es --offline
  - Mic, cloud allowed: python sos_assistant.py --mic --target-lang fr
  - File: python sos_assistant.py --audio path/to.wav --target-lang es
  - Text: python sos_assistant.py --text "Translate: I need help" --target-lang es
"""

import argparse
import os
import sys
import time
import tempfile
import sounddevice as sd
import soundfile as sf
from typing import List, Dict, Any

from google import genai
from google.genai import types

sys.path.insert(0, "cactus/python/src")

from cactus import cactus_init, cactus_destroy, cactus_transcribe  # noqa: E402
from main import generate_hybrid  # noqa: E402

# ---------- Tool schemas ----------
TOOLS = [
    {
        "name": "translate_text",
        "description": "Translate text to a target language",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "target_language": {"type": "string", "description": "Target language (e.g., es, fr)"},
            },
            "required": ["text", "target_language"],
        },
    },
    {
        "name": "compose_sos",
        "description": "Draft a short emergency/SOS message from situation details",
        "parameters": {
            "type": "object",
            "properties": {
                "situation": {"type": "string", "description": "What is happening"},
                "location": {"type": "string", "description": "Location description"},
            },
            "required": ["situation"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to a contact",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "start_safety_timer",
        "description": "Start a safety timer; when it ends, alert or remind the user",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer"},
            },
            "required": ["minutes"],
        },
    },
    {
        "name": "get_help_instructions",
        "description": "Provide quick emergency guidance for a given location and emergency type",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "emergency_type": {"type": "string"},
            },
            "required": ["emergency_type"],
        },
    },
    {
        "name": "save_emergency_note",
        "description": "Persist a short note locally for later reference",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
]


# ---------- Tool implementations (local stubs) ----------
def translate_text(args: Dict[str, Any], offline: bool) -> str:
    text = args.get("text", "")
    target = args.get("target_language", "es")
    if offline or not os.environ.get("GEMINI_API_KEY"):
        return f"[offline stub→{target}] {text}"
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(role="user", parts=[types.Part(text=f"Translate into {target}:\n{text}")])
            ],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        for cand in resp.candidates:
            for part in cand.content.parts:
                if part.text:
                    return part.text.strip()
    except Exception:
        return f"[translation error→{target}] {text}"
    return f"[translation unavailable→{target}] {text}"


def compose_sos(args: Dict[str, Any]) -> str:
    situation = args.get("situation", "Emergency")
    location = args.get("location", "unknown location")
    return f"SOS: {situation}. Location: {location}. Please send help."


def send_message(args: Dict[str, Any]) -> str:
    recipient = args.get("recipient", "contact")
    message = args.get("message", "")
    return f"Sent to {recipient}: {message}"


def start_safety_timer(args: Dict[str, Any]) -> str:
    minutes = args.get("minutes", 5)
    return f"Safety timer started for {minutes} minutes (simulation)."


def get_help_instructions(args: Dict[str, Any]) -> str:
    emergency = args.get("emergency_type", "general emergency")
    location = args.get("location", "your area")
    return (
        f"Checklist for {emergency} at {location}: "
        "1) Stay safe; 2) Call local emergency number; 3) Provide location; 4) Follow dispatcher instructions."
    )


def save_emergency_note(args: Dict[str, Any]) -> str:
    text = args.get("text", "")
    path = "sos_notes.txt"
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    return f"Saved note to {path}"


TOOL_IMPLS = {
    "translate_text": lambda a, offline: translate_text(a, offline),
    "compose_sos": lambda a, offline: compose_sos(a),
    "send_message": lambda a, offline: send_message(a),
    "start_safety_timer": lambda a, offline: start_safety_timer(a),
    "get_help_instructions": lambda a, offline: get_help_instructions(a),
    "save_emergency_note": lambda a, offline: save_emergency_note(a),
}


# ---------- Input helpers ----------
def transcribe_audio(audio_path: str) -> str:
    model = cactus_init("cactus/weights/whisper-small")
    prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    raw = cactus_transcribe(model, audio_path, prompt=prompt)
    cactus_destroy(model)
    try:
        return raw["response"] if isinstance(raw, dict) else raw
    except Exception:
        return str(raw)


def build_messages(text: str, target_language: str) -> List[Dict[str, str]]:
    system = {
        "role": "system",
        "content": "You are SOS Local, an emergency assistant. Prefer concise tool calls. Use translate_text when asked to translate.",
    }
    user = {"role": "user", "content": text + (f" Translate to {target_language}." if target_language else "")}
    return [system, user]


# ---------- Runner ----------
def run(text: str, offline: bool, target_language: str, debug: bool = False):
    if offline:
        os.environ["CACTUS_OFFLINE"] = "1"
    else:
        os.environ.pop("CACTUS_OFFLINE", None)

    messages = build_messages(text, target_language)
    start = time.time()
    result = generate_hybrid(messages, TOOLS)
    latency = (time.time() - start) * 1000

    print(f"\nRouting source: {result.get('source', 'unknown')}  |  total_time_ms={latency:.1f}")
    if debug:
        print(f"Raw result: {result}")
    if result.get("function_calls"):
        print("Tool calls returned:")
        for call in result["function_calls"]:
            print(f"  - {call.get('name')} args={call.get('arguments', {})}")
    calls = result.get("function_calls", [])
    if not calls:
        print("No function calls returned.")
        return

    for call in calls:
        name = call.get("name")
        args = call.get("arguments", {})
        impl = TOOL_IMPLS.get(name)
        if impl:
            output = impl(args, offline)
            print(f"\n→ Executed {name} with args {args}\n  Result: {output}")
        else:
            print(f"\n→ No local implementation for {name}; args={args}")


def main():
    parser = argparse.ArgumentParser(description="SOS Local: Offline-first emergency assistant")
    parser.add_argument("--text", help="Text input (fallback if no audio)", default="")
    parser.add_argument("--audio", help="Path to audio file for transcription (optional)")
    parser.add_argument("--mic", action="store_true", help="Record audio from microphone instead of file")
    parser.add_argument("--mic-seconds", type=int, default=8, help="Seconds to record from mic (default: 8)")
    parser.add_argument("--target-lang", default="", help="Target language code for translation, e.g., es, fr")
    parser.add_argument("--offline", action="store_true", help="Block cloud calls (set CACTUS_OFFLINE=1)")
    parser.add_argument("--debug", action="store_true", help="Print raw routing result")
    args = parser.parse_args()

    if args.mic:
        duration = args.mic_seconds
        sr = 16000
        print(f"Recording {duration}s from default microphone at {sr} Hz ...")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            tmp_path = tmp.name
        print(f"Saved temp audio: {tmp_path}")
        print("Transcribing audio locally...")
        text = transcribe_audio(tmp_path)
        os.unlink(tmp_path)
    elif args.audio:
        print("Transcribing audio locally...")
        text = transcribe_audio(args.audio)
    else:
        text = args.text or "Emergency: I need help."

    run(text, offline=args.offline, target_language=args.target_lang, debug=args.debug)


if __name__ == "__main__":
    main()
