import os
import json
import argparse
import datetime

from main import generate_hybrid


TOOLS = [
    {
        "name": "compose_sos",
        "description": "Compose a short emergency message for a given situation and optional location and language.",
        "parameters": {
            "type": "object",
            "properties": {
                "situation": {"type": "string", "description": "What happened / what you need (urgent)"},
                "location": {"type": "string", "description": "Where you are (city/area or 'unknown')"},
                "language": {"type": "string", "description": "Language name for the message (e.g., English, Spanish)"},
            },
            "required": ["situation"],
        },
    },
    {
        "name": "translate_text",
        "description": "Translate the given text into a target language.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "target_language": {"type": "string", "description": "Language to translate into (e.g., Spanish)"},
            },
            "required": ["text", "target_language"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to a recipient (simulated: prints and logs).",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Recipient name"},
                "message": {"type": "string", "description": "Message to send"},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "start_safety_timer",
        "description": "Start a safety check-in timer (simulated).",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Minutes until check-in"},
            },
            "required": ["minutes"],
        },
    },
    {
        "name": "save_note",
        "description": "Save an emergency note locally to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Note text to save"},
            },
            "required": ["text"],
        },
    },
]


STATE = {
    "last_message": None,
    "last_translation": None,
}

LOG_DIR = "sos_logs"


def _ensure_logs():
    os.makedirs(LOG_DIR, exist_ok=True)


def exec_compose_sos(args):
    situation = (args.get("situation") or "").strip()
    location = (args.get("location") or "unknown").strip()
    language = (args.get("language") or "English").strip()

    msg = f"EMERGENCY: {situation}. My location: {location}. Please help."
    if language.lower() != "english":
        msg = f"[{language}] {msg}"

    STATE["last_message"] = msg
    print(f"\n[compose_sos]\n{msg}")
    return msg


def exec_translate_text(args):
    text = (args.get("text") or "").strip()
    target = (args.get("target_language") or "").strip()

    out = f"[{target}] {text}"

    STATE["last_translation"] = out
    print(f"\n[translate_text]\n{out}")
    return out


def exec_send_message(args):
    recipient = (args.get("recipient") or "").strip()
    message = (args.get("message") or "").strip()
    ts = datetime.datetime.now().isoformat(timespec="seconds")

    _ensure_logs()
    log_path = os.path.join(LOG_DIR, "sent_messages.log")
    with open(log_path, "a") as f:
        f.write(f"{ts} | to={recipient} | {message}\n")

    print(
        f"\n[send_message]\nTo: {recipient}\n{message}\n(logged to {log_path})")
    return True


def exec_start_safety_timer(args):
    minutes = args.get("minutes", None)
    print(
        f"\n[start_safety_timer]\nSafety timer started for {minutes} minutes (simulated).")
    return True


def exec_save_note(args):
    text = (args.get("text") or "").strip()
    _ensure_logs()
    path = os.path.join(LOG_DIR, "notes.txt")
    with open(path, "a") as f:
        f.write(text + "\n---\n")
    print(f"\n[save_note]\nSaved note to {path}")
    return path


EXECUTORS = {
    "compose_sos": exec_compose_sos,
    "translate_text": exec_translate_text,
    "send_message": exec_send_message,
    "start_safety_timer": exec_start_safety_timer,
    "save_note": exec_save_note,
}


def transcribe_audio(audio_path: str) -> str:
    """
    Uses Cactus Whisper model locally to transcribe an audio file.
    Requires weights at: cactus/weights/whisper-small (or change path below).
    """
    try:
        from cactus import cactus_init, cactus_transcribe, cactus_destroy
    except Exception:
        return ""

    whisper_path = "cactus/weights/whisper-small"
    if not os.path.exists(whisper_path):
        return ""

    whisper = cactus_init(whisper_path)
    prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
    raw = cactus_transcribe(whisper, audio_path, prompt=prompt)
    cactus_destroy(whisper)

    try:
        return (json.loads(raw).get("response") or "").strip()
    except Exception:
        return ""


def run_query(user_text: str, offline: bool = False):
    if offline:
        os.environ.pop("GEMINI_API_KEY", None)

    messages = [{"role": "user", "content": user_text}]
    result = generate_hybrid(messages, TOOLS)

    print("\n" + "=" * 60)
    print("SOS Local — Hybrid Result")
    print("=" * 60)
    print(f"Input: {user_text}")
    print(f"Source: {result.get('source')}")
    print(f"Total time: {result.get('total_time_ms', 0):.2f}ms")
    if "local_confidence" in result:
        print(f"Local confidence: {result.get('local_confidence')}")

    calls = result.get("function_calls", [])
    print("\nFunction calls:")
    print(json.dumps(calls, indent=2))

    print("\nExecuting calls:")
    for c in calls:
        name = c.get("name")
        args = c.get("arguments", {})
        fn = EXECUTORS.get(name)
        if not fn:
            print(f"[skip] No executor for tool: {name}")
            continue
        try:
            fn(args)
        except Exception as e:
            print(f"[error] Tool {name} failed: {e}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="SOS Local — Offline Emergency Assistant (terminal)")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_text = sub.add_parser("text", help="Run with typed input")
    p_text.add_argument(
        "query", type=str, help="Your request (e.g., 'Compose SOS and send to Maria')")
    p_text.add_argument("--offline", action="store_true",
                        help="Force local-only (disable cloud)")

    p_voice = sub.add_parser(
        "voice", help="Run with an audio file (WAV recommended)")
    p_voice.add_argument("--audio", required=True, help="Path to audio file")
    p_voice.add_argument("--offline", action="store_true",
                         help="Force local-only (disable cloud)")

    args = parser.parse_args()

    if args.mode == "text":
        run_query(args.query, offline=args.offline)
        return

    if args.mode == "voice":
        transcript = transcribe_audio(args.audio)
        if not transcript:
            print(
                "Could not transcribe audio. Make sure whisper weights exist at cactus/weights/whisper-small.")
            print("To download: cactus download openai/whisper-small --reconvert")
            return
        print("\nTranscript:")
        print(transcript)
        run_query(transcript, offline=args.offline)
        return


if __name__ == "__main__":
    main()
