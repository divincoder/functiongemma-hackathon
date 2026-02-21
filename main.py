
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    # Preserve full conversation (system + history) for better tool grounding.
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])])
        for m in messages
    ]
    # Add steering hint to encourage completeness and format (Gemini expects roles user/model).
    contents.insert(0, types.Content(role="user", parts=[types.Part(
        text="Instruction: Return function_calls with one entry per requested action. Use only provided tools. Extract argument values directly from the user's message — use short, exact values as they appear in the text. No prose."
    )]))

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


############## Hybrid Routing ##############

# Warm model: load once, reuse across calls
_warm_model = None


def _get_model():
    global _warm_model
    if _warm_model is None:
        _warm_model = cactus_init(functiongemma_path)
    return _warm_model


def _on_device_call(messages, tools, tool_rag_top_k=None, extra_system=None, temperature=None):
    """Run a single on-device inference using the warm model."""
    model = _get_model()
    cactus_reset(model)

    cactus_tools = [{"type": "function", "function": t} for t in tools]
    system_prompt = "You are a tool-calling assistant. Respond with exactly one function call in correct JSON format. Extract argument values directly from the user's message — use short, exact values, not paraphrases. No prose."
    if extra_system:
        system_prompt += " " + extra_system
    kwargs = dict(
        force_tools=True,
        max_tokens=512,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        confidence_threshold=0.01,
    )
    if tool_rag_top_k is not None:
        kwargs["tool_rag_top_k"] = tool_rag_top_k
    if temperature is not None:
        kwargs["temperature"] = temperature

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_prompt}] + messages,
        tools=cactus_tools,
        **kwargs,
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def _fix_args(calls, tools):
    """Schema-driven argument coercion — no tool-specific logic."""
    tool_map = {t["name"]: t for t in tools}

    def _coerce(prop, value):
        ptype = prop.get("type")
        if ptype == "integer":
            # Handle time-like strings ("7:30" → 7) and numeric strings
            if isinstance(value, str) and ":" in value:
                try:
                    return abs(int(value.split(":")[0].strip()))
                except (ValueError, IndexError):
                    pass
            try:
                return abs(int(float(value)))
            except (ValueError, TypeError):
                return value
        if ptype == "number":
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        if ptype == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lower = value.strip().lower()
                if lower in ["true", "yes", "1"]:
                    return True
                if lower in ["false", "no", "0"]:
                    return False
            return value
        if ptype == "string" and isinstance(value, str):
            # Strip surrounding quotes, trailing punctuation, and whitespace
            cleaned = value.strip().strip('"').strip("'").strip()
            cleaned = cleaned.rstrip(".")
            return cleaned
        # Enum coercion: case-insensitive match
        if "enum" in prop and isinstance(value, str):
            for option in prop["enum"]:
                if value.lower() == str(option).lower():
                    return option
        return value

    for call in calls:
        name = call.get("name", "")
        if name not in tool_map:
            continue
        schema = tool_map[name]["parameters"]
        props = schema.get("properties", {})
        args = call.get("arguments", {})

        # Coerce existing arguments by schema type
        for k, v in list(args.items()):
            if k in props:
                args[k] = _coerce(props[k], v)

        # Fill missing required args that have a default in the schema
        for req in schema.get("required", []):
            if req not in args and req in props:
                default = props[req].get("default")
                if default is not None:
                    args[req] = default


def _valid_calls(calls, tools):
    """Check all function calls have valid names, required params, and correct types."""
    type_check = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
    }
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        name = call.get("name", "")
        if name not in tool_map:
            return False
        schema = tool_map[name]["parameters"]
        required = schema.get("required", [])
        props = schema.get("properties", {})
        args = call.get("arguments", {})
        if any(r not in args for r in required):
            return False
        # Validate argument types against schema
        for k, v in args.items():
            if k in props:
                expected_type = props[k].get("type")
                checker = type_check.get(expected_type)
                if checker and not checker(v):
                    return False
    return True


def _has_all_required(calls, tools):
    """Return True only if every call includes all required args."""
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        name = call.get("name", "")
        if name not in tool_map:
            return False
        required = tool_map[name]["parameters"].get("required", [])
        args = call.get("arguments", {})
        if any(r not in args for r in required):
            return False
    return True


def _split_actions(text):
    """Split a multi-action query into individual action segments with pronoun resolution."""
    import re
    if ", and " in text:
        last_split = text.rsplit(", and ", 1)
        segments = last_split[0].split(", ")
        segments.append(last_split[1])
    elif " and " in text:
        segments = text.split(" and ")
    else:
        segments = [text]
    segments = [s.strip().rstrip(".") for s in segments if len(s.strip()) > 3]

    # Lightweight coreference resolution: carry forward proper nouns
    last_person = None
    last_location = None
    resolved = []
    for seg in segments:
        # Extract person name (capitalized word after common prepositions/verbs)
        person_match = re.search(
            r"(?:to|from|for|text|message|call|find|look up|search for|contact)\s+([A-Z][a-z]+)",
            seg,
        )
        if person_match:
            last_person = person_match.group(1)

        # Extract location (capitalized words after "in")
        loc_match = re.search(r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", seg)
        if loc_match:
            last_location = loc_match.group(1)

        # Resolve pronouns using previously extracted entities
        if last_person:
            seg = re.sub(r"\bhim\b", last_person, seg, flags=re.IGNORECASE)
            seg = re.sub(r"\bher\b", last_person, seg, flags=re.IGNORECASE)
            seg = re.sub(r"\bthem\b", last_person, seg, flags=re.IGNORECASE)
        if last_location:
            seg = re.sub(r"\bthere\b", f"in {last_location}", seg, flags=re.IGNORECASE)

        resolved.append(seg)

    return resolved


def _is_multi_action(text):
    """Check if text likely contains multiple action requests."""
    lower = text.lower()
    return " and " in lower or lower.count(",") > 1


def generate_hybrid(messages, tools, confidence_threshold=0.7):
    """Hybrid inference: on-device with structural validation, cloud fallback."""
    start = time.time()
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    multi = _is_multi_action(user_text) and len(tools) > 1
    segments = _split_actions(user_text) if multi else None

    # Adaptive temperature based on tool count
    base_temp = 0.01 if len(tools) == 1 else 0.10 if len(tools) <= 3 else 0.20
    tool_names = ", ".join(t["name"] for t in tools)

    # ==== MULTI-ACTION PATH ====
    if multi:
        # Quick attempt: try whole query on-device in one shot (only for 2-segment)
        if len(segments) == 2:
            whole = _on_device_call(
                messages,
                tools,
                tool_rag_top_k=0,
                extra_system=f"Return ALL requested function calls. Available tools: {tool_names}. Fill all required arguments for each call. No prose.",
                temperature=0.2,
            )
            _fix_args(whole["function_calls"], tools)
            if (
                whole["function_calls"]
                and len(whole["function_calls"]) >= 2
                and _valid_calls(whole["function_calls"], tools)
                and _has_all_required(whole["function_calls"], tools)
            ):
                return {
                    "function_calls": whole["function_calls"],
                    "total_time_ms": (time.time() - start) * 1000,
                    "source": "on-device",
                }

        # Segment-by-segment decomposition with context
        all_calls = []
        all_ok = True
        for segment in segments:
            sub = _on_device_call(
                [{"role": "user", "content": segment}],
                tools,
                tool_rag_top_k=0,
                extra_system=f"Original request: '{user_text}'. Available tools: {tool_names}. Handle ONLY this sub-task: '{segment}'. Use exactly one tool call. Fill all required arguments. No prose.",
                temperature=0.2,
            )
            _fix_args(sub["function_calls"], tools)
            if sub["function_calls"] and _valid_calls(sub["function_calls"], tools):
                all_calls.extend(sub["function_calls"])
            else:
                all_ok = False

        if all_ok and all_calls and len(all_calls) >= len(segments) and _has_all_required(all_calls, tools):
            return {
                "function_calls": all_calls,
                "total_time_ms": (time.time() - start) * 1000,
                "source": "on-device",
            }

        # Single cloud call, merge with any on-device successes
        cloud = generate_cloud(messages, tools)
        _fix_args(cloud["function_calls"], tools)
        if all_calls:
            merged = list(all_calls)
            on_device_names = {c.get("name") for c in all_calls}
            for c in cloud["function_calls"]:
                if c.get("name") not in on_device_names:
                    merged.append(c)
            if _valid_calls(merged, tools) and _has_all_required(merged, tools) and len(merged) >= len(segments):
                return {
                    "function_calls": merged,
                    "total_time_ms": (time.time() - start) * 1000,
                    "source": "on-device",
                }

        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] = (time.time() - start) * 1000
        return cloud

    # ==== SINGLE-INTENT PATH (max 2 on-device attempts) ====

    # Attempt 1: Focused call with tool guidance
    result = _on_device_call(
        messages,
        tools,
        tool_rag_top_k=min(2, len(tools)),
        extra_system=f"Available tools: {tool_names}. Pick the most relevant tool. Fill all required arguments. No prose.",
        temperature=base_temp,
    )
    _fix_args(result["function_calls"], tools)

    if result["function_calls"] and _valid_calls(result["function_calls"], tools):
        return {
            "function_calls": result["function_calls"],
            "total_time_ms": (time.time() - start) * 1000,
            "source": "on-device",
        }

    # Attempt 2: all tools visible, different temperature
    if len(tools) == 1:
        retry = _on_device_call(
            messages,
            tools,
            tool_rag_top_k=0,
            extra_system=f"You MUST call '{tools[0]['name']}' with all required arguments. No prose.",
            temperature=0.1,
        )
    else:
        retry = _on_device_call(
            messages,
            tools,
            tool_rag_top_k=0,
            extra_system=f"Available tools: {tool_names}. Use the most relevant tool. Fill all required arguments. No prose.",
            temperature=min(base_temp + 0.15, 0.35),
        )
    _fix_args(retry["function_calls"], tools)

    if retry["function_calls"] and _valid_calls(retry["function_calls"], tools):
        return {
            "function_calls": retry["function_calls"],
            "total_time_ms": (time.time() - start) * 1000,
            "source": "on-device",
        }

    # Cloud fallback
    cloud = generate_cloud(messages, tools)
    _fix_args(cloud["function_calls"], tools)
    cloud["source"] = "cloud (fallback)"
    cloud["total_time_ms"] = (time.time() - start) * 1000
    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
