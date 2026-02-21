import time
import os
import json
from google.genai import types
from google import genai
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"


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
                        k: types.Schema(type=v["type"].upper(
                        ), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

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


# def generate_hybrid(messages, tools, confidence_threshold=0.99):

#     # cache the local mode
#     if not hasattr(generate_hybrid, "_model"):
#         generate_hybrid._model = cactus_init(functiongemma_path)

#     model = generate_hybrid._model

#     cactus_tools = [{"type": "function", "function": t} for t in tools]

#     raw_str = cactus_complete(
#         model,
#         [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
#         tools=cactus_tools,
#         force_tools=True,
#         max_tokens=256,
#         stop_sequences=["<|im_end|>", "<end_of_turn>"],
#     )

#     # reset
#     cactus_reset(model)

#     try:
#         raw = json.loads(raw_str)
#         local = {
#             "function_calls": raw.get("function_calls", []),
#             "total_time_ms": raw.get("total_time_ms", 0),
#             "confidence": raw.get("confidence", 0),
#         }
#     except json.JSONDecodeError:
#         local = {
#             "function_calls": [],
#             "total_time_ms": 0,
#             "confidence": 0,
#         }

#     if local["confidence"] >= confidence_threshold:
#         local["source"] = "on-device"
#         return local

#     cloud = generate_cloud(messages, tools)
#     cloud["source"] = "cloud (fallback)"
#     cloud["local_confidence"] = local["confidence"]
#     cloud["total_time_ms"] += local["total_time_ms"]
#     return cloud

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    # cache the local model
    if not hasattr(generate_hybrid, "_model"):
        generate_hybrid._model = cactus_init(functiongemma_path)

    model = generate_hybrid._model

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    # reset state
    cactus_reset(model)

    try:
        raw = json.loads(raw_str)
        local = {
            "function_calls": raw.get("function_calls", []),
            "total_time_ms": raw.get("total_time_ms", 0),
            "confidence": raw.get("confidence", 0),
        }
    except json.JSONDecodeError:
        local = {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }
    tool_by_name = {t.get("name"): t for t in tools}

    def _type_ok(val, json_type: str) -> bool:
        if json_type == "string":
            return isinstance(val, str)
        if json_type == "integer":
            return isinstance(val, int) and not isinstance(val, bool)
        if json_type == "number":
            return (isinstance(val, (int, float)) and not isinstance(val, bool))
        if json_type == "boolean":
            return isinstance(val, bool)
        if json_type == "object":
            return isinstance(val, dict)
        if json_type == "array":
            return isinstance(val, list)
        return False

    def _call_valid(call: dict) -> bool:
        name = call.get("name")
        if not name or name not in tool_by_name:
            return False

        args = call.get("arguments", {})
        if not isinstance(args, dict):
            return False

        schema = tool_by_name[name].get("parameters", {})
        props = schema.get("properties", {}) or {}
        required = schema.get("required", []) or []

        # required keys must exist
        for k in required:
            if k not in args:
                return False

        # no unknown keys + type checks
        for k, v in args.items():
            if k not in props:
                return False
            expected_type = (props[k].get("type") or "").lower()
            if expected_type and not _type_ok(v, expected_type):
                return False

        return True

    local_calls = local.get("function_calls", [])
    if local_calls and all(_call_valid(c) for c in local_calls):
        local["source"] = "on-device"
        return local

    # fallback
    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(
            f"Local confidence (below threshold): {result['local_confidence']:.4f}")
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
