"""
Unified AI provider for Ollama, OpenAI, and Claude.

All feature functions request **structured** output: a JSON Schema is sent
to the provider (via response_format / tool_use / format) and the reply is
validated against a Pydantic model. If structured decoding fails for any
reason we fall back to best-effort JSON extraction.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, List, Optional, Type

import requests
from pydantic import BaseModel, ValidationError

from services.ai_validator import (
    ClipPlan,
    ClipSuggestion,
    FillerReport,
    FocusDeletion,
    FocusPlan,
    validate_clip_plan,
    validate_filler_report,
    validate_focus_plan,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


class AIProvider:
    """Routes completion requests to the configured provider."""

    @staticmethod
    def list_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    @staticmethod
    def complete_structured(
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
    ) -> BaseModel:
        """
        Request a structured completion and validate against response_model.

        Each provider path returns a dict (already JSON-parsed). We then run
        it through the Pydantic model so the caller gets a typed object —
        and can trust the shape.
        """
        schema = response_model.model_json_schema()
        if provider == "ollama":
            raw = _ollama_structured(
                system_prompt, user_prompt, model or "llama3",
                base_url or "http://localhost:11434", temperature, schema,
            )
        elif provider == "openai":
            raw = _openai_structured(
                system_prompt, user_prompt, model or "gpt-4o", api_key or "",
                temperature, schema, response_model.__name__,
            )
        elif provider == "claude":
            raw = _claude_structured(
                system_prompt, user_prompt, model or "claude-sonnet-4-6",
                api_key or "", temperature, schema, response_model.__name__,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        try:
            return response_model.model_validate(raw)
        except ValidationError as e:
            logger.error(f"{response_model.__name__} validation failed: {e}; raw={str(raw)[:400]}")
            # Try one more time: maybe the model nested the payload.
            if isinstance(raw, dict):
                for v in raw.values():
                    if isinstance(v, dict):
                        try:
                            return response_model.model_validate(v)
                        except ValidationError:
                            continue
            raise


# ---------------------------------------------------------------------------
# Schema adapters
# ---------------------------------------------------------------------------


def _strictify_schema(schema: dict) -> dict:
    """
    Convert a Pydantic schema into an OpenAI strict-mode-compatible schema.

    OpenAI strict mode requires: additionalProperties: false, every property
    listed in required. Defaults are OK on the Pydantic side — we enforce
    presence at the wire level and let Pydantic supply defaults on parse.
    """
    s = copy.deepcopy(schema)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            # OpenAI strict mode rejects `default` on schema properties —
            # every field must be required and the model supplies a value.
            node.pop("default", None)
            node.pop("title", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                props = node.get("properties", {})
                node["required"] = list(props.keys())
                for k, v in props.items():
                    props[k] = walk(v)
            if "items" in node:
                node["items"] = walk(node["items"])
            if "$defs" in node:
                for k, v in node["$defs"].items():
                    node["$defs"][k] = walk(v)
            if "anyOf" in node:
                node["anyOf"] = [walk(x) for x in node["anyOf"]]
        return node

    return walk(s)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def _ollama_structured(
    system: str, user: str, model: str, base_url: str,
    temperature: float, schema: dict,
) -> Any:
    """
    Ollama 0.5+ accepts a JSON schema in `format`. Smaller models sometimes
    ignore it; we fall back to brace-extraction if validation fails.
    """
    body = {
        "model": model,
        "prompt": user,
        "system": system,
        "stream": False,
        "format": schema,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(f"{base_url}/api/generate", json=body, timeout=180)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        raise

    return _parse_json_forgiving(text)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


def _openai_structured(
    system: str, user: str, model: str, api_key: str,
    temperature: float, schema: dict, name: str,
) -> Any:
    try:
        from openai import OpenAI, BadRequestError
    except ImportError as e:
        raise RuntimeError("openai package not installed") from e

    client = OpenAI(api_key=api_key)
    strict_schema = _strictify_schema(schema)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": strict_schema},
    }
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
    }
    # Reasoning / GPT-5 models only accept default temperature.
    is_fixed_temp = model.startswith(("o1", "o3", "o4", "gpt-5"))
    if not is_fixed_temp:
        kwargs["temperature"] = temperature

    try:
        response = client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        msg = str(e)
        if "temperature" in msg and "temperature" in kwargs:
            kwargs.pop("temperature", None)
            response = client.chat.completions.create(**kwargs)
        elif "response_format" in msg or "json_schema" in msg:
            # Older model: fall back to plain JSON mode
            logger.warning(f"{model} doesn't support json_schema; falling back to json_object")
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["messages"][0]["content"] = system + "\n\nReturn JSON matching this schema:\n" + json.dumps(strict_schema)
            response = client.chat.completions.create(**kwargs)
        else:
            raise

    text = response.choices[0].message.content or ""
    return _parse_json_forgiving(text)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _claude_structured(
    system: str, user: str, model: str, api_key: str,
    temperature: float, schema: dict, name: str,
) -> Any:
    """
    Use tool use to force structured output. We invent a tool whose
    input_schema is the response schema, and force tool_choice to it.
    """
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package not installed") from e

    client = anthropic.Anthropic(api_key=api_key)
    tool = {
        "name": f"record_{name.lower()}",
        "description": f"Record the {name} result. You MUST call this tool with the full result.",
        "input_schema": schema,
    }
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return block.input
    # Shouldn't happen when tool_choice is forced, but be defensive.
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
    return _parse_json_forgiving(text)


# ---------------------------------------------------------------------------
# JSON fallback
# ---------------------------------------------------------------------------


def _parse_json_forgiving(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl >= 0:
            text = text[nl + 1:]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


# ---------------------------------------------------------------------------
# Feature: filler detection (multilingual, principle-based)
# ---------------------------------------------------------------------------


_FILLER_SYSTEM = """# Role
You are a precise, multilingual transcript-editing assistant.

# Task
Identify filler words and hesitations in the provided transcript. The
transcript may be in any language — detect the language from context and
apply the filler-word conventions of that language.

# Rules
- Flag only: hesitation sounds (e.g. English "um/uh", Spanish "este/eh",
  French "euh", German "äh", Japanese "eto/ano", Mandarin "呃/那个"),
  meaningless discourse markers used as crutches, and immediate stammer
  repetitions ("I I I", "the the").
- NEVER flag a word that carries meaning in context. Words like "like",
  "actually", "well", "so" are only fillers when they do not contribute
  to the sentence's meaning.
- Be conservative — when in doubt, do not flag.
- Include a confidence score (0.0-1.0) per flagged word. Use >=0.8 for
  obvious hesitations, 0.5-0.8 for context-dependent calls.
- Also flag any user-specified extra filler words with confidence 0.9.

# Output
Return a structured result matching the provided schema. Indices must
reference the integer IDs shown in the transcript. Do not invent indices.
"""


def detect_filler_words(
    transcript: str,
    words: List[dict],
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    custom_filler_words: Optional[str] = None,
) -> dict:
    """Returns a validated, json-serialisable FillerReport dict."""

    word_list = "\n".join(f"{w['index']}: {w['word']}" for w in words)
    extras = ""
    if custom_filler_words and custom_filler_words.strip():
        extras = f"\n\n# Extra user-specified fillers (always flag)\n{custom_filler_words.strip()}\n"

    user_prompt = f"""# Transcript
Detect the language, then flag fillers according to its conventions.

{extras}
# Words (index: token)
{word_list}
"""

    try:
        report = AIProvider.complete_structured(
            system_prompt=_FILLER_SYSTEM,
            user_prompt=user_prompt,
            response_model=FillerReport,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
        )
    except Exception as e:
        logger.error(f"Filler detection failed: {e}", exc_info=True)
        return FillerReport(warnings=[f"Detection failed: {e}"], needs_review=True).model_dump()

    validated = validate_filler_report(report, word_count=len(words))
    return validated.model_dump()


# ---------------------------------------------------------------------------
# Feature: clip suggestions (social-ready, duration-aware)
# ---------------------------------------------------------------------------


_CLIP_SYSTEM = """# Role
You are an expert social-media editor. You find short, shareable segments
inside long-form transcripts.

# Task
Find EVERY candidate clip that would plausibly work on social media —
not just the single best one. Longer videos should produce more clips.
Aim for up to ~15 total across all requested durations, but return fewer
if the material doesn't support it.

Each clip should:
- Be roughly the requested target duration (±30% is fine).
- Feel reasonably self-contained — not perfectly self-contained, but
  understandable to a viewer who hasn't watched the rest of the video.
- Have a clear hook (first sentence grabs attention) and a resolving
  ending (not cut off mid-thought).
- Start and end at word boundaries that align with sentence-like breaks
  when possible.
- NOT overlap substantially with another returned clip of the same
  duration. If two overlap, keep the stronger one.

# Rules
- Use ONLY the word indices and timestamps provided. Do not invent.
- If multiple target durations are requested, distribute clips across
  them — include several per duration when good material exists.
- If the transcript has no socially compelling material, return an empty
  clips array with rationale explaining why. Do NOT invent weak clips.
- Confidence score (0.0-1.0): >=0.7 for clearly shareable, 0.5-0.7 for
  decent but not standout, <0.5 for borderline (filtered downstream).
- Set target_duration on each clip to the duration bucket it belongs to.
- Titles should be short (2-6 words), descriptive, and mostly ASCII.
  Avoid emojis and punctuation that would be awkward in a filename.
- Mark plan needs_review=true if your best confidence is below 0.6.

# Output
Return a structured ClipPlan matching the provided schema.
"""


def create_clip_suggestion(
    transcript: str,
    words: List[dict],
    target_durations: Optional[List[int]] = None,
    target_duration: int = 60,
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict:
    """Returns a validated ClipPlan dict with clips keyed by word index."""

    durations = target_durations or [target_duration]
    durations = [int(d) for d in durations if 5 <= int(d) <= 600]
    if not durations:
        durations = [60]

    word_list = "\n".join(
        f"{w['index']}: \"{w['word']}\" ({w.get('start', 0):.2f}s-{w.get('end', 0):.2f}s)"
        for w in words
    )
    duration_line = ", ".join(f"{d}s" for d in durations)

    user_prompt = f"""# Target durations
{duration_line}

# Words (index: token  start-end)
{word_list}

Find the strongest social-media-ready segments at the target duration(s).
For each clip, set target_duration to the specific duration it matches.
"""

    try:
        plan = AIProvider.complete_structured(
            system_prompt=_CLIP_SYSTEM,
            user_prompt=user_prompt,
            response_model=ClipPlan,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.5,
        )
    except Exception as e:
        logger.error(f"Clip creation failed: {e}", exc_info=True)
        return ClipPlan(warnings=[f"Clip creation failed: {e}"], needs_review=True).model_dump()

    audio_dur = max((w.get("end", 0) for w in words), default=None)
    validated = validate_clip_plan(plan, words=words, audio_duration=audio_dur)
    return validated.model_dump()


# ---------------------------------------------------------------------------
# Feature: focus modes (redundancy, tighten, topic, q&a, key_points)
# ---------------------------------------------------------------------------


_FOCUS_MODE_RULES = {
    "redundancy": (
        "Remove near-duplicate sentences and repeated points. Keep the clearest "
        "instance of each idea. Do NOT remove content that merely relates to other "
        "content — only actual repetition."
    ),
    "tighten": (
        "Remove meta-commentary, throat-clearing, false starts, 'where was I' "
        "moments, and tangential asides. Keep the substantive through-line."
    ),
    "topic": (
        "Keep only content that is on-topic for the user-provided topic. Remove "
        "content that does not support, illustrate, or directly relate to the topic."
    ),
    "qa_extract": (
        "Keep only questions and their direct answers. Remove setup, tangents, and "
        "meta-commentary that isn't part of a Q&A pair."
    ),
    "key_points": (
        "Keep the speaker's thesis, main claims, and 1-2 supporting sentences per "
        "claim. Remove examples, digressions, and illustrative anecdotes."
    ),
}

_FOCUS_SYSTEM_TEMPLATE = """# Role
You are a precise video-editing assistant focused on tightening long-form
content without altering meaning.

# Task
Propose word-index ranges to DELETE from the transcript so that the
remaining text satisfies the selected mode.

# Mode: {mode}
{mode_rules}

# Rules
- Work in the source language. Detect it from the transcript.
- Propose contiguous ranges by [startIndex, endIndex] (inclusive). The ranges
  should align with sentence breaks whenever possible so the final cut feels
  natural — prefer deleting whole sentences over mid-sentence snips.
- Confidence 0.0-1.0 per range. Use >=0.7 when deletion is clearly safe;
  0.4-0.7 when subjective.
- Never delete more than 80% of the transcript. If the mode genuinely
  requires more, set needs_review=true and explain in summary.
- If nothing qualifies, return empty deletions and explain in summary.

# Output
Return a structured FocusPlan matching the provided schema.
"""


def focus_transcript(
    transcript: str,
    words: List[dict],
    mode: str,
    topic: Optional[str] = None,
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict:
    """
    Produce a FocusPlan — a set of word-index ranges to delete.

    Modes: redundancy | tighten | topic | qa_extract | key_points.
    For topic mode, the caller supplies a topic string.
    """
    if mode not in _FOCUS_MODE_RULES:
        raise ValueError(f"Unknown focus mode: {mode}")

    mode_rules = _FOCUS_MODE_RULES[mode]
    if mode == "topic":
        if not topic or not topic.strip():
            return FocusPlan(mode=mode, needs_review=True,
                             warnings=["topic mode requires a non-empty topic string"]).model_dump()
        mode_rules += f"\nUser-provided topic: \"{topic.strip()}\""

    system = _FOCUS_SYSTEM_TEMPLATE.format(mode=mode, mode_rules=mode_rules)

    word_list = "\n".join(f"{w['index']}: {w['word']}" for w in words)
    user_prompt = f"""# Words (index: token)
{word_list}

Return a FocusPlan. The `mode` field must be set to "{mode}".
"""

    try:
        plan = AIProvider.complete_structured(
            system_prompt=system,
            user_prompt=user_prompt,
            response_model=FocusPlan,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
        )
    except Exception as e:
        logger.error(f"Focus plan failed: {e}", exc_info=True)
        return FocusPlan(mode=mode, warnings=[f"Focus failed: {e}"], needs_review=True).model_dump()

    # Ensure the mode echoed back matches what we asked for — models sometimes drift.
    if plan.mode != mode:
        plan = FocusPlan(**{**plan.model_dump(), "mode": mode})

    validated = validate_focus_plan(plan, word_count=len(words))
    return validated.model_dump()
