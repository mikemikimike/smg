# crates/symphony/scripts/render_gpt_oss_turns.py
"""Render the gpt-oss chat template for the turn shapes a tool-calling conversation takes, and
write each assistant turn as the model writes it, with the message it stands for, to the fixture
`tests/fixtures/rendered/gpt_oss.json`, which `tests/rendered_turns.rs` replays through the
`gpt_oss` table at every cut.

The template is `chat_template.jinja` of `openai/gpt-oss-20b`; gpt-oss-120b ships the same file,
and the safeguard checkpoints' differs in how it escapes its newlines alone, which `--also` checks
by rendering every case with it too. It is rendered as transformers renders a chat template: a
Jinja environment with `trim_blocks`, `lstrip_blocks` and the `loopcontrols` extension, `tojson`
as `json.dumps(..., ensure_ascii=False)`, `raise_exception`, and `strftime_now` pinned to one day
so that the prompt is the same on every run. A call's arguments are given as an object, which is
how a serving layer hands them to the template (the API carries them as a string, loaded before
rendering).

A turn is what the model writes after the generation prompt: the history rendered with the turn's
messages, minus the same history rendered without them and with the generation prompt, minus the
stop token that ends the turn (`<|return|>` after an answer, `<|call|>` after a call), which the
stop decoder hides from the parser. A history of two calls keeps the `<|call|>` between them as
text, since the template writes each call as a message of its own; a live stream stops at the
first. The message a turn stands for is the one given to the template: its `thinking` is the
thought, its `content` the answer, its `tool_calls` the calls, read back as names and arguments.

Usage: render_gpt_oss_turns.py <chat_template.jinja> [--also <chat_template.jinja> ...] [--check]
  With --check nothing is written: the fixture is rendered again and compared byte for byte, and
  the script exits 1 on a difference. Run with Python 3.12 and Jinja2 3.1.
"""

import hashlib
import json
import sys
from pathlib import Path

from jinja2 import Environment
from jinja2.ext import loopcontrols

OUT = Path(__file__).parents[1] / "tests" / "fixtures" / "rendered" / "gpt_oss.json"
REPOSITORY = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
TEMPLATE_SHA256 = "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146"
ALSO = ["openai/gpt-oss-120b", "openai/gpt-oss-safeguard-20b", "openai/gpt-oss-safeguard-120b"]
GENERATION_PROMPT = "<|start|>assistant"
STOPS = ("<|return|>", "<|call|>")
DAY = "2026-01-01"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "A stock's price.",
            "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "meta": {"type": "object"},
                },
            },
        },
    },
]


def call(name, arguments):
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


def assistant(content=None, thinking=None, tool_calls=None):
    message = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if thinking is not None:
        message["thinking"] = thinking
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return message


USER = {"role": "user", "content": "Weather in Paris?"}
TOOL_RESULT = {"role": "tool", "content": '{"temperature": 18}'}
PARIS = call("get_weather", {"city": "Paris"})

# Each case: its name and the messages; the turn is the run of assistant messages at the end,
# after the last user or tool message.
CASES = [
    ("answer", [USER, assistant(content="It is sunny in Paris.")]),
    (
        "thought-answer",
        [
            USER,
            assistant(
                thinking="The user asks about Paris; no tool is needed.",
                content="It is sunny in Paris.",
            ),
        ],
    ),
    ("call", [USER, assistant(tool_calls=[PARIS])]),
    ("thought-call", [USER, assistant(thinking="Need the weather.", tool_calls=[PARIS])]),
    (
        "calls-two",
        [
            USER,
            assistant(tool_calls=[PARIS]),
            assistant(tool_calls=[call("get_weather", {"city": "Rome", "days": 3})]),
        ],
    ),
    (
        "thought-calls-three",
        [
            USER,
            assistant(thinking="Paris, Rome and a note.", tool_calls=[PARIS]),
            assistant(tool_calls=[call("get_weather", {"city": "Rome", "unit": "celsius"})]),
            assistant(tool_calls=[call("save_note", {"title": "Trip", "tags": ["a", "b"]})]),
        ],
    ),
    ("call-empty-arguments", [USER, assistant(tool_calls=[call("get_stock_price", {})])]),
    (
        "call-nested-unicode",
        [
            USER,
            assistant(
                tool_calls=[
                    call(
                        "save_note",
                        {
                            "title": "計画 🌍",
                            "body": 'a "q" \\ b\nc',
                            "tags": ["a", "b"],
                            "meta": {"nested": {"deep": [1, 2.5, True, None]}},
                        },
                    )
                ]
            ),
        ],
    ),
    (
        "answer-after-tool-result",
        [
            USER,
            assistant(tool_calls=[PARIS]),
            TOOL_RESULT,
            assistant(content="It is 18 degrees in Paris."),
        ],
    ),
    (
        "thought-call-after-tool-result",
        [
            USER,
            assistant(tool_calls=[PARIS]),
            TOOL_RESULT,
            assistant(thinking="Rome too.", tool_calls=[call("get_weather", {"city": "Rome"})]),
        ],
    ),
]


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    """transformers' `tojson`: `json.dumps` with these defaults, no HTML escaping."""
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def raise_exception(message):
    raise ValueError(message)


def environment(template_text):
    env = Environment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: DAY
    return env.from_string(template_text)


def turn_start(messages):
    """Where the final run of assistant messages starts."""
    start = len(messages)
    while start > 0 and messages[start - 1]["role"] == "assistant":
        start -= 1
    return start


def render_case(template, name, messages):
    start = turn_start(messages)
    prompt = template.render(messages=messages[:start], tools=TOOLS, add_generation_prompt=True)
    full = template.render(messages=messages, tools=TOOLS, add_generation_prompt=False)
    assert prompt.endswith(GENERATION_PROMPT), f"{name}: the prompt ends with {prompt[-40:]!r}"
    assert full.startswith(prompt), f"{name}: the history does not start with the prompt"
    output = full[len(prompt) :]
    stop = next((s for s in STOPS if output.endswith(s)), None)
    assert stop, f"{name}: the turn ends with no stop token: {output[-40:]!r}"
    output = output[: -len(stop)]
    turn = messages[start:]
    reasoning = "".join(m.get("thinking", "") for m in turn)
    content = "".join(m.get("content", "") for m in turn if "tool_calls" not in m)
    calls = [
        {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
        for m in turn
        for c in m.get("tool_calls", [])
    ]
    tail = prompt[prompt.rfind("<|start|>user<|message|>") :]
    return {
        "name": name,
        "messages": turn,
        "tail": tail,
        "output": output,
        "reasoning": reasoning,
        "content": content,
        "calls": calls,
    }


def render_all(template):
    return [render_case(template, name, messages) for name, messages in CASES]


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    check = "--check" in argv
    also = []
    if "--also" in argv:
        also = args[1:]
        args = args[:1]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    text = Path(args[0]).read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert digest == TEMPLATE_SHA256, f"{args[0]}: sha256 {digest}, not the recorded template's"
    turns = render_all(environment(text))
    for other in also:
        other_turns = render_all(environment(Path(other).read_text(encoding="utf-8")))
        assert other_turns == turns, f"{other} renders the cases differently"
    fixture = {
        "table": "gpt_oss",
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "file": "chat_template.jinja",
            "sha256": TEMPLATE_SHA256,
            "also": ALSO,
            "rendered_as": "transformers renders a chat template: trim_blocks, lstrip_blocks, "
            "loopcontrols, tojson = json.dumps(ensure_ascii=False), strftime_now pinned to " + DAY,
            "script": "scripts/render_gpt_oss_turns.py",
        },
        "notes": [
            "A turn is the history rendered with the turn's messages minus the generation prompt, "
            "minus the stop token that ends it (<|return|> after an answer, <|call|> after a "
            "call), which the stop decoder hides from the parser.",
            "The template writes a call's recipient in the header's role section "
            "(`<|start|>assistant to=functions.NAME<|channel|>commentary json<|message|>`) and "
            "each call as a message of its own, so a turn of two calls holds the first's <|call|> "
            "as text; a live stream stops at the first.",
            "The template renders the thought of a message with calls only while no later "
            "message answers; an answer's thought is dropped from the history, so a thought stands "
            "in a fixture only where the turn is the last one rendered.",
            "The tail is the prompt from its last user message; the replay starts after the "
            "prompt's last <|start|>assistant.",
        ],
        "tools": TOOLS,
        "turns": turns,
    }
    rendered = json.dumps(fixture, ensure_ascii=False, indent=1) + "\n"
    if check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else None
        if current != rendered:
            print(f"{OUT}: differs from what the template renders", file=sys.stderr)
            return 1
        print(f"{OUT}: as rendered ({len(turns)} turns)")
        return 0
    OUT.write_text(rendered, encoding="utf-8")
    print(f"{OUT}: {len(turns)} turns written")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
