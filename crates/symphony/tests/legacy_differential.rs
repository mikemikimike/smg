//! The legacy parsers and the tables, side by side on the recorded corpus.
//!
//! The gateway reads a model's output with the tool and reasoning parser crates by default and
//! with a table of this crate when an operator opts in. This test replays every recorded parse
//! case of every model whose table has a legacy parser for its family through both, on the same
//! bytes, and compares what the two say: the calls (name, and arguments as JSON values), the
//! content, the reasoning and the finish reason. Each case is read whole (the legacy
//! `detect_and_parse_reasoning` and `parse_complete_with_tools`, the table on one delta) and
//! streamed under the same cuts: at the boundaries of the tokens bellwether recorded with the case
//! (one token per delta, the shape an engine's stream has, and two of its random token plans) and
//! at characters (two of the seeded plans the corpus tests use and, up to
//! [`CHARACTER_PLAN_UP_TO`] bytes, a character at a time), the legacy incremental parsers with the
//! flush and the unstreamed text and arguments the gateway takes at the end of a stream. The
//! legacy side runs as the gateway runs it: the reasoning parser first, armed when the prompt
//! leaves the model inside its thought, then the tool parser on the text outside the thought when
//! the request declares tools (a list, empty or not, as the gateway reads it; a request with a
//! `tool_choice` is not routed and counted); whitespace-only content is absent, and `stop` after
//! a call is `tool_calls`.
//!
//! What the two say is compared at three strengths and every difference is classified against
//! the template reference bellwether recorded with the case: `bitwise` (the same strings), `equal`
//! (the same channels, the arguments equal as JSON values: the legacy parsers re-serialize them),
//! `equivalent` (the same once whitespace at the edges of the content and the reasoning is
//! trimmed: the template's separators, which the two paths keep differently); past that,
//! `symphony right` when the table says what the reference says and the legacy parsers do not,
//! `legacy right` the other way round, and `both wrong` when neither does. A case whose calls or
//! thought the legacy pair of its table cannot read ([`Legacy`]: a family with a reasoning parser
//! and no tool parser) is `out of reach` and not compared; a difference on one of the two probes
//! that put a marker inside text ([`PROBES`]) is `listed`. The run fails on a `legacy right` case:
//! the opt-in path must never read a recorded output worse than the path it replaces; and on a
//! table that panics. A legacy parser that panics is counted and reported, not a failure.
//!
//! The run is opt-in: `BELLWETHER_FIXTURES` points at the `fixtures/` directory of a bellwether
//! checkout; without it the test prints a skip notice and passes. `SYMPHONY_SAMPLE_EVERY=n`
//! replays every n-th case of each recorded set and the hand-written `common` set whole, as the
//! corpus tests do; `SYMPHONY_DIFFERENTIAL_SLUGS=a,b` restricts the run to those slugs. Every
//! model's table comes from the registry ([`for_model`]) by the model id its manifest names; the
//! legacy parsers are named per table ([`Legacy::of`]), and the way its template ends the
//! generation prompt per slug ([`PROMPTS`]), as the corpus tests state it. The per-case report,
//! one verdict line per case and the two sides and the reference of every difference, goes to
//! `legacy_differential.txt` under `BELLWETHER_REPORT_DIR`, or under the build's `target/tmp`;
//! stdout shows each model's summary and the per-table totals.

#[expect(
    dead_code,
    reason = "the integration tests share these helpers, and this one replays after a prompt only"
)]
mod common;

use std::{
    collections::BTreeMap,
    fmt, fs,
    io::{BufRead, BufReader, BufWriter, Write},
    num::NonZeroUsize,
    panic::{catch_unwind, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
};

use common::{chunkings, replay_after};
use openai_protocol::common::Tool;
use reasoning_parser::{ParserFactory as ReasoningFactory, ReasoningParser};
use serde::Deserialize;
use serde_json::Value;
use symphony::{
    adapt,
    formats::{for_model, table_for, Table},
    Declared, Engine, EngineFinish, Event, Format,
};
use tokio::runtime::{Builder, Runtime};
use tool_parser::{ParserFactory as ToolFactory, ToolParser};

const FIXTURES_ENV: &str = "BELLWETHER_FIXTURES";
const SAMPLE_ENV: &str = "SYMPHONY_SAMPLE_EVERY";
const SLUGS_ENV: &str = "SYMPHONY_DIFFERENTIAL_SLUGS";
const REPORT_DIR_ENV: &str = "BELLWETHER_REPORT_DIR";
const REPORT_FILE: &str = "legacy_differential.txt";
/// The hand-written set in every slug's directory, never thinned by the sample.
const COMMON_SET: &str = "common";
/// Up to this many bytes an output is also streamed a character at a time; past it the two
/// seeded plans alone keep a case's cost linear in its size.
const CHARACTER_PLAN_UP_TO: usize = 512;
/// How many differences of a class stdout shows per model; every one is in the report.
const SHOWN: usize = 5;

/// The legacy parsers that read a table's family: the names the gateway's `--tool-call-parser`
/// and `--reasoning-parser` take for it, each `None` where no parser of the tool or reasoning
/// crate reads the family's markers. A family without either is not replayed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Legacy {
    tool: Option<&'static str>,
    reasoning: Option<&'static str>,
}

impl Legacy {
    const NONE: Self = Self {
        tool: None,
        reasoning: None,
    };

    /// The legacy pair for `table`: the alias the table answers to in the gateway
    /// (`Table::ALIASES`) where it has one, else the parser whose format the family writes.
    fn of(table: Table) -> Self {
        let (tool, reasoning) = match table {
            // `<tool_call>` JSON and `<think>`, the model's own.
            Table::Qwen3 => (Some("qwen"), Some("qwen3")),
            // Qwen3-Coder's tags; the thought opened in the prompt.
            Table::Qwen3Tagged | Table::Qwen3TaggedDirect => (Some("qwen_xml"), Some("qwen3")),
            // No thought in the template.
            Table::Qwen2_5 => (Some("qwen"), None),
            Table::DeepSeekV4 => (Some("deepseek_v4"), Some("deepseek_v4")),
            Table::DeepSeekV4_1 => (Some("deepseek_v41"), Some("deepseek_v41")),
            Table::Hy4 => (Some("hy_v4"), Some("hy_v4")),
            // Ling writes GLM 4.5's keyed spelling (the name on its own line); GLM 4.7's parser
            // reads both spellings.
            Table::Ling | Table::Glm => (Some("glm47_moe"), Some("glm45")),
            Table::Glm4 => (Some("glm45_moe"), Some("glm45")),
            // A bare JSON list of calls, or one object: the generic JSON parser.
            Table::Xlam | Table::Json => (Some("json"), None),
            Table::MinimaxM2 => (Some("minimax_m2"), Some("minimax")),
            Table::MinimaxM3 => (Some("minimax_m3"), Some("minimax_m3")),
            Table::KimiK3 => (Some("kimi_k3"), Some("kimi_k3")),
            Table::Inkling => (Some("inkling"), Some("inkling")),
            Table::Gemma4 => (Some("gemma4"), Some("gemma4")),
            // `<think>` the model's own, or opened in the prompt; the calls (`<iquest_tool_call>`,
            // `<function name="…">`, `<dots_function_call>`, `<TOOLCALL>`, LFM2.5's Python list)
            // have no legacy parser: those cases are out of reach.
            Table::IQuest | Table::MiniCpm5 | Table::Dots3 | Table::Lfm2_5 => (None, Some("qwen3")),
            Table::LlamaNemotron => (None, Some("nano_v3")),
            // No legacy parser reads the family's markers: Seed-OSS's `<seed:…>` tags, Olmo 3's
            // `<function_calls>`, Mistral v13's `name[ARGS]{…}`, FunctionGemma's tokens,
            // K2-Horizon's `<ifm|…>` tags, Muse Spark's messages; the plain table has nothing to
            // parse.
            Table::SeedOss
            | Table::Olmo3
            | Table::MistralV13
            | Table::FunctionGemma
            | Table::K2Horizon
            | Table::MuseSpark
            | Table::Plain => (None, None),
        };
        Self { tool, reasoning }
    }

    fn is_none(self) -> bool {
        self == Self::NONE
    }
}

/// How a table spells the thought's markers, for the prompt's tail.
fn think_markers(table: Table) -> (&'static str, &'static str) {
    match table {
        Table::SeedOss => ("<seed:think>", "</seed:think>"),
        Table::Hy4 => ("<think:opensource>", "</think:opensource>"),
        Table::MinimaxM3 => ("<mm:think>", "</mm:think>"),
        Table::KimiK3 => ("<|open|>think<|sep|>", "<|close|>think<|sep|>"),
        Table::MuseSpark => (" to=self<|message|>", "<|eom|>"),
        Table::Inkling => ("<|content_thinking|>", "<|end_message|>"),
        Table::Gemma4 => ("<|channel>thought\n", "<channel|>"),
        Table::K2Horizon => ("<ifm|think>", "</ifm|think>"),
        Table::MistralV13 => ("[THINK]", "[/THINK]"),
        _ => ("<think>", "</think>"),
    }
}

/// How a model's template ends the generation prompt, as the corpus tests state it per slug.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Prompt {
    /// The turn's opener and nothing more; the template has no thought.
    Plain,
    /// Nothing, and the model writes its own thought; thinking off closes one in the prompt.
    ModelWritesTheThought,
    /// The thought opened in the prompt whatever the request says.
    AlwaysOpensTheThought,
    /// The thought opened in the prompt, closed there when the request turns thinking off.
    OpensTheThought,
    /// The thought opened only when the request turns thinking on, closed otherwise.
    OpensTheThoughtWhenAsked,
    /// Nothing, whatever the request says; the model writes the thought or not.
    ModelDecides,
    /// The thought's opener alone when the request turns thinking on, its closer alone otherwise.
    MarkerAlone,
}

/// Every recorded slug whose table has a legacy pair, by how its template ends the prompt.
const PROMPTS: &[(Prompt, &[&str])] = &[
    (
        Prompt::Plain,
        &[
            "qwen3-4b-instruct-2507",
            "qwen3-30b-a3b-instruct-2507",
            "qwen3-235b-a22b-instruct-2507",
            "qwen3-next-80b-a3b-instruct",
            "qwen3-vl-2b-instruct",
            "qwen3-vl-4b-instruct",
            "qwen3-vl-8b-instruct",
            "qwen3-vl-30b-a3b-instruct",
            "qwen3-vl-32b-instruct",
            "qwen3-vl-235b-a22b-instruct",
            "qwen2.5-7b-instruct-1m",
            "qwen2.5-14b-instruct-1m",
            "qwen2.5-vl-3b-instruct",
            "qwen2.5-vl-7b-instruct",
            "qwen2.5-vl-32b-instruct",
            "qwen2.5-omni-7b",
            "qwen3.5-0.8b",
            "qwen3-coder-30b-a3b-instruct",
            "qwen3-coder-next",
            "hermes-4-14b",
            "granite-4.1-3b",
            "ai21-jamba2-3b",
            "llama-xlam-2-8b-fc-r",
        ],
    ),
    (
        Prompt::ModelWritesTheThought,
        &[
            "qwen3-8b",
            "qwen3-4b-saferl",
            "qwen3-0.6b",
            "qwen3-235b-a22b",
            "qwen3-30b-a3b",
            "qwen3-omni-30b-a3b-instruct",
            "qwen3-omni-30b-a3b-thinking",
            "webworld-32b",
            "webworld-14b",
            "mimo-v2.5",
            "glm-4.6",
            "minicpm5-2b",
            "dots3-note-prev",
        ],
    ),
    (
        Prompt::AlwaysOpensTheThought,
        &[
            "qwen3-4b-thinking-2507",
            "qwen3-30b-a3b-thinking-2507",
            "qwen3-235b-a22b-thinking-2507",
            "qwen3-next-80b-a3b-thinking",
            "qwen3-vl-2b-thinking",
            "qwen3-vl-4b-thinking",
            "qwen3-vl-8b-thinking",
            "qwen3-vl-30b-a3b-thinking",
            "qwen3-vl-32b-thinking",
            "qwen3-vl-235b-a22b-thinking",
            "qwen3.8-2.4t-a95b",
            "kimi-k3",
            "step-3.5-flash",
            "hy4-preview",
            "glm-5.3-flash",
            "minimax-m2",
            "minimax-m2.7",
        ],
    ),
    (
        Prompt::OpensTheThought,
        &[
            "qwen3.5-4b",
            "qwen3.5-9b",
            "qwen3.5-27b",
            "qwen3.5-35b-a3b",
            "qwen3.5-122b-a10b",
            "qwen3.6-27b",
            "qwen3.6-35b-a3b",
            "qwen3.8-27b",
            "qwen3.8-flash-next",
            "qwen-agentworld-35b-a3b",
            "qwen-drive-1.0-4b",
            "deepseek-v4.1-flash",
            "k-exaone-236b-a23b",
            "nanbeige4.2-3b",
            "nvidia-nemotron-3-nano-30b-a3b-bf16",
            "ling-3.0-flash",
            "iquest-q1",
            "glm-4.7-flash",
        ],
    ),
    (Prompt::OpensTheThoughtWhenAsked, &["qwen3.5-2b"]),
    (
        Prompt::ModelDecides,
        &[
            "lfm2.5-1.2b-instruct",
            "minimax-m3",
            "inkling",
            "gemma-4-e4b-it",
        ],
    ),
    (Prompt::MarkerAlone, &["laguna-xs.2"]),
];

impl Prompt {
    /// The prompt stated for `slug`, if any.
    fn of(slug: &str) -> Option<Self> {
        PROMPTS
            .iter()
            .find(|(_, slugs)| slugs.contains(&slug))
            .map(|(prompt, _)| *prompt)
    }

    /// The bytes after the assistant header for the case's request, in the table's spelling of
    /// the thought's markers.
    fn tail(self, request: &Request, markers: (&str, &str)) -> String {
        let (open, close) = markers;
        let opened = format!("{open}\n");
        let closed = format!("{open}\n\n{close}\n\n");
        match (self, request.thinking_off()) {
            (Self::Plain | Self::ModelDecides, _) | (Self::ModelWritesTheThought, false) => {
                String::new()
            }
            (Self::OpensTheThought | Self::ModelWritesTheThought, true) => closed,
            (Self::OpensTheThought, false) | (Self::AlwaysOpensTheThought, _) => opened,
            (Self::OpensTheThoughtWhenAsked, _) if request.thinking_on() => opened,
            (Self::OpensTheThoughtWhenAsked, _) => closed,
            (Self::MarkerAlone, _) if request.thinking_on() => open.to_string(),
            (Self::MarkerAlone, _) => close.to_string(),
        }
    }
}

#[derive(Deserialize)]
struct Fixture {
    id: String,
    #[serde(default)]
    request: Request,
    reference: Reference,
    /// The text each output token contributes, as bellwether records it.
    output_pieces: Option<Vec<String>>,
    /// Chunk sizes in tokens by plan name; `whole` and `per_token` carry none and are derived.
    #[serde(default)]
    chunk_plans: BTreeMap<String, Option<Vec<usize>>>,
}

#[derive(Default, Deserialize)]
struct Request {
    /// The declared tools: the gateway runs its tool parser when the request declares a list,
    /// an empty one included, and not when it declares none.
    tools: Option<Vec<Tool>>,
    /// A `tool_choice` takes the gateway's legacy path elsewhere (`none` runs no tool parser, a
    /// forced or named choice reads with the JSON schema parser): a case that sets one is not
    /// routed here, and counted. The recorded corpus sets none.
    tool_choice: Option<Value>,
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
}

impl Request {
    fn thinking_off(&self) -> bool {
        self.enable_thinking() == Some(false)
    }

    fn thinking_on(&self) -> bool {
        self.enable_thinking() == Some(true)
    }

    fn enable_thinking(&self) -> Option<bool> {
        self.chat_template_kwargs
            .as_ref()
            .and_then(|kwargs| kwargs.enable_thinking)
    }
}

#[derive(Deserialize)]
struct ChatTemplateKwargs {
    enable_thinking: Option<bool>,
}

#[derive(Deserialize)]
struct Reference {
    text: String,
    message: ReferenceMessage,
    finish_reason: String,
}

#[derive(Deserialize)]
struct ReferenceMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ReferenceCall>,
}

#[derive(Deserialize)]
struct ReferenceCall {
    function: ReferenceFunction,
}

#[derive(Deserialize)]
struct ReferenceFunction {
    name: String,
    arguments: String,
}

/// What a parser said an output means, in the terms the chat message uses: content and reasoning
/// absent when empty (content when whitespace-only, as the gateway and the chat adapter have it),
/// the calls in order as name and arguments text, and the finish reason.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Said {
    content: Option<String>,
    reasoning: Option<String>,
    calls: Vec<(String, String)>,
    finish: String,
}

impl Said {
    fn new(
        content: String,
        reasoning: String,
        calls: Vec<(String, String)>,
        finish: String,
    ) -> Self {
        Self {
            content: (!content.trim().is_empty()).then_some(content),
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            calls,
            finish,
        }
    }

    /// What the table said: the message the chat adapter folds from its events.
    fn of_events(events: &[Event]) -> Self {
        let choice = adapt::chat::message(0, events);
        Self {
            content: choice.message.content,
            reasoning: choice.message.reasoning_content,
            calls: choice
                .message
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(|call| {
                    (
                        call.function.name,
                        call.function.arguments.unwrap_or_default(),
                    )
                })
                .collect(),
            finish: choice.finish_reason.unwrap_or_default(),
        }
    }

    fn of_reference(reference: &Reference) -> Self {
        Self::new(
            reference.message.content.clone().unwrap_or_default(),
            reference
                .message
                .reasoning_content
                .clone()
                .unwrap_or_default(),
            reference
                .message
                .tool_calls
                .iter()
                .map(|call| (call.function.name.clone(), call.function.arguments.clone()))
                .collect(),
            reference.finish_reason.clone(),
        )
    }

    /// The same channels and finish, the calls' arguments equal as JSON values (or as text where
    /// a side's do not parse).
    fn equal(&self, other: &Self) -> bool {
        self.content == other.content
            && self.reasoning == other.reasoning
            && self.finish == other.finish
            && calls_equal(&self.calls, &other.calls)
    }

    /// [`Self::equal`] once whitespace at the edges of the content and the reasoning is trimmed.
    fn equivalent(&self, other: &Self) -> bool {
        self.trimmed().equal(&other.trimmed())
    }

    fn trimmed(&self) -> Self {
        let trim = |part: &Option<String>| {
            part.as_deref()
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(str::to_string)
        };
        Self {
            content: trim(&self.content),
            reasoning: trim(&self.reasoning),
            calls: self.calls.clone(),
            finish: self.finish.clone(),
        }
    }
}

/// How many bytes of a channel or an argument the report shows of a difference.
const BRIEF: usize = 240;

/// `text` cut to [`BRIEF`] bytes on a character boundary, with how much was cut.
fn brief(text: &str) -> String {
    if text.len() <= BRIEF {
        return format!("{text:?}");
    }
    let cut = text
        .char_indices()
        .map(|(at, _)| at)
        .take_while(|&at| at <= BRIEF)
        .last()
        .unwrap_or(0);
    format!("{:?}… (+{} bytes)", &text[..cut], text.len() - cut)
}

impl fmt::Display for Said {
    /// The report's rendering: each channel and argument cut to [`BRIEF`] bytes, so a report of a
    /// long output stays readable and a run of a benchmark-scale set stays in memory.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let channel =
            |part: &Option<String>| part.as_deref().map_or_else(|| "None".to_string(), brief);
        write!(
            f,
            "content: {}, reasoning: {}, calls: [",
            channel(&self.content),
            channel(&self.reasoning)
        )?;
        for (at, (name, arguments)) in self.calls.iter().enumerate() {
            let separator = if at == 0 { "" } else { ", " };
            write!(f, "{separator}({name:?}, {})", brief(arguments))?;
        }
        write!(f, "], finish: {:?}", self.finish)
    }
}

/// Where two sides first differ, field by field: the field, the byte, and a window of each side
/// around it, so a report whose brief rendering shows the same prefix for both still says what
/// tells them apart.
fn located(symphony: &Said, legacy: &Said) -> String {
    let mut fields: Vec<(String, &str, &str)> = vec![
        (
            "content".to_string(),
            symphony.content.as_deref().unwrap_or(""),
            legacy.content.as_deref().unwrap_or(""),
        ),
        (
            "reasoning".to_string(),
            symphony.reasoning.as_deref().unwrap_or(""),
            legacy.reasoning.as_deref().unwrap_or(""),
        ),
    ];
    for (at, (a, b)) in symphony.calls.iter().zip(&legacy.calls).enumerate() {
        fields.push((format!("calls[{at}].name"), &a.0, &b.0));
        fields.push((format!("calls[{at}].arguments"), &a.1, &b.1));
    }
    fields.push(("finish".to_string(), &symphony.finish, &legacy.finish));
    let mut out = String::new();
    for (field, a, b) in fields {
        if a == b {
            continue;
        }
        let at = a
            .bytes()
            .zip(b.bytes())
            .position(|(x, y)| x != y)
            .unwrap_or(a.len().min(b.len()));
        out.push_str(&format!(
            "      {field} differs at byte {at}: symphony {} legacy {}\n",
            window(a, at),
            window(b, at)
        ));
    }
    if symphony.calls.len() != legacy.calls.len() {
        out.push_str(&format!(
            "      {} call(s) on the symphony side, {} on the legacy side\n",
            symphony.calls.len(),
            legacy.calls.len()
        ));
    }
    out
}

/// Up to 60 characters of `text` before `at` and 60 from it, on character boundaries, with the
/// bar at `at`.
fn window(text: &str, at: usize) -> String {
    let mut at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    let from = text[..at]
        .char_indices()
        .rev()
        .nth(59)
        .map_or(0, |(i, _)| i);
    let to = text[at..]
        .char_indices()
        .nth(60)
        .map_or(text.len(), |(i, _)| at + i);
    format!(
        "{}{:?}|{:?}{}",
        if from > 0 { "…" } else { "" },
        &text[from..at],
        &text[at..to],
        if to < text.len() { "…" } else { "" }
    )
}

/// Whether two call lists name the same tools with the same arguments, the arguments compared as
/// JSON values where both parse and as text otherwise.
fn calls_equal(a: &[(String, String)], b: &[(String, String)]) -> bool {
    let value = |text: &str| serde_json::from_str::<Value>(text).ok();
    a.len() == b.len()
        && a.iter().zip(b).all(|((name, x), (other, y))| {
            name == other
                && (x == y || matches!((value(x), value(y)), (Some(p), Some(q)) if p == q))
        })
}

/// What one replay's comparison gave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// Neither side could be compared: a side panicked.
    Panicked,
    /// The legacy pair cannot read the case's calls or thought.
    OutOfReach,
    /// A probe the two paths are known to read otherwise than the reference ([`PROBES`]).
    Listed,
    Bitwise,
    Equal,
    Equivalent,
    SymphonyRight,
    LegacyRight,
    BothWrong,
}

impl Verdict {
    fn name(self) -> &'static str {
        match self {
            Self::Panicked => "PANICKED",
            Self::OutOfReach => "out of reach",
            Self::Listed => "listed",
            Self::Bitwise => "bitwise",
            Self::Equal => "equal",
            Self::Equivalent => "equivalent",
            Self::SymphonyRight => "symphony right",
            Self::LegacyRight => "LEGACY RIGHT",
            Self::BothWrong => "both wrong",
        }
    }

    /// Whether the verdict is a difference the report shows in full.
    fn differs(self) -> bool {
        matches!(
            self,
            Self::SymphonyRight
                | Self::LegacyRight
                | Self::BothWrong
                | Self::Listed
                | Self::Panicked
        )
    }
}

/// What the legacy pair of a table can read.
#[derive(Clone, Copy)]
struct Reach {
    calls: bool,
    reasoning: bool,
}

impl Reach {
    fn of(legacy: Legacy) -> Self {
        Self {
            calls: legacy.tool.is_some(),
            reasoning: legacy.reasoning.is_some(),
        }
    }

    /// Whether the pair can read everything the reference says the case holds. The reference
    /// alone decides: a call or a thought the table finds where the reference has none is the
    /// table's to answer for, so such a case is compared.
    fn covers(self, reference: &Said) -> bool {
        (self.calls || reference.calls.is_empty())
            && (self.reasoning || reference.reasoning.is_none())
    }
}

/// The probe cases of bellwether's common set (bellwether #16) that put a marker inside text: a
/// `</think>` inside the reasoning and a complete `<tool_call>` block inside a code fence. Every
/// marker parser reads the marker, the reference keeps the text, and the two paths differ from
/// each other by how far each one's streaming parser gets before it does: a difference on them
/// says nothing about either path, so it is listed, counted and shown, never a failure.
const PROBES: &[&str] = &[
    "parse/reasoning-with-marker-text",
    "parse/content-with-marker-in-code-fence",
];

/// Whether `id` names one of the [`PROBES`], whatever its slug.
fn is_probe(id: &str) -> bool {
    let after_slug = id.split_once('/').map_or(id, |(_, rest)| rest);
    PROBES.contains(&after_slug)
}

/// The verdict for one replay: how far the two sides agree, and which is right when they differ.
fn verdict(id: &str, symphony: &Said, legacy: &Said, reference: &Said, reach: Reach) -> Verdict {
    if !reach.covers(reference) {
        Verdict::OutOfReach
    } else if symphony == legacy {
        Verdict::Bitwise
    } else if symphony.equal(legacy) {
        Verdict::Equal
    } else if symphony.equivalent(legacy) {
        Verdict::Equivalent
    } else if is_probe(id) {
        Verdict::Listed
    } else {
        match (symphony.equivalent(reference), legacy.equivalent(reference)) {
            (true, _) => Verdict::SymphonyRight,
            (false, true) => Verdict::LegacyRight,
            (false, false) => Verdict::BothWrong,
        }
    }
}

/// The legacy parser crates' factories and a runtime for their futures, one set per model.
struct Factories {
    tools: ToolFactory,
    reasoning: ReasoningFactory,
    runtime: Runtime,
}

impl Factories {
    fn new() -> Result<Self, String> {
        Ok(Self {
            tools: ToolFactory::new(),
            reasoning: ReasoningFactory::new(),
            runtime: Builder::new_current_thread()
                .build()
                .map_err(|e| format!("cannot build a runtime: {e}"))?,
        })
    }

    #[expect(
        clippy::panic,
        reason = "a legacy name the differential declares that no factory registers is a mistake \
                  in its table"
    )]
    fn tool_parser(&self, legacy: Legacy) -> Option<Box<dyn ToolParser>> {
        legacy
            .tool
            .map(|name| match self.tools.registry().create_parser(name) {
                Some(parser) => parser,
                None => panic!("no legacy tool parser is registered as {name}"),
            })
    }

    #[expect(
        clippy::panic,
        reason = "a legacy name the differential declares that no factory registers is a mistake \
                  in its table"
    )]
    fn reasoning_parser(&self, legacy: Legacy) -> Option<Box<dyn ReasoningParser>> {
        legacy
            .reasoning
            .map(|name| match self.reasoning.registry().create_parser(name) {
                Some(parser) => parser,
                None => panic!("no legacy reasoning parser is registered as {name}"),
            })
    }
}

/// The finish reason the gateway gives: `stop` after a call is `tool_calls`; a truncation stays.
fn legacy_finish(finish: &str, calls: bool) -> String {
    if calls && finish == "stop" {
        "tool_calls".to_string()
    } else {
        finish.to_string()
    }
}

/// The legacy path on a whole output, as the non-streaming gateway runs it: the reasoning parser
/// first, armed when the prompt leaves the model inside its thought (`opened`), then the tool
/// parser on the text outside the thought when the request declares `tools` (a list, empty or
/// not; `None` declares none).
fn legacy_whole(
    factories: &Factories,
    legacy: Legacy,
    text: &str,
    tools: Option<&[Tool]>,
    opened: bool,
    finish: &str,
) -> Said {
    let mut rest = text.to_string();
    let mut reasoning = String::new();
    if let Some(mut parser) = factories.reasoning_parser(legacy) {
        if opened {
            parser.mark_reasoning_started();
        }
        if let Ok(result) = parser.detect_and_parse_reasoning(&rest) {
            reasoning = result.reasoning_text;
            rest = result.normal_text;
        }
    }
    let mut calls = Vec::new();
    if let (Some(parser), Some(tools)) = (factories.tool_parser(legacy), tools) {
        if let Ok((normal, parsed)) = factories
            .runtime
            .block_on(parser.parse_complete_with_tools(&rest, tools))
        {
            calls = parsed
                .into_iter()
                .map(|call| (call.function.name, call.function.arguments))
                .collect();
            rest = normal;
        }
    }
    let finish = legacy_finish(finish, !calls.is_empty());
    Said::new(rest, reasoning, calls, finish)
}

/// The legacy path on an output cut at `cuts`, as the streaming gateway runs it: every delta
/// through the reasoning parser (armed as above, the prompt's `<think>` stripped), its normal text
/// through the tool parser when the request declares `tools` and the reasoning parser is not
/// inside the thought, else to the content; at the end the reasoning parser's flush, then the
/// tool parser's unstreamed text and arguments. A client folds the deltas as the chat message
/// does.
fn legacy_streamed(
    factories: &Factories,
    legacy: Legacy,
    text: &str,
    cuts: &[usize],
    tools: Option<&[Tool]>,
    opened: bool,
    finish: &str,
) -> Said {
    let mut reasoning_parser = factories.reasoning_parser(legacy);
    if let (Some(parser), true) = (reasoning_parser.as_mut(), opened) {
        parser.mark_reasoning_started();
        parser.mark_think_start_stripped();
    }
    let mut tool_parser = tools.and_then(|_| factories.tool_parser(legacy));
    let tools = tools.unwrap_or_default();
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut calls: BTreeMap<usize, (Option<String>, String)> = BTreeMap::new();
    let mut any_call = false;
    let end = text.len();
    let mut from = 0;
    let pieces = cuts
        .iter()
        .chain(std::iter::once(&end))
        .filter_map(|&cut| {
            let piece = (cut > from).then(|| &text[from..cut]);
            from = cut.max(from);
            piece
        })
        .map(Some)
        .chain(std::iter::once(None));
    for piece in pieces {
        let (normal, in_reasoning) = match (reasoning_parser.as_mut(), piece) {
            (Some(parser), Some(piece)) => {
                match parser.parse_reasoning_streaming_incremental(piece) {
                    Ok(result) => {
                        reasoning.push_str(&result.reasoning_text);
                        (result.normal_text, parser.is_in_reasoning())
                    }
                    Err(_) => (piece.to_string(), false),
                }
            }
            (Some(parser), None) => match parser.flush() {
                Ok(result) => {
                    reasoning.push_str(&result.reasoning_text);
                    (result.normal_text, parser.is_in_reasoning())
                }
                Err(_) => (String::new(), false),
            },
            (None, Some(piece)) => (piece.to_string(), false),
            (None, None) => (String::new(), false),
        };
        if piece.is_none() && normal.is_empty() {
            continue;
        }
        match tool_parser.as_mut() {
            Some(parser) if !in_reasoning => {
                if let Ok(result) = factories
                    .runtime
                    .block_on(parser.parse_incremental(&normal, tools))
                {
                    content.push_str(&result.normal_text);
                    for item in result.calls {
                        any_call = true;
                        let call = calls.entry(item.tool_index).or_default();
                        if let Some(name) = item.name {
                            call.0.get_or_insert(name);
                        }
                        call.1.push_str(&item.parameters);
                    }
                }
            }
            _ => content.push_str(&normal),
        }
    }
    if let Some(parser) = tool_parser.as_mut() {
        content.push_str(&parser.take_unstreamed_normal_text());
        for item in parser.get_unstreamed_tool_args().unwrap_or_default() {
            calls
                .entry(item.tool_index)
                .or_default()
                .1
                .push_str(&item.parameters);
        }
    }
    let calls = calls
        .into_values()
        .map(|(name, arguments)| (name.unwrap_or_default(), arguments))
        .collect();
    Said::new(content, reasoning, calls, legacy_finish(finish, any_call))
}

/// The table on the output cut at `cuts`, after the prompt's `tail`.
#[expect(
    clippy::panic,
    reason = "a replay the engine refuses ends the test with the case"
)]
fn symphony_said(
    format: &Format,
    tools: &[Tool],
    tail: &str,
    text: &str,
    cuts: &[usize],
    finish: &EngineFinish,
) -> Said {
    let mut engine = Engine::new(format.clone(), Declared::of(tools));
    let events = replay_after(&mut engine, tail, text, cuts, finish, false)
        .unwrap_or_else(|e| panic!("the table refused a replay: {e}"));
    Said::of_events(&events)
}

fn engine_finish(reference: &str) -> EngineFinish {
    match reference {
        "length" => EngineFinish::Length,
        _ => EngineFinish::Stop,
    }
}

/// The character plans a case is streamed on: the first two seeded plans the corpus tests use
/// and, up to [`CHARACTER_PLAN_UP_TO`] bytes, a character at a time. Cuts that fall inside a
/// marker, which an engine's token-aligned deltas never make.
fn character_plans(text: &str) -> Vec<(String, Vec<usize>)> {
    chunkings(text, 0)
        .into_iter()
        .filter(|(name, _)| {
            name == "seeded plan 1"
                || name == "seeded plan 2"
                || (name == "character by character" && text.len() <= CHARACTER_PLAN_UP_TO)
        })
        .collect()
}

/// The token plans a case is streamed on, from the pieces bellwether recorded with it: one token
/// per delta, the shape an engine's stream has, and the first two of its random plans; each as
/// byte cuts at the chunks' token boundaries. None for a case without pieces, or whose pieces do
/// not give back the text.
fn token_plans(fixture: &Fixture) -> Vec<(String, Vec<usize>)> {
    let Some(pieces) = fixture.output_pieces.as_deref() else {
        return Vec::new();
    };
    if pieces.concat() != fixture.reference.text {
        return Vec::new();
    }
    let mut ends = Vec::with_capacity(pieces.len());
    let mut at = 0;
    for piece in pieces {
        at += piece.len();
        ends.push(at);
    }
    fixture
        .chunk_plans
        .iter()
        .filter_map(|(name, sizes)| {
            let sizes = match (name.as_str(), sizes) {
                ("per_token", None) => vec![1; pieces.len()],
                ("random-1" | "random-2", Some(sizes)) => sizes.clone(),
                _ => return None,
            };
            let mut cuts: Vec<usize> = Vec::new();
            let mut token = 0;
            for size in sizes {
                token += size;
                if token >= pieces.len() {
                    break;
                }
                if cuts.last().is_none_or(|&last| ends[token - 1] > last) {
                    cuts.push(ends[token - 1]);
                }
            }
            Some((format!("token plan {name}"), cuts))
        })
        .collect()
}

/// The verdicts of one kind of replay, counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Tally {
    replays: usize,
    panicked: usize,
    out_of_reach: usize,
    listed: usize,
    bitwise: usize,
    equal: usize,
    equivalent: usize,
    symphony_right: usize,
    legacy_right: usize,
    both_wrong: usize,
}

impl Tally {
    fn count(&mut self, verdict: Verdict) {
        self.replays += 1;
        match verdict {
            Verdict::Panicked => self.panicked += 1,
            Verdict::OutOfReach => self.out_of_reach += 1,
            Verdict::Listed => self.listed += 1,
            Verdict::Bitwise => self.bitwise += 1,
            Verdict::Equal => self.equal += 1,
            Verdict::Equivalent => self.equivalent += 1,
            Verdict::SymphonyRight => self.symphony_right += 1,
            Verdict::LegacyRight => self.legacy_right += 1,
            Verdict::BothWrong => self.both_wrong += 1,
        }
    }

    fn add(&mut self, other: Self) {
        self.replays += other.replays;
        self.panicked += other.panicked;
        self.out_of_reach += other.out_of_reach;
        self.listed += other.listed;
        self.bitwise += other.bitwise;
        self.equal += other.equal;
        self.equivalent += other.equivalent;
        self.symphony_right += other.symphony_right;
        self.legacy_right += other.legacy_right;
        self.both_wrong += other.both_wrong;
    }
}

impl fmt::Display for Tally {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} replays: {} bitwise, {} equal, {} equivalent, {} symphony right, {} legacy right, \
             {} both wrong, {} listed, {} out of reach, {} panicked",
            self.replays,
            self.bitwise,
            self.equal,
            self.equivalent,
            self.symphony_right,
            self.legacy_right,
            self.both_wrong,
            self.listed,
            self.out_of_reach,
            self.panicked
        )
    }
}

/// The kinds of streamed replay: at the recorded tokens' boundaries, and at characters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cuts {
    Token,
    Character,
}

/// Streams that do not say what the same side's whole parse says, as equivalents, by the kind of
/// cuts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Drift {
    token: usize,
    character: usize,
}

impl Drift {
    fn count(&mut self, cuts: Cuts) {
        match cuts {
            Cuts::Token => self.token += 1,
            Cuts::Character => self.character += 1,
        }
    }
}

/// What one model's replay gave.
#[derive(Clone, Debug, Default)]
struct Counts {
    cases: usize,
    /// Cases whose request sets a `tool_choice`, which the differential does not route.
    unrouted: usize,
    /// Cases without recorded pieces, which have no token plans.
    without_pieces: usize,
    whole: Tally,
    token: Tally,
    character: Tally,
    legacy_drift: Drift,
    /// The corpus tests prove the table's streams say what its whole parse says on every plan;
    /// counted here too.
    symphony_drift: Drift,
    legacy_panics: usize,
    symphony_panics: usize,
}

impl Counts {
    fn add(&mut self, other: &Self) {
        self.cases += other.cases;
        self.unrouted += other.unrouted;
        self.without_pieces += other.without_pieces;
        self.whole.add(other.whole);
        self.token.add(other.token);
        self.character.add(other.character);
        self.legacy_drift.token += other.legacy_drift.token;
        self.legacy_drift.character += other.legacy_drift.character;
        self.symphony_drift.token += other.symphony_drift.token;
        self.symphony_drift.character += other.symphony_drift.character;
        self.legacy_panics += other.legacy_panics;
        self.symphony_panics += other.symphony_panics;
    }

    fn streamed(&mut self, cuts: Cuts) -> &mut Tally {
        match cuts {
            Cuts::Token => &mut self.token,
            Cuts::Character => &mut self.character,
        }
    }
}

impl fmt::Display for Counts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "  {} cases ({} without recorded pieces, {} with a tool_choice, not routed)",
            self.cases, self.without_pieces, self.unrouted
        )?;
        writeln!(f, "  whole:           {}", self.whole)?;
        writeln!(f, "  token plans:     {}", self.token)?;
        writeln!(f, "  character plans: {}", self.character)?;
        writeln!(
            f,
            "  streams that differ from their side's whole parse: legacy {} on token plans, {} on \
             character plans; symphony {} and {}; panics: {} legacy, {} symphony",
            self.legacy_drift.token,
            self.legacy_drift.character,
            self.symphony_drift.token,
            self.symphony_drift.character,
            self.legacy_panics,
            self.symphony_panics
        )
    }
}

/// A side's parse, caught: what it said, or the message of its panic.
fn caught<T>(side: &str, parse: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(parse)).map_err(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "a panic without a message".to_string());
        format!("{side} panicked: {message}")
    })
}

/// One model replayed: its slug, table and legacy pair, the counts, the report (one line per
/// case, every difference in full) and the failures (a legacy-right case, a table's panic).
struct Replayed {
    slug: String,
    table: Table,
    legacy: Legacy,
    counts: Counts,
    report: String,
    shown: String,
    failures: Vec<String>,
}

/// Why a recorded slug is left out.
enum Skipped {
    NoTable(String),
    NoLegacyParser(String, Table),
    NoPromptRow(String, Table),
    NotAsked,
}

/// A recorded slug: its directory under the corpus root and the model id its manifest names.
struct Recorded {
    slug: String,
    model: String,
    parse: PathBuf,
}

/// Every slug under `root` with a parse set, in order, with the model id its manifest names.
fn recorded(root: &Path) -> Result<Vec<Recorded>, String> {
    let mut slugs: Vec<PathBuf> = fs::read_dir(root)
        .map_err(|e| format!("cannot read {}: {e}", root.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join("parse").is_dir())
        .collect();
    slugs.sort();
    slugs
        .into_iter()
        .map(|dir| {
            let manifest = dir.join("manifest.toml");
            let text = fs::read_to_string(&manifest)
                .map_err(|e| format!("cannot read {}: {e}", manifest.display()))?;
            let model =
                model_in(&text).ok_or_else(|| format!("{}: no model line", manifest.display()))?;
            Ok(Recorded {
                slug: dir
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                model,
                parse: dir.join("parse"),
            })
        })
        .collect()
}

/// The quoted value of the manifest's `model` key.
fn model_in(manifest: &str) -> Option<String> {
    manifest.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        (key.trim() == "model").then(|| value.trim().trim_matches('"').to_string())
    })
}

/// Reads every `.jsonl` set under `dir`, in the files' order, every `every`-th case of each
/// recorded set and the common set whole, and hands each case to `each` as it is read, so that a
/// run holds one case of a set at a time however many models are in flight; how many it handed
/// over.
fn each_fixture(dir: &Path, every: usize, each: &mut dyn FnMut(Fixture)) -> Result<usize, String> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    files.sort();
    let mut cases = 0;
    for file in files {
        let step = if file.file_stem().is_some_and(|stem| stem == COMMON_SET) {
            1
        } else {
            every.max(1)
        };
        let reader = BufReader::new(
            fs::File::open(&file).map_err(|e| format!("cannot read {}: {e}", file.display()))?,
        );
        for (index, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| format!("cannot read {}: {e}", file.display()))?;
            if line.trim().is_empty() || !index.is_multiple_of(step) {
                continue;
            }
            each(
                serde_json::from_str(&line)
                    .map_err(|e| format!("{}:{}: {e}", file.display(), index + 1))?,
            );
            cases += 1;
        }
    }
    Ok(cases)
}

/// The sample's step from [`SAMPLE_ENV`]: 1 when it is unset or empty.
fn sample_every() -> usize {
    match std::env::var(SAMPLE_ENV)
        .ok()
        .filter(|value| !value.is_empty())
    {
        None => 1,
        Some(value) => {
            let every: usize = value.parse().unwrap_or(0);
            assert!(
                every > 0,
                "{SAMPLE_ENV}={value:?} is not a positive integer"
            );
            every
        }
    }
}

/// The slugs [`SLUGS_ENV`] restricts the run to, if it is set and not empty.
fn asked_slugs() -> Option<Vec<String>> {
    std::env::var(SLUGS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value
                .split(',')
                .map(|slug| slug.trim().to_string())
                .collect()
        })
}

/// Replay one recorded slug, or say why it is left out.
#[expect(
    clippy::panic,
    reason = "a set that cannot be read ends the test with its path"
)]
fn replay_model(
    recorded: &Recorded,
    every: usize,
    asked: Option<&[String]>,
) -> Result<Replayed, Skipped> {
    let slug = recorded.slug.clone();
    if asked.is_some_and(|asked| !asked.contains(&slug)) {
        return Err(Skipped::NotAsked);
    }
    let (Some(format), Some(table)) = (for_model(&recorded.model), table_for(&recorded.model))
    else {
        return Err(Skipped::NoTable(slug));
    };
    let legacy = Legacy::of(table);
    if legacy.is_none() {
        return Err(Skipped::NoLegacyParser(slug, table));
    }
    let Some(prompt) = Prompt::of(&slug) else {
        return Err(Skipped::NoPromptRow(slug, table));
    };
    let factories = Factories::new().unwrap_or_else(|e| panic!("{e}"));
    let reach = Reach::of(legacy);
    let markers = think_markers(table);
    let mut counts = Counts::default();
    let mut report = String::new();
    let mut shown = Shown::default();
    let mut failures = Vec::new();
    let mut equivalents_shown = 0;
    let cases = each_fixture(&recorded.parse, every, &mut |fixture: Fixture| {
        let fixture = &fixture;
        if fixture.request.tool_choice.is_some() {
            counts.unrouted += 1;
            report.push_str(&format!("  {:16} {}\n", "not routed", fixture.id));
            return;
        }
        let text = fixture.reference.text.as_str();
        let declared = fixture.request.tools.as_deref();
        let tools = declared.unwrap_or_default();
        let tail = prompt.tail(&fixture.request, markers);
        let opened = format.reasoning_open(&tail);
        let finish = engine_finish(&fixture.reference.finish_reason);
        // The engine's reason, which both paths turn into `tool_calls` after a call: the
        // reference's `tool_calls` is the message's, not the engine's.
        let engine_reason = if fixture.reference.finish_reason == "length" {
            "length"
        } else {
            "stop"
        };
        let mut case = Case {
            id: &fixture.id,
            reference: Said::of_reference(&fixture.reference),
            reach,
            counts: &mut counts,
            failures: &mut failures,
            equivalents_shown: &mut equivalents_shown,
            plans: Vec::new(),
            details: String::new(),
        };
        let (whole_verdict, symphony_whole, legacy_whole_said) = case.compare(
            "whole",
            caught("the table", || {
                symphony_said(&format, tools, &tail, text, &[], &finish)
            }),
            caught("the legacy parsers", || {
                legacy_whole(&factories, legacy, text, declared, opened, engine_reason)
            }),
        );
        case.counts.whole.count(whole_verdict);
        let token_plans = token_plans(fixture);
        if token_plans.is_empty() {
            case.counts.without_pieces += 1;
        }
        let streamed = token_plans
            .into_iter()
            .map(|plan| (Cuts::Token, plan))
            .chain(
                character_plans(text)
                    .into_iter()
                    .map(|plan| (Cuts::Character, plan)),
            );
        for (kind, (plan, cuts)) in streamed {
            let (streamed_verdict, symphony_streamed, legacy_streamed_said) = case.compare(
                &plan,
                caught("the table", || {
                    symphony_said(&format, tools, &tail, text, &cuts, &finish)
                }),
                caught("the legacy parsers", || {
                    legacy_streamed(
                        &factories,
                        legacy,
                        text,
                        &cuts,
                        declared,
                        opened,
                        engine_reason,
                    )
                }),
            );
            case.counts.streamed(kind).count(streamed_verdict);
            if let (Some(whole), Some(streamed)) = (&legacy_whole_said, &legacy_streamed_said) {
                if !whole.equivalent(streamed) {
                    case.counts.legacy_drift.count(kind);
                    case.details.push_str(&format!(
                        "    {plan}: the legacy stream differs from the legacy whole parse\n      \
                         whole:    {whole}\n      streamed: {streamed}\n"
                    ));
                }
            }
            if let (Some(whole), Some(streamed)) = (&symphony_whole, &symphony_streamed) {
                if !whole.equivalent(streamed) {
                    case.counts.symphony_drift.count(kind);
                    case.details.push_str(&format!(
                        "    {plan}: the table's stream differs from its whole parse\n      \
                         whole:    {whole}\n      streamed: {streamed}\n"
                    ));
                }
            }
        }
        let line = format!("  {:16} {}\n", whole_verdict.name(), fixture.id);
        report.push_str(&line);
        if !case.details.is_empty() {
            report.push_str(&format!("    plans: {}\n", case.plans.join("; ")));
            report.push_str(&case.details);
            shown.differs(whole_verdict, &line, &case.details);
        }
    })
    .unwrap_or_else(|e| panic!("{e}"));
    counts.cases = cases;
    let summary = format!("{slug} ({}, legacy {legacy:?}):\n{counts}", table.name());
    report.push_str(&summary);
    Ok(Replayed {
        slug,
        table,
        legacy,
        counts,
        report,
        shown: shown.text() + &summary,
        failures,
    })
}

/// One case under comparison: the reference, what the legacy pair reaches, the model's counts
/// and failures, and the case's own record (each plan's verdict and every difference in full).
struct Case<'a> {
    id: &'a str,
    reference: Said,
    reach: Reach,
    counts: &'a mut Counts,
    failures: &'a mut Vec<String>,
    /// How many equivalent-only replays of the model the report has shown in full: the first
    /// [`SHOWN`] say what the two paths keep differently at the channels' edges.
    equivalents_shown: &'a mut usize,
    plans: Vec<String>,
    details: String,
}

impl Case<'_> {
    /// Compare the two sides of one replay: the verdict, counted and recorded, and the two
    /// `Said`s where the sides did not panic.
    fn compare(
        &mut self,
        plan: &str,
        symphony: Result<Said, String>,
        legacy: Result<Said, String>,
    ) -> (Verdict, Option<Said>, Option<Said>) {
        let (verdict, symphony, legacy) = match (symphony, legacy) {
            (Ok(symphony), Ok(legacy)) => (
                verdict(self.id, &symphony, &legacy, &self.reference, self.reach),
                Some(symphony),
                Some(legacy),
            ),
            (symphony, legacy) => {
                for side in [&symphony, &legacy] {
                    if let Err(message) = side {
                        self.details.push_str(&format!("    {plan}: {message}\n"));
                    }
                }
                if symphony.is_err() {
                    self.counts.symphony_panics += 1;
                    self.failures
                        .push(format!("{}: {plan}: the table panicked", self.id));
                }
                if legacy.is_err() {
                    self.counts.legacy_panics += 1;
                }
                (Verdict::Panicked, symphony.ok(), legacy.ok())
            }
        };
        let show_equivalent = verdict == Verdict::Equivalent && *self.equivalents_shown < SHOWN;
        if show_equivalent {
            *self.equivalents_shown += 1;
        }
        if verdict.differs() || show_equivalent {
            if let (Some(symphony), Some(legacy)) = (&symphony, &legacy) {
                self.details.push_str(&format!(
                    "    {plan}: {}\n      symphony:  {symphony}\n      legacy:    {legacy}\n      \
                     reference: {}\n{}",
                    verdict.name(),
                    self.reference,
                    located(symphony, legacy)
                ));
            }
            if verdict == Verdict::LegacyRight {
                self.failures.push(format!(
                    "{}: {plan}: the legacy parsers say what the reference says and the table \
                     does not",
                    self.id
                ));
            }
        }
        self.plans.push(format!("{plan} {}", verdict.name()));
        (verdict, symphony, legacy)
    }
}

/// The first differences of each class a model has, as stdout shows them.
#[derive(Default)]
struct Shown {
    lines: String,
    by_class: BTreeMap<&'static str, usize>,
}

impl Shown {
    fn differs(&mut self, verdict: Verdict, line: &str, details: &str) {
        let count = self.by_class.entry(verdict.name()).or_default();
        *count += 1;
        if *count <= SHOWN {
            self.lines.push_str(line);
            self.lines.push_str(details);
        }
    }

    fn text(self) -> String {
        let mut text = self.lines;
        for (class, count) in self.by_class {
            if count > SHOWN {
                text.push_str(&format!(
                    "  ... and {} more {class} cases, every one in the report\n",
                    count - SHOWN
                ));
            }
        }
        text
    }
}

/// Run `each` on every item, on as many threads as the run has CPUs, and hand what it gives to
/// `then` in the items' order. A panic in `each` ends the run once the items in flight are done.
fn in_order<I: Sync, T: Send>(items: &[I], each: impl Fn(&I) -> T + Sync, mut then: impl FnMut(T)) {
    let threads = std::thread::available_parallelism()
        .map_or(1, NonZeroUsize::get)
        .min(items.len())
        .max(1);
    let stop = AtomicBool::new(false);
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let tx = tx.clone();
            let (next, each, stop) = (&next, &each, &stop);
            scope.spawn(move || {
                let _raised_on_panic = StopOnPanic(stop);
                while !stop.load(Ordering::Relaxed) {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(index) else {
                        break;
                    };
                    if tx.send((index, each(item))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut pending = BTreeMap::new();
        let mut done = 0;
        for (index, outcome) in rx {
            pending.insert(index, outcome);
            while let Some(outcome) = pending.remove(&done) {
                then(outcome);
                done += 1;
            }
        }
    });
}

/// Raises the flag it holds when its thread unwinds from a panic.
struct StopOnPanic<'a>(&'a AtomicBool);

impl Drop for StopOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::Relaxed);
        }
    }
}

/// The report file under [`REPORT_DIR_ENV`] or the build's `target/tmp`; a write that fails is
/// kept and reported when the file is finished, so that the run's closures need not handle it.
struct Report {
    path: PathBuf,
    file: BufWriter<fs::File>,
    error: Option<std::io::Error>,
}

impl Report {
    fn create() -> Result<Self, String> {
        let dir = std::env::var_os(REPORT_DIR_ENV)
            .map_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")), PathBuf::from);
        fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let path = dir.join(REPORT_FILE);
        let file = fs::File::create(&path)
            .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        Ok(Self {
            path,
            file: BufWriter::new(file),
            error: None,
        })
    }

    fn write(&mut self, text: &str) {
        if let Err(e) = self.file.write_all(text.as_bytes()) {
            self.error.get_or_insert(e);
        }
    }

    /// Close the file: its path, or the first error met.
    fn finish(mut self) -> Result<PathBuf, String> {
        if let Err(e) = self.file.flush() {
            self.error.get_or_insert(e);
        }
        match self.error {
            Some(e) => Err(format!("cannot write {}: {e}", self.path.display())),
            None => Ok(self.path),
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "the skip notices and the summaries are test diagnostic output"
)]
fn every_recorded_output_reads_no_worse_with_its_table_than_with_the_legacy_parsers() {
    let Some(root) = std::env::var_os(FIXTURES_ENV).map(PathBuf::from) else {
        eprintln!(
            "skipping: {FIXTURES_ENV} is not set; \
             point it at the fixtures/ directory of a bellwether checkout"
        );
        return;
    };
    let slugs = recorded(&root).unwrap_or_else(|e| panic!("{e}"));
    let every = sample_every();
    let asked = asked_slugs();
    let mut file = Report::create().unwrap_or_else(|e| panic!("{e}"));
    let mut by_table: BTreeMap<&'static str, (Legacy, usize, Counts)> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut replayed = 0;
    in_order(
        &slugs,
        |recorded| replay_model(recorded, every, asked.as_deref()),
        |outcome| match outcome {
            Err(Skipped::NotAsked) => {}
            Err(Skipped::NoTable(slug)) => {
                eprintln!("skipping {slug}: the registry has no table for its model");
            }
            Err(Skipped::NoLegacyParser(slug, table)) => {
                eprintln!(
                    "skipping {slug}: no legacy parser reads the {} family",
                    table.name()
                );
            }
            Err(Skipped::NoPromptRow(slug, table)) => {
                failures.push(format!(
                    "{slug} ({}) is recorded but PROMPTS does not say how its template ends the \
                     prompt; add it",
                    table.name()
                ));
            }
            Ok(model) => {
                print!("{}", model.shown);
                file.write(&model.report);
                let entry = by_table
                    .entry(model.table.name())
                    .or_insert_with(|| (model.legacy, 0, Counts::default()));
                entry.1 += 1;
                entry.2.add(&model.counts);
                failures.extend(
                    model
                        .failures
                        .into_iter()
                        .map(|f| format!("{}/{f}", model.slug)),
                );
                replayed += 1;
            }
        },
    );
    if replayed == 0 {
        eprintln!(
            "skipping: no recorded slug under {} has a table with a legacy parser",
            root.display()
        );
    }
    let mut totals = String::new();
    for (table, (legacy, models, counts)) in &by_table {
        totals.push_str(&format!(
            "{table} (legacy {legacy:?}, {models} model(s)):\n{counts}"
        ));
    }
    let sample = if every > 1 {
        format!(" (1 in {every} cases of each recorded set, {COMMON_SET} whole)")
    } else {
        String::new()
    };
    println!("per table{sample}:\n{totals}");
    file.write(&format!("per table{sample}:\n{totals}"));
    for failure in &failures {
        file.write(&format!("{failure}\n"));
    }
    let path = file.finish().unwrap_or_else(|e| panic!("{e}"));
    println!("per-case report: {}", path.display());
    assert!(
        failures.is_empty(),
        "{} failures; the first {} (every one in {}):\n{}",
        failures.len(),
        failures.len().min(20),
        path.display(),
        failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn every_legacy_name_the_tables_declare_is_registered() {
    let factories = Factories::new().expect("a runtime");
    for table in Table::ALL {
        let legacy = Legacy::of(table);
        if let Some(name) = legacy.tool {
            assert!(
                factories.tools.registry().has_parser(name),
                "{}: no legacy tool parser {name}",
                table.name()
            );
        }
        if let Some(name) = legacy.reasoning {
            assert!(
                factories.reasoning.registry().has_parser(name),
                "{}: no legacy reasoning parser {name}",
                table.name()
            );
        }
        // Where the gateway binds legacy names to the table (a tool parser's or a reasoning
        // parser's), the differential's pair holds one of them.
        let aliases: Vec<&str> = Table::ALIASES
            .iter()
            .filter(|(alias, aliased)| {
                *aliased == table
                    && (factories.tools.registry().has_parser(alias)
                        || factories.reasoning.registry().has_parser(alias))
            })
            .map(|(alias, _)| *alias)
            .collect();
        if !aliases.is_empty() {
            assert!(
                [legacy.tool, legacy.reasoning]
                    .iter()
                    .flatten()
                    .any(|name| aliases.contains(name)),
                "{}: the gateway reads {aliases:?} with this table, the differential {legacy:?}",
                table.name()
            );
        }
    }
}

#[test]
fn a_difference_past_the_shown_prefix_is_located_by_its_first_byte() {
    let long = "x".repeat(300);
    let said = |arguments: String| Said {
        content: None,
        reasoning: None,
        calls: vec![("f".to_string(), arguments)],
        finish: "tool_calls".to_string(),
    };
    let text = located(&said(format!("{long}\n")), &said(long.clone()));
    assert!(
        text.contains("calls[0].arguments differs at byte 300"),
        "{text}"
    );
    assert!(text.contains("\"\\n\""), "{text}");
    assert_eq!(window("héllo", 1), "\"h\"|\"éllo\"");
    assert_eq!(window("héllo", 2), "\"h\"|\"éllo\"");
}

#[test]
fn no_slug_is_in_two_prompt_rows() {
    let mut seen = std::collections::BTreeSet::new();
    for (_, slugs) in PROMPTS {
        for slug in *slugs {
            assert!(seen.insert(*slug), "{slug} is in two rows");
        }
    }
}

#[test]
fn the_verdict_grades_the_agreement_and_names_the_right_side() {
    let said = |content: &str, arguments: &str| Said {
        content: (!content.is_empty()).then(|| content.to_string()),
        reasoning: Some("plan".to_string()),
        calls: vec![("f".to_string(), arguments.to_string())],
        finish: "tool_calls".to_string(),
    };
    let reach = Reach {
        calls: true,
        reasoning: true,
    };
    let reference = said("Done.", "{\"a\": 1}");
    assert_eq!(
        verdict(
            "m/parse/x",
            &reference,
            &reference.clone(),
            &reference,
            reach
        ),
        Verdict::Bitwise
    );
    // The legacy parsers re-serialize the arguments: equal as values.
    assert_eq!(
        verdict(
            "m/parse/x",
            &reference,
            &said("Done.", "{\"a\":1}"),
            &reference,
            reach
        ),
        Verdict::Equal
    );
    // The template's separators at the content's edges: equivalent.
    assert_eq!(
        verdict(
            "m/parse/x",
            &said("\n\nDone.", "{\"a\": 1}"),
            &said("Done.", "{\"a\":1}"),
            &reference,
            reach
        ),
        Verdict::Equivalent
    );
    assert_eq!(
        verdict(
            "m/parse/x",
            &reference,
            &said("Done", "{\"a\": 1}"),
            &reference,
            reach
        ),
        Verdict::SymphonyRight
    );
    assert_eq!(
        verdict(
            "m/parse/x",
            &said("Done", "{\"a\": 1}"),
            &reference,
            &reference,
            reach
        ),
        Verdict::LegacyRight
    );
    assert_eq!(
        verdict(
            "m/parse/x",
            &said("Done", "{\"a\": 1}"),
            &said("", "{\"a\": 1}"),
            &reference,
            reach
        ),
        Verdict::BothWrong
    );
    // A probe's difference is listed, not judged.
    assert_eq!(
        verdict(
            "m/parse/reasoning-with-marker-text",
            &said("Done", "{\"a\": 1}"),
            &reference,
            &reference,
            reach
        ),
        Verdict::Listed
    );
    // A pair without a tool parser cannot read a case with calls.
    assert_eq!(
        verdict(
            "m/parse/x",
            &reference,
            &reference,
            &reference,
            Reach {
                calls: false,
                reasoning: true
            }
        ),
        Verdict::OutOfReach
    );
}

#[test]
fn the_legacy_path_reads_a_qwen_output_as_the_gateway_does_whole_and_streamed() {
    let factories = Factories::new().expect("a runtime");
    let legacy = Legacy::of(Table::Qwen3);
    let tools: Vec<Tool> = serde_json::from_value(serde_json::json!([{
        "type": "function",
        "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}}}
    }]))
    .expect("a tool");
    let text = "<think>\nplan\n</think>\n\nLet me check.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>";
    let whole = legacy_whole(&factories, legacy, text, Some(&tools), false, "stop");
    assert_eq!(whole.reasoning.as_deref(), Some("\nplan\n"));
    assert_eq!(whole.content.as_deref(), Some("\n\nLet me check.\n"));
    assert_eq!(whole.calls.len(), 1);
    assert_eq!(whole.calls[0].0, "f");
    assert_eq!(whole.finish, "tool_calls");
    let cuts: Vec<usize> = text.char_indices().map(|(at, _)| at).skip(1).collect();
    let streamed = legacy_streamed(&factories, legacy, text, &cuts, Some(&tools), false, "stop");
    assert!(
        whole.equivalent(&streamed),
        "whole {whole:?}\nstreamed {streamed:?}"
    );
    // The table says the same, as values.
    let table = Table::Qwen3.format();
    let symphony = symphony_said(&table, &tools, "", text, &[], &EngineFinish::Stop);
    let reach = Reach::of(legacy);
    assert!(matches!(
        verdict("m/parse/x", &symphony, &whole, &symphony, reach),
        Verdict::Bitwise | Verdict::Equal | Verdict::Equivalent
    ));
    // Opened in the prompt: the reasoning parser starts inside the thought.
    let opened = legacy_whole(
        &factories,
        legacy,
        "plan</think>Done.",
        Some(&tools),
        true,
        "stop",
    );
    assert_eq!(opened.reasoning.as_deref(), Some("plan"));
    assert_eq!(opened.content.as_deref(), Some("Done."));
    assert_eq!(opened.finish, "stop");
}
