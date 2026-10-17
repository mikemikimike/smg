//! gpt-oss: a turn of messages, each on a channel. The template opens the turn with
//! `<|start|>assistant`, and a message is a header, `<|message|>`, its body, and `<|end|>` or the
//! stop that ends the turn. The thought is the message on the `analysis` channel; the answer the
//! one on the `final` channel, which `<|return|>` ends; an update the user sees the one on the
//! `commentary` channel; and a call the message whose header names the tool as its recipient,
//! `to=functions.NAME`, on the `commentary` channel, its body one JSON value and its end
//! `<|call|>`. The recipient stands in the header's role section or after the channel, and a
//! content type (`<|constrain|>json`, or `json` alone) may follow it. As the checkpoint's template
//! renders a turn for `openai/gpt-oss-20b`, the recipient after the role and the content type
//! bare, after the generation prompt's `<|start|>assistant`:
//!
//! ```text
//! <|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"city": "Paris"}<|call|>
//! ```
//!
//! and as the format's own documentation writes a call, the recipient after the channel and the
//! content type constrained:
//!
//! ```text
//! <|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city": "Paris"}<|call|>
//! ```
//!
//! A header is read as the format's reader reads it: the channel's name is the terminal
//! (`<|channel|>analysis`, `<|channel|>commentary`, `<|channel|>final`), whatever follows it up to
//! `<|message|>` is the header's rest (a content type such as ` <|constrain|>json` or ` json`,
//! which the model writes on any channel when a response format is in play; a recipient), held in
//! a header state the table frames, and `<|message|>` opens the body. One terminal reads a call in
//! both spellings: `to=functions.` from the turn's own state or from either channel's header (the
//! model writes a function call on the analysis channel too), and a call given as data
//! ([`CallSyntax::Shaped`]) whose header takes the recipient's name up to a space or `<`, skips
//! whatever stands between it and `<|message|>` (the content type, the channel when the recipient
//! came first), and reads one JSON value. A message to a built-in tool of the format (`to=python`,
//! `to=browser.…`, `to=container.…`), on either channel or with the recipient first, has its body
//! read as reasoning and its header framed, as the gateway's own Harmony reader keeps it: out of
//! the answer, in the thought. The text between the messages (`<|start|>assistant`, a space) is
//! framing the template drops. The same template ships with
//! gpt-oss-120b and, up to its whitespace escapes, with the safeguard checkpoints. bellwether has
//! no recorded group for the family: the table stands on the turns the template renders
//! (`tests/fixtures/rendered/gpt_oss.json`, replayed at every cut by `tests/rendered_turns.rs`)
//! and on the format's documented spellings, held in the tests below.
//!
//! Notes on this table, and on what it leaves open: `docs/tables.md`.

use crate::{
    format::{CallSyntax, Emits, Format},
    shape::{self, Arguments, Atom, CallShape, Class, Dialect, NameSource, Pattern, Slot},
};

/// The most bytes a call's header holds undecided, the matcher's own default.
const HOLD_CAP_BYTES: u32 = 8192;

/// The gpt-oss table.
pub fn gpt_oss() -> Format {
    Format::new("gpt_oss")
        .terminal("analysis", "<|channel|>analysis")
        .terminal("commentary", "<|channel|>commentary")
        .terminal("final", "<|channel|>final")
        .terminal("message", "<|message|>")
        .terminal("end", "<|end|>")
        .terminal("return", "<|return|>")
        .terminal("call_open", "to=functions.")
        .terminal("call_close", "<|call|>")
        .terminal("python", "to=python")
        .terminal("browser", "to=browser.")
        .terminal("container", "to=container.")
        .state("message", Emits::Content)
        .state("analysis_header", Emits::Content)
        .state("commentary_header", Emits::Content)
        .state("final_header", Emits::Content)
        .state("builtin_header", Emits::Content)
        .state("builtin", Emits::Reasoning)
        .state("reasoning", Emits::Reasoning)
        .state("content", Emits::Content)
        .state("call", Emits::Arguments)
        .transition("message", "analysis", "analysis_header")
        .transition("message", "commentary", "commentary_header")
        .transition("message", "final", "final_header")
        .transition("analysis_header", "message", "reasoning")
        .transition("commentary_header", "message", "content")
        .transition("final_header", "message", "content")
        .transition("message", "call_open", "call")
        .transition("analysis_header", "call_open", "call")
        .transition("commentary_header", "call_open", "call")
        .transition("message", "python", "builtin_header")
        .transition("message", "browser", "builtin_header")
        .transition("message", "container", "builtin_header")
        .transition("analysis_header", "python", "builtin_header")
        .transition("analysis_header", "browser", "builtin_header")
        .transition("analysis_header", "container", "builtin_header")
        .transition("commentary_header", "python", "builtin_header")
        .transition("commentary_header", "browser", "builtin_header")
        .transition("commentary_header", "container", "builtin_header")
        .transition("builtin_header", "message", "builtin")
        .transition("builtin", "call_close", "message")
        .transition("builtin", "end", "message")
        .transition("reasoning", "end", "message")
        .transition("content", "end", "message")
        .transition("content", "return", "message")
        .transition("call", "call_close", "message")
        .transition("call", "end", "message")
        .calls(CallSyntax::Shaped(shape::intern(call_shape())))
        .framing("message")
        .framing("analysis_header")
        .framing("commentary_header")
        .framing("final_header")
        .framing("builtin_header")
        .closes_at_stop("call")
        .turn_anchor("<|start|>assistant")
        .stop_tokens(["<|return|>", "<|endoftext|>", "<|call|>"])
}

/// A call as the header spells it after `to=functions.`: the recipient's name, every character up
/// to a space or `<` (the format's own reader splits the header on whitespace and cuts the
/// `<|channel|>` and `<|constrain|>` sections out), then whatever stands before `<|message|>`
/// (` <|constrain|>json`, ` json`, `<|channel|>commentary <|constrain|>json`, or nothing), then one
/// JSON value; the stop closes the call, so nothing trails the value.
fn call_shape() -> CallShape {
    CallShape {
        header: Pattern(
            [
                Atom::Run {
                    class: Class::new(
                        [
                            (' ', ' '),
                            ('\t', '\t'),
                            ('\n', '\n'),
                            ('\r', '\r'),
                            ('<', '<'),
                        ],
                        true,
                    ),
                    min: 1,
                    max: None,
                    greedy: true,
                    capture: Some(Slot::Name),
                },
                Atom::Run {
                    class: Class::any(),
                    min: 0,
                    max: None,
                    greedy: false,
                    capture: None,
                },
                Atom::Lit("<|message|>".into()),
            ]
            .into(),
        ),
        name: NameSource::Header,
        arguments: Arguments::Json(Dialect::default()),
        trailer: Pattern::default(),
        word: Class::new([('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false),
        hold_cap: HOLD_CAP_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::Engine,
        event::{DropReason, Event, Events, FinishReason, Text},
        input::{EngineFinish, Input},
        parser::Parser,
        tagged::Declared,
    };

    /// The generation prompt's tail: the user's message and the turn's opener.
    const PROMPT: &str = "<|start|>user<|message|>Weather in Paris?<|end|><|start|>assistant";

    fn run_with(prompt: &str, pieces: &[&str], finish: EngineFinish) -> Vec<Event> {
        let mut parser = Engine::new(gpt_oss(), Declared::default());
        let mut out = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut out,
            )
            .expect("prompt");
        for piece in pieces {
            parser
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text: piece,
                        spans: &[],
                    },
                    &mut out,
                )
                .expect("delta");
        }
        parser.feed(Input::End { finish }, &mut out).expect("end");
        out.drain()
    }

    fn run(prompt: &str, pieces: &[&str]) -> Vec<Event> {
        run_with(prompt, pieces, EngineFinish::Stop)
    }

    /// Every byte the events carry, in order: the output, when the engine accounts for all of it.
    fn bytes(events: &[Event]) -> String {
        events
            .iter()
            .map(|event| match event {
                Event::Content(text) | Event::Reasoning(text) => text.text.as_str(),
                Event::Dropped { text, .. } | Event::Malformed { text, .. } => text.text.as_str(),
                Event::ToolCallStart { source, .. }
                | Event::ToolCallArguments { source, .. }
                | Event::ToolCallEnd { source, .. } => source.text.as_str(),
                Event::ReasoningStart | Event::ReasoningEnd | Event::Finish { .. } => "",
            })
            .collect()
    }

    fn calls(events: &[Event]) -> Vec<(String, String)> {
        let mut calls: Vec<(String, String)> = Vec::new();
        for event in events {
            match event {
                Event::ToolCallStart { name, .. } => calls.push((name.clone(), String::new())),
                Event::ToolCallArguments { json, .. } => {
                    calls.last_mut().expect("a call started").1.push_str(json);
                }
                _ => {}
            }
        }
        calls
    }

    fn content(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Content(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn reasoning(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Reasoning(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn malformed(events: &[Event]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Malformed { text, .. } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The bytes the table dropped as framing or wrapping, in order.
    fn framing(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Dropped {
                    text,
                    why: DropReason::Wrapper,
                } => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The engine's finish: its reason as the engine gave it (the adapters refine it to
    /// `tool_calls`), and the calls counted.
    fn finish(events: &[Event]) -> (FinishReason, u32) {
        events
            .iter()
            .find_map(|event| match event {
                Event::Finish {
                    reason, tool_calls, ..
                } => Some((reason.clone(), *tool_calls)),
                _ => None,
            })
            .expect("a finish")
    }

    /// A call as the format's documentation and its reader spell it, beside the template's: the
    /// recipient after the channel with the constrained content type, spaced and not, with the
    /// bare content type, and with none; the recipient after the role with the constrained type;
    /// the template's own, at the turn's start, where the generation prompt already wrote
    /// `<|start|>assistant`; and a call on the analysis channel, which the model writes too.
    const SPELLINGS: &[&str] = &[
        "<|channel|>analysis to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Paris\"}",
        "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Paris\"}",
        "<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{\"city\": \"Paris\"}",
        "<|channel|>commentary to=functions.get_weather json<|message|>{\"city\": \"Paris\"}",
        "<|channel|>commentary to=functions.get_weather<|message|>{\"city\": \"Paris\"}",
        "<|start|>assistant to=functions.get_weather<|channel|>commentary <|constrain|>json<|message|>{\"city\": \"Paris\"}",
        " to=functions.get_weather<|channel|>commentary json<|message|>{\"city\": \"Paris\"}",
    ];

    #[test]
    fn every_spelling_of_a_call_is_the_same_call_at_every_cut() {
        for output in SPELLINGS {
            let whole = run(PROMPT, &[output]);
            assert_eq!(bytes(&whole), *output, "{output:?}");
            assert_eq!(
                calls(&whole),
                [(
                    "get_weather".to_string(),
                    "{\"city\": \"Paris\"}".to_string()
                )],
                "{output:?}"
            );
            assert_eq!(content(&whole), "", "{output:?}");
            assert_eq!(malformed(&whole), Vec::<&str>::new(), "{output:?}");
            assert_eq!(finish(&whole), (FinishReason::Stop, 1), "{output:?}");
            for (at, _) in output.char_indices().skip(1) {
                let (head, rest) = output.split_at(at);
                let events = run(PROMPT, &[head, rest]);
                assert_eq!(bytes(&events), *output, "{output:?} cut at {at}");
                assert_eq!(calls(&events), calls(&whole), "{output:?} cut at {at}");
                assert_eq!(framing(&events), framing(&whole), "{output:?} cut at {at}");
                assert_eq!(
                    finish(&events),
                    (FinishReason::Stop, 1),
                    "{output:?} cut at {at}"
                );
            }
        }
    }

    #[test]
    fn the_headers_and_the_text_between_the_messages_are_framing() {
        let output = concat!(
            "<|channel|>analysis<|message|>Need the weather.<|end|>",
            "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>",
            "{\"city\": \"Paris\"}"
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            events[..5],
            [
                Event::Dropped {
                    text: Text::uncounted("<|channel|>analysis"),
                    why: DropReason::Wrapper,
                },
                Event::Dropped {
                    text: Text::uncounted("<|message|>"),
                    why: DropReason::Wrapper,
                },
                Event::ReasoningStart,
                Event::Reasoning(Text::uncounted("Need the weather.")),
                Event::Dropped {
                    text: Text::uncounted("<|end|>"),
                    why: DropReason::Wrapper,
                },
            ]
        );
        assert_eq!(
            framing(&events),
            "<|channel|>analysis<|message|><|end|><|start|>assistant to=functions."
        );
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallStart { index: 0, name, source, .. }
                if name == "get_weather" && source.text == "get_weather<|channel|>commentary json<|message|>"
        )));
        assert_eq!(content(&events), "");
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
    }

    #[test]
    fn a_commentary_message_without_a_recipient_is_content_the_user_sees() {
        let output = concat!(
            "<|channel|>commentary<|message|>Checking the weather first.<|end|>",
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json",
            "<|message|>{\"city\": \"Paris\"}<|call|>",
            "<|start|>assistant<|channel|>final<|message|>Done."
        );
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(content(&events), "Checking the weather first.Done.");
        assert_eq!(reasoning(&events), "");
        assert_eq!(
            calls(&events),
            [(
                "get_weather".to_string(),
                "{\"city\": \"Paris\"}".to_string()
            )]
        );
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
    }

    #[test]
    fn a_call_the_turn_ends_is_closed_by_the_stop_and_cut_by_any_other_finish() {
        let piece = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Par";
        let events = run(PROMPT, &[piece]);
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), "{\"city\": \"Par".to_string())]
        );
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
        let events = run_with(PROMPT, &[piece], EngineFinish::Length);
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), "{\"city\": \"Par".to_string())]
        );
        assert_eq!(finish(&events).0, FinishReason::Length);
        // `<|end|>` after a call's value ends the message, and the call with it, as the format's
        // reader ends every message there.
        let events = run(
            PROMPT,
            &["<|channel|>commentary to=functions.get_weather<|message|>{}<|end|><|start|>assistant<|channel|>final<|message|>Done."],
        );
        assert_eq!(
            calls(&events),
            [("get_weather".to_string(), "{}".to_string())]
        );
        assert_eq!(content(&events), "Done.");
        assert_eq!(malformed(&events), Vec::<&str>::new());
    }

    #[test]
    fn a_closing_marker_inside_a_string_argument_ends_the_call_there() {
        // The engine's rule for every table (bellwether #16): a terminal inside a string in a
        // call's arguments is a terminal. The live stream agrees: the model's `<|call|>` is a
        // stop token, so the stream ends there whatever the model meant by it.
        let output = "<|channel|>commentary to=functions.save_note <|constrain|>json<|message|>{\"body\": \"say <|call|> please\"}";
        let events = run(PROMPT, &[output]);
        assert_eq!(bytes(&events), output);
        assert_eq!(
            calls(&events),
            [("save_note".to_string(), "{\"body\": \"say ".to_string())]
        );
        assert_eq!(content(&events), "");
        assert!(framing(&events).ends_with(" please\"}"));
        assert_eq!(finish(&events), (FinishReason::Stop, 1));
    }

    #[test]
    fn a_header_with_a_content_type_opens_its_channel_all_the_same() {
        // The model writes a content type on any channel when a response format is in play; the
        // header's rest is framing and `<|message|>` opens the body.
        for (output, body) in [
            (
                "<|channel|>final <|constrain|>json<|message|>{\"answer\": 42}",
                "{\"answer\": 42}",
            ),
            (
                "<|channel|>final<|constrain|>json<|message|>{\"answer\": 42}",
                "{\"answer\": 42}",
            ),
            (
                "<|channel|>final json<|message|>{\"answer\": 42}",
                "{\"answer\": 42}",
            ),
            (
                "<|channel|>final<|message|>{\"answer\": 42}",
                "{\"answer\": 42}",
            ),
            (
                "<|channel|>commentary <|constrain|>json<|message|>{\"note\": 1}<|end|>",
                "{\"note\": 1}",
            ),
        ] {
            let whole = run(PROMPT, &[output]);
            assert_eq!(bytes(&whole), output, "{output:?}");
            assert_eq!(content(&whole), body, "{output:?}");
            assert_eq!(reasoning(&whole), "", "{output:?}");
            assert!(calls(&whole).is_empty(), "{output:?}");
            assert_eq!(malformed(&whole), Vec::<&str>::new(), "{output:?}");
            for (at, _) in output.char_indices().skip(1) {
                let (head, rest) = output.split_at(at);
                let events = run(PROMPT, &[head, rest]);
                assert_eq!(content(&events), body, "{output:?} cut at {at}");
                assert_eq!(framing(&events), framing(&whole), "{output:?} cut at {at}");
            }
        }
        let events = run(
            PROMPT,
            &["<|channel|>analysis <|constrain|>json<|message|>{\"plan\": 1}<|end|><|start|>assistant<|channel|>final<|message|>Done."],
        );
        assert_eq!(reasoning(&events), "{\"plan\": 1}");
        assert_eq!(content(&events), "Done.");
    }

    #[test]
    fn a_message_to_a_built_in_tool_is_a_thought_on_either_channel() {
        // The template renders no built-in tool for a chat request, and the gateway's own Harmony
        // reader keeps a message to one out of the answer and in the thought: so does the table,
        // on either channel, with the recipient after the channel or before it; the header is
        // framing, the body reasoning, and the next message reads as its header says.
        for (output, body) in [
            ("<|channel|>analysis to=python<|message|>print(1)<|call|>", "print(1)"),
            (
                "<|channel|>commentary to=browser.search <|constrain|>json<|message|>{\"query\": \"x\"}<|call|>",
                "{\"query\": \"x\"}",
            ),
            (
                "<|channel|>commentary to=container.exec<|message|>{\"cmd\": [\"ls\"]}<|end|>",
                "{\"cmd\": [\"ls\"]}",
            ),
            (" to=python<|channel|>analysis<|message|>print(1)<|call|>", "print(1)"),
        ] {
            let output = format!("{output}<|start|>assistant<|channel|>final<|message|>Done.");
            let events = run(PROMPT, &[&output]);
            assert_eq!(bytes(&events), output, "{output:?}");
            assert_eq!(reasoning(&events), body, "{output:?}");
            assert_eq!(
                framing(&events).len() + body.len() + "Done.".len(),
                output.len(),
                "{output:?}: the header and the close are framing"
            );
            assert_eq!(content(&events), "Done.", "{output:?}");
            assert!(calls(&events).is_empty(), "{output:?}");
            assert_eq!(finish(&events), (FinishReason::Stop, 0), "{output:?}");
        }
        // A live turn that ends at a built-in call is never empty: the stop closes the thought.
        let events = run(
            PROMPT,
            &["<|channel|>analysis to=python<|message|>print(1)"],
        );
        assert_eq!(reasoning(&events), "print(1)");
        assert_eq!(finish(&events), (FinishReason::Stop, 0));
        // A recipient with no name opens no call: the header needs one, and the next header reads.
        let events = run(
            PROMPT,
            &["<|channel|>commentary to=functions.<|channel|>final<|message|>Hi"],
        );
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), "Hi");
    }

    #[test]
    fn the_prompt_decides_where_the_output_starts() {
        let format = gpt_oss();
        // After a tool result the template writes a fresh `<|start|>assistant`: the output starts
        // in the turn's own state, and the result's header, which quotes a recipient, is before
        // the anchor.
        let prompt = concat!(
            "<|start|>user<|message|>Weather in Paris?<|end|>",
            "<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>",
            "{\"city\": \"Paris\"}<|call|>",
            "<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>",
            "\"{\\\"temperature\\\": 18}\"<|end|><|start|>assistant"
        );
        assert!(!format.reasoning_open(prompt));
        let events = run(prompt, &["<|channel|>final<|message|>It is 18 degrees."]);
        assert_eq!(content(&events), "It is 18 degrees.");
        assert!(calls(&events).is_empty());
        // A header quoted in an earlier turn moves nothing.
        let quoted = "<|start|>user<|message|>What does <|channel|>final<|message|> mean?<|end|><|start|>assistant";
        let events = run(quoted, &["<|channel|>final<|message|>A header."]);
        assert_eq!(content(&events), "A header.");
        assert_eq!(framing(&events), "<|channel|>final<|message|>");
        // A message the prompt opens is continued: a prefilled answer, a prefilled thought.
        let events = run(
            "<|start|>assistant<|channel|>final<|message|>",
            &["Hi<|return|>"],
        );
        assert_eq!(content(&events), "Hi");
        assert_eq!(framing(&events), "<|return|>");
        assert!(format.reasoning_open("<|start|>assistant<|channel|>analysis<|message|>"));
        let events = run(
            "<|start|>assistant<|channel|>analysis<|message|>",
            &["A thought.<|end|><|start|>assistant<|channel|>final<|message|>Hi"],
        );
        assert_eq!(reasoning(&events), "A thought.");
        assert_eq!(content(&events), "Hi");
    }

    #[test]
    fn the_table_validates_and_names_its_stops_and_anchor() {
        let format = gpt_oss();
        format.validate().expect("a valid table");
        assert_eq!(format.name(), "gpt_oss");
        assert!(format.is_stop_token("<|call|>"));
        assert!(format.is_stop_token("<|return|>"));
        assert!(!format.is_stop_token("<|end|>"));
        assert_eq!(format.turn_opener(), None);
    }
}
