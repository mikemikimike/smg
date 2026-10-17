//! Whether argument bytes are still a valid JSON prefix, judged as they arrive.
//!
//! [`Prefix`] takes the bytes of one JSON value in pieces and says where, if anywhere, they stop
//! being a prefix of a value. It accepts exactly what [`PartialJson`](super::PartialJson) consumes
//! whole in prefix mode and stops where that parser stops, so the promise `ToolCallArguments`
//! makes, that the fragments so far form a valid prefix as far as the prefix parser can tell, is
//! exactly as strong as before; a test runs the two over every cut of a corpus of sound and broken
//! values. What makes it different is cost: it keeps its place between pieces and reads each byte
//! once, where the parser read the whole text again on every piece, which made a long arguments
//! string cost the square of its length.
//!
//! What the prefix parser tolerates, this does too: a literal's prefix at the end (`nu`), a
//! number cut anywhere (`-`, `1.`, `1e+`), a trailing comma before a closing bracket, a value left
//! out before a comma or a closing brace (`[,,]`, `{"a": }`), an escape JSON lacks (`\q` is `q`),
//! a `\u` escape with fewer than four hex digits, and any Unicode whitespace between tokens. What
//! it refuses, where that parser stops: a word that is no literal (judged from the word's first
//! letter, as the parser rolls back to it), a bracket closed by the wrong kind where the parent
//! cannot take it (`{"a": [1}` is whole, the array ending at the brace that closes the object;
//! `[1, 2}` stops at the brace), a surrogate escape without its other half, a container nested
//! past the depth limit, and anything after the value. Inside a string the only error is a broken
//! surrogate pair; a string cut anywhere else is a prefix.

use std::borrow::Cow;

/// The nesting the prefix parser allows, which is [`PartialJson::DEFAULT_MAX_DEPTH`].
const DEEPEST: usize = 32;

/// A JSON prefix being judged as its bytes arrive.
#[derive(Clone, Debug)]
pub struct Prefix {
    /// The containers open around the position, outermost first.
    open: Vec<Container>,
    /// What the grammar admits next.
    expect: Expect,
    /// The token being read, when a string, a literal or a number is under way.
    token: Token,
    /// Bytes judged so far.
    at: usize,
    /// Where the bytes stopped being a prefix, once they did.
    invalid: Option<usize>,
    /// Where the top-level value ended, once something ended it.
    ended: Option<usize>,
    /// The raw control characters met inside strings, in order: JSON forbids them there, and an
    /// assembler writes the fragment that carries one with its escape ([`Self::escaped`]).
    /// Empty for all but the rare output that has one.
    controls: Vec<Control>,
}

/// A raw control character (U+0000..U+001F) inside a string.
#[derive(Clone, Copy, Debug)]
struct Control {
    /// Its byte offset in the whole text.
    at: usize,
    /// Whether a backslash stood right before it, which then opens the escape written for it.
    escaped: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expect {
    /// A value, at the top or inside a container; inside one, the container's close is also
    /// taken (a trailing comma, or `[]`), and a comma stands for a missing value.
    Value,
    /// An object's key, or its close.
    Key,
    /// The colon after a key.
    Colon,
    /// A comma or the container's close, after a value.
    CommaOrClose,
    /// Nothing: the top-level value is complete.
    End,
}

#[derive(Clone, Copy, Debug)]
enum Token {
    None,
    /// A string, as a key when `key` is set.
    Str {
        state: Str,
        key: bool,
    },
    /// `true`, `false` or `null` being spelled; `start` is where it began, `len` how many of its
    /// letters have matched so far.
    Lit {
        start: usize,
        expected: &'static str,
        len: usize,
    },
    Num(Num),
}

#[derive(Clone, Copy, Debug)]
enum Str {
    Text,
    /// After a backslash.
    Escape,
    /// Inside `\u`, with the hex digits read so far and their value.
    Unicode {
        digits: u8,
        value: u32,
    },
    /// After a high surrogate's four digits: only `\u` and a low surrogate may follow.
    AfterHigh,
    /// After the backslash that follows a high surrogate: only `u` may follow; `at` is the
    /// backslash's offset, where the parser stops when something else comes.
    AfterHighBackslash {
        at: usize,
    },
    /// Inside the low surrogate's `\u`.
    Low {
        digits: u8,
        value: u32,
    },
}

#[derive(Clone, Copy, Debug)]
enum Num {
    /// After `-`: a digit must come, though the parser takes `-` alone as a number too.
    Sign,
    /// After a leading `0`: only a fraction or an exponent may follow.
    Zero,
    Integer,
    /// After `.`.
    Fraction,
    /// After `e` or `E`.
    Exponent,
    /// After the exponent's sign.
    ExponentSign,
    ExponentDigits,
}

impl Default for Prefix {
    fn default() -> Self {
        Self {
            open: Vec::new(),
            expect: Expect::Value,
            token: Token::None,
            at: 0,
            invalid: None,
            ended: None,
            controls: Vec::new(),
        }
    }
}

impl Prefix {
    /// Judge the next bytes, which follow every byte judged before, and return where the bytes
    /// stopped being a prefix of a value, if they have: an offset into the whole text, never past
    /// what has been fed, but anywhere at or after the start, bytes judged sound before included,
    /// since the parser rolls back to a word's first letter when the word is no literal, to the
    /// backslash before a surrogate's missing other half, and to the value's start when a
    /// top-level string breaks. A caller that has already passed on the sound bytes clamps the
    /// answer to what it passed on, as the assembler does.
    pub fn feed(&mut self, bytes: &str) -> Option<usize> {
        for (offset, c) in bytes.char_indices() {
            if self.invalid.is_some() {
                break;
            }
            let at = self.at + offset;
            self.step(at, c);
            if self.ended.is_none() && self.expect == Expect::End {
                // The character closed the value, or came after a number or a word that it ended
                // and was refused, as nothing may follow the value.
                self.ended = Some(if self.invalid == Some(at) {
                    at
                } else {
                    at + c.len_utf8()
                });
            }
        }
        if self.invalid.is_none() {
            self.at += bytes.len();
        }
        self.invalid
    }

    /// Where the top-level value ended, once something ended it: just past its closing quote or
    /// bracket, or before the character that ended a number or a word. Whether the bytes up to
    /// there are whole JSON is the caller's to say: this judge also ends a value it tolerates
    /// unfinished, such as an array a brace cut short (`[1}`) or a literal's prefix (`nu`).
    pub fn ended(&self) -> Option<usize> {
        self.ended
    }

    /// Where the word under way starts, while one is: a literal being spelled (`nu`), which is a
    /// prefix of `null` so far, and which the judge refuses from its first letter if it turns out
    /// to be no literal; at the top, from the value's start, as the parser refuses the whole
    /// value then.
    pub fn word_start(&self) -> Option<usize> {
        match self.token {
            Token::Lit { start, .. } if self.invalid.is_none() => {
                Some(if self.open.is_empty() { 0 } else { start })
            }
            _ => None,
        }
    }

    /// Where the surrogate escape under way starts, while a high surrogate awaits its low half
    /// behind a backslash: the backslash, where the judge stops when what follows it is not the
    /// low half, one byte before the character that says so.
    pub fn escape_start(&self) -> Option<usize> {
        match self.token {
            Token::Str {
                state: Str::AfterHighBackslash { at },
                ..
            } if self.invalid.is_none() => Some(at),
            _ => None,
        }
    }

    /// Where the judge may still go back to, if anywhere: the earliest of the word's start
    /// ([`Self::word_start`]) and the escape's ([`Self::escape_start`]), and the value's start
    /// while nothing but whitespace has come at the top, since a word may yet come and have the
    /// whole value refused, whitespace included. An assembler that passes bytes on as fragments
    /// holds from there, since a fragment passed on cannot be taken back.
    ///
    /// A top-level string is not held, though the parser refuses it whole when a surrogate
    /// escape breaks inside it: a string-valued arguments (a stringified call object) streams
    /// as written, and a surrogate that breaks inside one is the one shape where the pieces have
    /// said the string's bytes before the break and the whole says nothing. The assemblers say
    /// so where they choose it.
    pub fn held_from(&self) -> Option<usize> {
        let before_the_value = self.open.is_empty()
            && self.expect == Expect::Value
            && matches!(self.token, Token::None)
            && self.invalid.is_none();
        [
            self.word_start(),
            self.escape_start(),
            if before_the_value { Some(0) } else { None },
        ]
        .into_iter()
        .flatten()
        .min()
    }

    /// `fragment`, the judged bytes from offset `from` on, with every raw control character the
    /// judge met inside a string there written as its JSON escape: `\n`, `\t`, `\r`, `\b`, `\f`
    /// or `\u00XX`, and after a backslash the escape without its own backslash, which already
    /// stands (`\` and a newline read as a newline, and are written `\n`). The bytes themselves,
    /// borrowed, where it met none, which is nearly every fragment. An assembler sends the result
    /// as the fragment's `json` and keeps the model's bytes as its `source`: what a client
    /// assembles is then JSON a strict reader accepts, which a raw control character is not.
    pub fn escaped<'a>(&self, from: usize, fragment: &'a str) -> Cow<'a, str> {
        let end = from + fragment.len();
        let first = self.controls.partition_point(|control| control.at < from);
        if self
            .controls
            .get(first)
            .is_none_or(|control| control.at >= end)
        {
            return Cow::Borrowed(fragment);
        }
        let mut out = String::with_capacity(fragment.len() + 8);
        let mut copied = 0;
        for control in self.controls[first..]
            .iter()
            .take_while(|control| control.at < end)
        {
            let at = control.at - from;
            out.push_str(&fragment[copied..at]);
            if !control.escaped {
                out.push('\\');
            }
            match fragment.as_bytes()[at] {
                b'\n' => out.push('n'),
                b'\t' => out.push('t'),
                b'\r' => out.push('r'),
                0x08 => out.push('b'),
                0x0c => out.push('f'),
                byte => {
                    out.push_str("u00");
                    out.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
                    out.push(char::from_digit(u32::from(byte & 0xf), 16).unwrap_or('0'));
                }
            }
            copied = at + 1;
        }
        out.push_str(&fragment[copied..]);
        Cow::Owned(out)
    }

    /// One character, at byte offset `at` of the whole text.
    fn step(&mut self, at: usize, c: char) {
        match self.token {
            Token::Str { state, key } => self.in_string(state, key, at, c),
            Token::Lit {
                start,
                expected,
                len,
            } => self.in_literal(start, expected, len, at, c),
            Token::Num(state) => self.in_number(state, at, c),
            Token::None => self.between(at, c),
        }
    }

    /// A character where no token is under way.
    fn between(&mut self, at: usize, c: char) {
        if self.expect != Expect::End && c.is_whitespace() {
            return;
        }
        match self.expect {
            Expect::Value => self.value(at, c),
            Expect::Key => match c {
                '}' => self.close(Container::Object),
                '"' => {
                    self.token = Token::Str {
                        state: Str::Text,
                        key: true,
                    };
                }
                _ => self.unwind(at, c),
            },
            Expect::Colon => {
                if c == ':' {
                    self.expect = Expect::Value;
                } else {
                    self.unwind(at, c);
                }
            }
            Expect::CommaOrClose => match (self.open.last(), c) {
                (Some(Container::Object), ',') => self.expect = Expect::Key,
                (Some(Container::Array), ',') => self.expect = Expect::Value,
                (Some(Container::Object), '}') => self.close(Container::Object),
                (Some(Container::Array), ']') => self.close(Container::Array),
                _ => self.unwind(at, c),
            },
            Expect::End => self.refuse(at),
        }
    }

    /// A character where a value may start.
    fn value(&mut self, at: usize, c: char) {
        match c {
            '{' => self.open(at, Container::Object),
            '[' => self.open(at, Container::Array),
            '"' => {
                self.token = Token::Str {
                    state: Str::Text,
                    key: false,
                };
            }
            't' | 'f' | 'n' => {
                let expected = match c {
                    't' => "true",
                    'f' => "false",
                    _ => "null",
                };
                self.token = Token::Lit {
                    start: at,
                    expected,
                    len: 1,
                };
            }
            '-' => self.token = Token::Num(Num::Sign),
            '0' => self.token = Token::Num(Num::Zero),
            '1'..='9' => self.token = Token::Num(Num::Integer),
            // No value starts here: the parser reads a missing value as `null` and leaves the
            // character to the container, which takes a comma or its own close.
            ',' => match self.open.last() {
                Some(Container::Object) => self.expect = Expect::Key,
                Some(Container::Array) => self.expect = Expect::Value,
                None => self.refuse(at),
            },
            '}' if self.open.last() == Some(&Container::Object) => self.close(Container::Object),
            ']' if self.open.last() == Some(&Container::Array) => self.close(Container::Array),
            _ => self.unwind(at, c),
        }
    }

    /// A character the innermost container cannot take. The parser returns that container to
    /// its parent as it stands and lets the parent judge the character, and so on up; at the top
    /// nothing takes it, and the bytes stop being a prefix there. That is how `{"a": [1}` is a
    /// whole object to the parser: the array ends at the brace, which then closes the object.
    fn unwind(&mut self, at: usize, c: char) {
        if self.open.pop().is_none() {
            self.refuse(at);
            return;
        }
        self.value_done();
        self.between(at, c);
    }

    fn open(&mut self, at: usize, container: Container) {
        if self.open.len() >= DEEPEST {
            self.refuse(at);
            return;
        }
        self.open.push(container);
        self.expect = match container {
            Container::Object => Expect::Key,
            Container::Array => Expect::Value,
        };
    }

    /// The innermost container, which is `container`, closes.
    fn close(&mut self, container: Container) {
        debug_assert_eq!(self.open.last(), Some(&container));
        self.open.pop();
        self.value_done();
    }

    /// A value is complete: its container wants a comma or its close; the top wants nothing.
    fn value_done(&mut self) {
        self.token = Token::None;
        self.expect = if self.open.is_empty() {
            Expect::End
        } else {
            Expect::CommaOrClose
        };
    }

    fn in_string(&mut self, state: Str, key: bool, at: usize, c: char) {
        let next = match state {
            Str::Text => match c {
                '\\' => Str::Escape,
                '"' => {
                    if key {
                        self.token = Token::None;
                        self.expect = Expect::Colon;
                    } else {
                        self.value_done();
                    }
                    return;
                }
                // JSON forbids a raw control character here; the parser reads it as the string's
                // text, as the old crate did, and the fragment that carries it is written escaped.
                c if c < ' ' => {
                    self.controls.push(Control { at, escaped: false });
                    Str::Text
                }
                _ => Str::Text,
            },
            Str::Escape => match c {
                'u' => Str::Unicode {
                    digits: 0,
                    value: 0,
                },
                // An escape JSON lacks reads as the character itself: a backslash and a raw
                // newline are a newline, written `\n` with the backslash that already stands.
                c if c < ' ' => {
                    self.controls.push(Control { at, escaped: true });
                    Str::Text
                }
                _ => Str::Text,
            },
            Str::Unicode { digits, value } => match c.to_digit(16) {
                Some(digit) => {
                    let value = (value << 4) | digit;
                    if digits + 1 < 4 {
                        Str::Unicode {
                            digits: digits + 1,
                            value,
                        }
                    } else if (0xD800..0xDC00).contains(&value) {
                        Str::AfterHigh
                    } else if (0xDC00..0xE000).contains(&value) {
                        // A low surrogate alone is no character: the parser stops after its
                        // digits.
                        self.string_broken(at + c.len_utf8(), None);
                        return;
                    } else {
                        Str::Text
                    }
                }
                // Fewer than four hex digits: the parser takes the escape as unfinished, and
                // reads this character as the string's text.
                None => {
                    self.token = Token::Str {
                        state: Str::Text,
                        key,
                    };
                    self.in_string(Str::Text, key, at, c);
                    return;
                }
            },
            Str::AfterHigh => match c {
                '\\' => Str::AfterHighBackslash { at },
                _ => {
                    self.string_broken(at, Some(c));
                    return;
                }
            },
            Str::AfterHighBackslash { at: backslash } => match c {
                'u' => Str::Low {
                    digits: 0,
                    value: 0,
                },
                _ => {
                    self.string_broken(backslash, Some('\\'));
                    return;
                }
            },
            Str::Low { digits, value } => match c.to_digit(16) {
                Some(digit) => {
                    let value = (value << 4) | digit;
                    if digits + 1 < 4 {
                        Str::Low {
                            digits: digits + 1,
                            value,
                        }
                    } else if (0xDC00..0xE000).contains(&value) {
                        Str::Text
                    } else {
                        self.string_broken(at + c.len_utf8(), None);
                        return;
                    }
                }
                None => {
                    self.token = Token::Str {
                        state: Str::Text,
                        key,
                    };
                    self.in_string(Str::Text, key, at, c);
                    return;
                }
            },
        };
        self.token = Token::Str { state: next, key };
    }

    fn in_literal(&mut self, start: usize, expected: &'static str, len: usize, at: usize, c: char) {
        if c.is_alphabetic() {
            // The parser reads the whole word first and compares it: a word that is not the
            // literal, or more than its prefix, is refused from its first letter.
            if expected.as_bytes().get(len).copied() == Some(c as u8) && c.is_ascii() {
                self.token = Token::Lit {
                    start,
                    expected,
                    len: len + 1,
                };
            } else {
                self.refuse_word(start);
            }
            return;
        }
        self.value_done();
        self.step(at, c);
    }

    fn in_number(&mut self, state: Num, at: usize, c: char) {
        let next = match (state, c) {
            (Num::Sign, '0') => Some(Num::Zero),
            (Num::Sign | Num::Integer, '1'..='9') | (Num::Integer, '0') => Some(Num::Integer),
            (Num::Zero | Num::Integer | Num::Sign, '.') => Some(Num::Fraction),
            (Num::Fraction, '0'..='9') => Some(Num::Fraction),
            (Num::Zero | Num::Integer | Num::Sign | Num::Fraction, 'e' | 'E') => {
                Some(Num::Exponent)
            }
            (Num::Exponent, '+' | '-') => Some(Num::ExponentSign),
            (Num::Exponent | Num::ExponentSign | Num::ExponentDigits, '0'..='9') => {
                Some(Num::ExponentDigits)
            }
            _ => None,
        };
        match next {
            Some(state) => self.token = Token::Num(state),
            None => {
                // The number ends before this character, which the container judges.
                self.value_done();
                self.step(at, c);
            }
        }
    }

    /// The bytes stop being a prefix at `at`.
    fn refuse(&mut self, at: usize) {
        if self.invalid.is_none() {
            self.invalid = Some(at);
        }
    }

    /// A string broken at `at` (a surrogate escape without its other half). The parser leaves
    /// the string there, returns its container to the parent as it stands, and the parent judges
    /// what follows: `next` when that is the character at `at`, else the characters to come. At
    /// the top the error is the value's own, and nothing of the value counts.
    fn string_broken(&mut self, at: usize, next: Option<char>) {
        if self.open.pop().is_none() {
            self.refuse(0);
            return;
        }
        self.value_done();
        if let Some(c) = next {
            self.between(at, c);
        }
    }

    /// A word that is no literal: refused from its first letter, as the parser rolls back to it;
    /// at the top the error is the value's, and nothing of the value counts.
    fn refuse_word(&mut self, start: usize) {
        self.refuse(if self.open.is_empty() { 0 } else { start });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::PartialJson;

    /// How many of `text`'s bytes are a prefix by the prefix parser's account: the bytes it
    /// consumes, or none when it refuses the value.
    fn parser_says(text: &str) -> usize {
        match PartialJson::default().parse(text, true) {
            Ok((_, consumed)) => consumed,
            Err(_) => 0,
        }
    }

    /// The same by [`Prefix`]: where it refused, or the whole text.
    fn prefix_says(text: &str) -> usize {
        Prefix::default().feed(text).unwrap_or(text.len())
    }

    /// The corpus: sound values, the broken ones the prefix parser has rules for, and the shapes
    /// the assembler's tests feed.
    const CORPUS: &[&str] = &[
        r#"{"city": "Paris", "days": [1, 2]}"#,
        r#"{"name": "test", "value": 42, "flags": [true, false, null], "ratio": -1.5e2}"#,
        r#"{"a": {"b": {"c": [[], {}, [{}]]}}, "d": ""}"#,
        r#""a string value""#,
        "[1, 2, 3]",
        "-0.5e-3",
        "true",
        r#"{"q": "say \"hi\"\n\t\\ done", "u": "\u00e9\u00E9"}"#,
        r#"{"pair": "\ud83d\ude00", "lone": "\udc00"}"#,
        r#"{"half": "\ud83dx"}"#,
        r#"{"half": "\ud83d\n"}"#,
        r#"{"half": "\ud83dA"}"#,
        r#"{"short": "\u00"}"#,
        r#"{"short": "\uZZZZ"}"#,
        r#"{"odd": "\q\/"}"#,
        r#""\udc00""#,
        "[1, truex]",
        "[1, falsey, 2]",
        "[1, nullx",
        "[nu",
        "truex",
        "trueé",
        "[1, 2, ]",
        r#"{"a": 1, }"#,
        "[,,]",
        r#"{"a": }"#,
        r#"{"a": , "b": 1}"#,
        r#"{"a": [1}"#,
        "[1, 2}",
        r#"{"a" 1}"#,
        "{a: 1}",
        r#"{"a": 1 "b": 2}"#,
        "[1 2]",
        "[01]",
        "[-]",
        "[1.]",
        "[1e+]",
        "[1.5.5]",
        "[1x]",
        "[-x]",
        r#"{"a":1} "#,
        r#"{"a":1}x"#,
        "\u{a0}[1,\u{a0}2]\u{2003}",
        r#"{"a": [1, {"b": [2, {"c": 3}]}]}"#,
        r#"{"name": "f", "arguments": {"k": "v"}}"#,
    ];

    fn deep(levels: usize) -> String {
        let mut text = "[".repeat(levels);
        text.push('1');
        text.push_str(&"]".repeat(levels));
        text
    }

    #[test]
    fn it_stops_exactly_where_the_prefix_parser_stops_at_every_cut() {
        let mut corpus: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        corpus.push(deep(32));
        corpus.push(deep(33));
        corpus.push(deep(40));
        let mut cuts = 0;
        for text in &corpus {
            for cut in 0..=text.len() {
                if !text.is_char_boundary(cut) {
                    continue;
                }
                let prefix = &text[..cut];
                assert_eq!(prefix_says(prefix), parser_says(prefix), "{prefix:?}");
                cuts += 1;
            }
        }
        assert!(cuts > 800, "{cuts} cuts");
    }

    #[test]
    fn the_judge_says_where_it_may_still_go_back_to() {
        let after = |text: &str| {
            let mut prefix = Prefix::default();
            assert_eq!(prefix.feed(text), None, "{text:?}");
            prefix
        };
        // A word: its first letter inside a container, the value's start at the top; and the
        // value's start while nothing but whitespace has come at the top.
        assert_eq!(after(r#"{"a": nu"#).held_from(), Some(6));
        assert_eq!(after("  nu").held_from(), Some(0));
        assert_eq!(after("  ").held_from(), Some(0));
        assert_eq!(after("").held_from(), Some(0));
        assert_eq!(after("  {").held_from(), None);
        assert_eq!(after("  1").held_from(), None);
        assert_eq!(after("null").held_from(), Some(0));
        assert_eq!(after("[null, ").held_from(), None);
        // A surrogate escape awaiting its low half behind a backslash: the backslash.
        assert_eq!(after(r#"{"e": "\ud83c\"#).held_from(), Some(13));
        assert_eq!(after(r#"{"e": "\ud83c\u"#).held_from(), None);
        assert_eq!(after(r#"{"e": "\ud83c"#).held_from(), None);
        // A string streams on, at the top too; a number and a container: nothing to go back to.
        assert_eq!(after(r#""ab\ud83c"#).held_from(), None);
        assert_eq!(after(r#""ab"#).held_from(), None);
        assert_eq!(after("12.").held_from(), None);
        assert_eq!(after(r#"{"a": [1, "#).held_from(), None);
    }

    #[test]
    fn feeding_a_text_in_pieces_judges_it_as_feeding_it_whole_does() {
        let mut corpus: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        corpus.push(deep(33));
        for text in &corpus {
            let whole = Prefix::default().feed(text);
            // Character by character.
            let mut by_char = Prefix::default();
            let mut answer = None;
            for (i, c) in text.char_indices() {
                answer = by_char.feed(&text[i..i + c.len_utf8()]);
                if answer.is_some() {
                    break;
                }
            }
            assert_eq!(answer, whole, "{text:?} character by character");
            // Every two-way split.
            for cut in 0..=text.len() {
                if !text.is_char_boundary(cut) {
                    continue;
                }
                let mut split = Prefix::default();
                let first = split.feed(&text[..cut]);
                let answer = first.or_else(|| split.feed(&text[cut..]));
                assert_eq!(answer, whole, "{text:?} split at {cut}");
            }
        }
    }

    /// A raw control character inside a string is written as its escape, in a key or a value,
    /// after a backslash as the escape's own letter; between tokens a newline is whitespace and
    /// stays; a fragment with none is the bytes themselves, borrowed; and the pieces, each
    /// written from its own offset, join to what the whole gives.
    #[test]
    fn a_raw_control_character_inside_a_string_is_written_as_its_escape() {
        let text = "{\"a\nb\": \"x\ty\u{12}\u{7f}\", \"c\": [1,\n2], \"d\": \"\\\n\"}";
        let expected = "{\"a\\nb\": \"x\\ty\\u0012\u{7f}\", \"c\": [1,\n2], \"d\": \"\\n\"}";
        let mut whole = Prefix::default();
        assert_eq!(whole.feed(text), None);
        let escaped = whole.escaped(0, text);
        assert_eq!(escaped, expected);
        assert!(serde_json::from_str::<serde_json::Value>(&escaped).is_ok());
        assert!(matches!(
            Prefix::default().escaped(0, "{\"a\": 1}"),
            Cow::Borrowed(_)
        ));
        assert!(matches!(whole.escaped(0, "{\"a"), Cow::Borrowed(_)));
        let mut by_char = Prefix::default();
        let mut joined = String::new();
        for (at, c) in text.char_indices() {
            let piece = &text[at..at + c.len_utf8()];
            assert_eq!(by_char.feed(piece), None);
            joined.push_str(&by_char.escaped(at, piece));
        }
        assert_eq!(joined, expected);
        let mut halves = Prefix::default();
        let cut = text.find('y').map_or(0, |at| at + 1);
        assert_eq!(halves.feed(&text[..cut]), None);
        assert_eq!(halves.feed(&text[cut..]), None);
        let mut joined = halves.escaped(0, &text[..cut]).into_owned();
        joined.push_str(&halves.escaped(cut, &text[cut..]));
        assert_eq!(joined, expected);
    }

    #[test]
    fn once_refused_it_stays_refused_where_it_was() {
        let mut prefix = Prefix::default();
        assert_eq!(prefix.feed(r#"{"a": 1 "#), None);
        assert_eq!(prefix.feed(r#""b""#), Some(8));
        assert_eq!(prefix.feed(": 2}"), Some(8));
    }
}
