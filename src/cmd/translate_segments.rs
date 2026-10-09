//! Structure-safe segmentation for `zola translate`.
//!
//! The model never sees page structure. A page (or any front-matter string) is
//! split into [`Piece`]s: `Keep` pieces (block-level HTML, attributes, code,
//! URLs, shortcodes, markdown markers, whitespace) are copied from the source
//! byte for byte, and only short `Text` [`Segment`]s go to the model. A damaged
//! translation can therefore lose words but never tags, attributes, code or
//! links, which is what used to put the footer inside `<main>`.
//!
//! Inline constructs (`<strong>`, `[text](url)`, `` `code` ``) stay inside the
//! segment as fixed-width `XHTML0003X` tokens so the sentence keeps its
//! context. When a segment's tokens come back missing or reordered it is retried
//! as plain-text fragments around the tokens, which cannot lose anything.

use std::sync::OnceLock;

use regex::Regex;

use super::translate::reads_as_prose;

/// A part of a segment: translatable prose, or an inline construct that is
/// carried through the model as a token and restored from the source.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Item {
    Text(String),
    Atom(String),
}

/// One run of prose with its inline constructs, between two hard boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    items: Vec<Item>,
    /// Set for an attribute value (`placeholder="..."`): the quote char the
    /// translation must not contain, escaped as an entity on render.
    quote: Option<char>,
}

/// A piece of a split document.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Piece {
    /// Copied from the source unchanged.
    Keep(String),
    /// Sent to the model.
    Text(Segment),
}

/// A split document: render it back with one translation per segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Doc {
    pieces: Vec<Piece>,
}

impl Doc {
    pub fn split(text: &str) -> Self {
        Self { pieces: Splitter::new().run(text) }
    }

    pub fn segments(&self) -> impl Iterator<Item = &Segment> {
        self.pieces.iter().filter_map(|p| match p {
            Piece::Text(s) => Some(s),
            Piece::Keep(_) => None,
        })
    }

    /// Rebuild the document with `translated[i]` in place of segment `i`. A
    /// segment with no translation keeps its source text.
    pub fn render(&self, translated: &[String]) -> String {
        let mut out = String::new();
        let mut next = translated.iter();
        for piece in &self.pieces {
            match piece {
                Piece::Keep(s) => out.push_str(s),
                Piece::Text(seg) => match next.next() {
                    Some(t) => out.push_str(&seg.escaped(t)),
                    None => out.push_str(&seg.source()),
                },
            }
        }
        out
    }
}

fn token(index: usize) -> String {
    format!("XHTML{index:04}X")
}

fn token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)XHTML(\d{4})X").expect("token regex"))
}

/// True when the text has a letter the model could translate.
fn has_prose(text: &str) -> bool {
    text.chars().any(char::is_alphabetic)
}

fn lead_ws(s: &str) -> &str {
    &s[..s.len() - s.trim_start().len()]
}

fn trail_ws(s: &str) -> &str {
    &s[s.trim_end().len()..]
}

impl Segment {
    /// The original text.
    pub fn source(&self) -> String {
        self.items
            .iter()
            .map(|i| match i {
                Item::Text(s) | Item::Atom(s) => s.as_str(),
            })
            .collect()
    }

    /// A translation made safe to sit where this segment sat: inside an
    /// attribute it must not contain the attribute's own quote.
    fn escaped(&self, translated: &str) -> String {
        match self.quote {
            Some('"') => translated.replace('"', "&quot;"),
            Some(q) => translated.replace(q, "&#39;"),
            None => translated.to_string(),
        }
    }

    /// The text without inline constructs, for prose checks.
    fn plain(&self) -> String {
        self.items
            .iter()
            .filter_map(|i| match i {
                Item::Text(s) => Some(s.as_str()),
                Item::Atom(_) => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn atom_count(&self) -> usize {
        self.items.iter().filter(|i| matches!(i, Item::Atom(_))).count()
    }

    /// The text sent to the model: every inline construct is a padded token.
    pub fn packed(&self) -> String {
        let mut out = String::new();
        let mut n = 0;
        for item in &self.items {
            match item {
                Item::Text(s) => out.push_str(s),
                Item::Atom(_) => {
                    out.push(' ');
                    out.push_str(&token(n));
                    out.push(' ');
                    n += 1;
                }
            }
        }
        out
    }

    /// Put the source constructs back into a translation of [`Self::packed`].
    /// Fails when a token is missing, repeated, reordered or left over.
    pub fn restore(&self, translated: &str) -> Result<String, String> {
        let hits: Vec<(usize, usize, usize)> = token_re()
            .captures_iter(translated)
            .filter_map(|c| {
                let m = c.get(0)?;
                Some((m.start(), m.end(), c.get(1)?.as_str().parse().ok()?))
            })
            .collect();
        let n = self.atom_count();
        if !hits.iter().map(|h| h.2).eq(0..n) {
            return Err(format!(
                "tokens missing, repeated or reordered (expected {n}, got {})",
                hits.len()
            ));
        }
        let mut parts = Vec::with_capacity(n + 1);
        let mut cursor = 0;
        for &(start, end, _) in &hits {
            parts.push(&translated[cursor..start]);
            cursor = end;
        }
        parts.push(&translated[cursor..]);

        let atoms = self.atom_context();
        let mut out = String::new();
        for (i, part) in parts.iter().enumerate() {
            let mut part = *part;
            if i > 0 {
                part = part.trim_start();
            }
            if i < n {
                part = part.trim_end();
            }
            out.push_str(part);
            if let Some((before, atom, after)) = atoms.get(i) {
                out.push_str(before);
                out.push_str(atom);
                out.push_str(after);
            }
        }
        if out.trim().is_empty() {
            return Err("translation is empty".into());
        }
        Ok(out)
    }

    /// For each atom: the source whitespace just before and after it, restored
    /// verbatim so the padding around a token never leaks into the output.
    fn atom_context(&self) -> Vec<(String, String, String)> {
        let mut out = Vec::new();
        for (i, item) in self.items.iter().enumerate() {
            let Item::Atom(atom) = item else { continue };
            let before = match i.checked_sub(1).and_then(|p| self.items.get(p)) {
                Some(Item::Text(t)) => trail_ws(t).to_string(),
                _ => String::new(),
            };
            let after = match self.items.get(i + 1) {
                Some(Item::Text(t)) => lead_ws(t).to_string(),
                _ => String::new(),
            };
            out.push((before, atom.clone(), after));
        }
        out
    }

    /// Plain prose runs, one per request in the fallback pass.
    pub fn fragments(&self) -> Vec<String> {
        self.items
            .iter()
            .filter_map(|i| match i {
                Item::Text(t) if has_prose(t) => Some(t.trim().to_string()),
                _ => None,
            })
            .collect()
    }

    /// Rebuild from per-fragment translations (same order as [`Self::fragments`]).
    pub fn restore_fragments(&self, translated: &[String]) -> String {
        let mut next = translated.iter();
        let mut out = String::new();
        for item in &self.items {
            match item {
                Item::Atom(a) => out.push_str(a),
                Item::Text(t) if has_prose(t) => {
                    out.push_str(lead_ws(t));
                    out.push_str(next.next().map_or(t.trim(), |s| s.trim()));
                    out.push_str(trail_ws(t));
                }
                Item::Text(t) => out.push_str(t),
            }
        }
        out
    }
}

/// One text sent to the model, tagged with the page it belongs to so the
/// transport can cap how many pages ride in one request.
pub struct Request<'a> {
    pub group: usize,
    pub text: &'a str,
}

/// A transport's answer for one text: `(translation, ok)` or an error message.
pub type Answer = Result<(String, bool), String>;

/// Translate segments through `send`, which maps requests to answers in order.
///
/// Pass 1 sends each segment with its tokens. A segment whose answer is
/// echoed, flagged `ok = false` or whose tokens do not round-trip is retried in
/// pass 2 as plain-text fragments, which carry no tokens and cannot damage
/// structure. A fragment the engine returns untouched is accepted only when it
/// does not read as prose (a name, a number); otherwise the segment fails and
/// so does its page. A transport error fails its segment without a retry.
pub fn translate_segments<F>(
    segments: &[(usize, &Segment)],
    send: &mut F,
) -> Vec<Result<String, String>>
where
    F: FnMut(&[Request<'_>]) -> Vec<Answer>,
{
    let packed: Vec<String> = segments.iter().map(|(_, s)| s.packed()).collect();
    let requests: Vec<Request<'_>> = segments
        .iter()
        .zip(&packed)
        .map(|((group, _), text)| Request { group: *group, text })
        .collect();
    let first = send(&requests);

    let mut out: Vec<Option<Result<String, String>>> = vec![None; segments.len()];
    let mut retry: Vec<usize> = Vec::new();
    for (i, (_, seg)) in segments.iter().enumerate() {
        match first.get(i) {
            Some(Ok((text, true))) => match seg.restore(text) {
                Ok(done) if !echoed(seg, &done) => out[i] = Some(Ok(done)),
                _ => retry.push(i),
            },
            Some(Ok((_, false))) if !reads_as_prose(&seg.plain()) => {
                out[i] = Some(Ok(seg.source()));
            }
            Some(Ok((_, false))) => retry.push(i),
            Some(Err(e)) => out[i] = Some(Err(e.clone())),
            None => out[i] = Some(Err("transport returned no answer".into())),
        }
    }
    if !retry.is_empty() {
        retry_as_fragments(segments, &retry, send, &mut out);
    }
    out.into_iter().map(|o| o.unwrap_or_else(|| Err("not translated".into()))).collect()
}

/// A translation that is the source again, for a segment that reads as prose.
fn echoed(seg: &Segment, restored: &str) -> bool {
    reads_as_prose(&seg.plain()) && restored.trim() == seg.source().trim()
}

fn retry_as_fragments<F>(
    segments: &[(usize, &Segment)],
    retry: &[usize],
    send: &mut F,
    out: &mut [Option<Result<String, String>>],
) where
    F: FnMut(&[Request<'_>]) -> Vec<Answer>,
{
    let fragments: Vec<(usize, Vec<String>)> =
        retry.iter().map(|&i| (i, segments[i].1.fragments())).collect();
    let requests: Vec<Request<'_>> = fragments
        .iter()
        .flat_map(|(i, frags)| {
            frags.iter().map(move |text| Request { group: segments[*i].0, text: text.as_str() })
        })
        .collect();
    let answers = send(&requests);
    let mut cursor = 0;
    for (i, frags) in &fragments {
        let mine = answers.get(cursor..cursor + frags.len());
        cursor += frags.len();
        out[*i] = Some(match mine {
            Some(slice) => join_fragments(segments[*i].1, frags, slice),
            None => Err("transport returned too few answers".into()),
        });
    }
}

fn join_fragments(seg: &Segment, sources: &[String], answers: &[Answer]) -> Result<String, String> {
    let mut texts = Vec::with_capacity(sources.len());
    for (source, answer) in sources.iter().zip(answers) {
        match answer {
            Ok((t, true))
                if !t.trim().is_empty()
                    && !(reads_as_prose(source) && t.trim() == source.trim()) =>
            {
                texts.push(t.clone());
            }
            // A name or a number reads the same in every language.
            Ok(_) if !reads_as_prose(source) => texts.push(source.clone()),
            Ok(_) => return Err(format!("untranslated: {}", clip(source))),
            Err(e) => return Err(e.clone()),
        }
    }
    Ok(seg.restore_fragments(&texts))
}

fn clip(s: &str) -> String {
    s.chars().take(60).collect()
}

// ---------------------------------------------------------------------------
// Splitting
// ---------------------------------------------------------------------------

/// Tags whose text is not prose to translate: copied through to the closer.
const RAW_TAGS: &[&str] = &["script", "style", "svg", "pre", "textarea", "template"];
/// Inline tags whose content is code-like: the whole element is one construct.
const CODE_TAGS: &[&str] = &["code", "kbd", "samp", "tt"];
/// Inline tags: they stay inside the sentence as tokens.
const INLINE_TAGS: &[&str] = &[
    "a", "abbr", "b", "bdi", "bdo", "br", "cite", "del", "dfn", "em", "i", "ins", "mark", "q", "s",
    "small", "span", "strong", "sub", "sup", "time", "u", "var", "wbr",
];

fn prefix_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*(?:>\s*)*(?:(?:[-*+]|\d{1,9}[.)])\s+(?:\[[ xX]\]\s+)?)?(?:#{1,6}\s+)?")
            .expect("prefix regex")
    })
}

fn refdef_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*\[[^\]]+\]:\s*\S+").expect("refdef regex"))
}

fn entity_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^&(?:#\d+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]{1,31});")
            .expect("entity regex")
    })
}

/// Reader-facing attributes: their value is prose, not structure.
fn attr_text_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#"(?i)\b(?:alt|title|placeholder|aria-label|aria-description)\s*=\s*(?:"([^"]*)"|'([^']*)')"#,
        )
        .expect("attr text regex")
    })
}

fn anchor_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\{#[\w-]+\}").expect("anchor regex"))
}

fn autolink_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^<(?:[A-Za-z][A-Za-z0-9+.-]{1,31}:[^\s<>]*|[^\s<>@]+@[^\s<>]+)>")
            .expect("autolink regex")
    })
}

/// Multi-line construct the scanner is inside of at the end of a line.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Open {
    None,
    Fence(char, usize),
    Comment,
    /// Inside a tag's attributes, with the open quote if any.
    Tag(Option<char>),
    /// Inside `<script>`, `<style>`, ... until the closing tag.
    Raw(String),
    /// Inside `{{ ... }}` or `{% ... %}` until the closer.
    Shortcode(&'static str),
}

struct Splitter {
    pieces: Vec<Piece>,
    run: Vec<Item>,
    open: Open,
}

enum Angle {
    /// Not markup: `<20 min`, `a < b`.
    Text,
    Comment(usize),
    /// A comment whose `-->` is on a later line.
    OpenComment,
    /// A complete tag ending at `end` (exclusive).
    Tag {
        end: usize,
        name: String,
        closing: bool,
        self_closing: bool,
    },
    /// A tag whose `>` is on a later line.
    OpenTag(Option<char>),
    Autolink(usize),
}

impl Splitter {
    fn new() -> Self {
        Self { pieces: Vec::new(), run: Vec::new(), open: Open::None }
    }

    fn run(mut self, text: &str) -> Vec<Piece> {
        for line in text.split_inclusive('\n') {
            let content = line.trim_end_matches(['\n', '\r']);
            let eol = &line[content.len()..];
            self.line(content);
            self.keep(eol);
        }
        self.flush();
        self.pieces
    }

    fn keep(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        self.flush();
        self.push_keep(s);
    }

    fn push_keep(&mut self, s: &str) {
        if let Some(Piece::Keep(prev)) = self.pieces.last_mut() {
            prev.push_str(s);
        } else {
            self.pieces.push(Piece::Keep(s.to_string()));
        }
    }

    fn text(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        if let Some(Item::Text(prev)) = self.run.last_mut() {
            prev.push_str(s);
        } else {
            self.run.push(Item::Text(s.to_string()));
        }
    }

    fn atom(&mut self, s: &str) {
        self.run.push(Item::Atom(s.to_string()));
    }

    /// Close the current run: peel constructs and whitespace off both ends
    /// into `Keep`, and keep it as a segment only if prose remains.
    fn flush(&mut self) {
        let mut items = std::mem::take(&mut self.run);
        if items.is_empty() {
            return;
        }
        let mut head = String::new();
        let mut tail = String::new();
        loop {
            match items.first() {
                Some(Item::Atom(a)) => {
                    head.push_str(a);
                    items.remove(0);
                }
                Some(Item::Text(t)) if !has_prose(t) => {
                    head.push_str(t);
                    items.remove(0);
                }
                Some(Item::Text(t)) if !lead_ws(t).is_empty() => {
                    head.push_str(lead_ws(t));
                    let rest = t.trim_start().to_string();
                    items[0] = Item::Text(rest);
                }
                _ => break,
            }
        }
        loop {
            match items.last() {
                Some(Item::Atom(a)) => {
                    tail.insert_str(0, a);
                    items.pop();
                }
                Some(Item::Text(t)) if !has_prose(t) => {
                    tail.insert_str(0, t);
                    items.pop();
                }
                Some(Item::Text(t)) if !trail_ws(t).is_empty() => {
                    tail.insert_str(0, trail_ws(t));
                    let rest = t.trim_end().to_string();
                    let last = items.len() - 1;
                    items[last] = Item::Text(rest);
                }
                _ => break,
            }
        }
        if !head.is_empty() {
            self.push_keep(&head);
        }
        if !items.is_empty() {
            self.pieces.push(Piece::Text(Segment { items, quote: None }));
        }
        if !tail.is_empty() {
            self.push_keep(&tail);
        }
    }

    /// Handle one line without its terminator.
    fn line(&mut self, line: &str) {
        let mut rest = line;
        // Finish a construct left open by an earlier line.
        while self.open != Open::None {
            match self.close_open(rest) {
                Some(end) => {
                    self.keep(&rest[..end]);
                    rest = &rest[end..];
                }
                None => {
                    self.keep(rest);
                    return;
                }
            }
        }
        if let Some((marker, len)) = fence_of(rest) {
            self.keep(rest);
            self.open = Open::Fence(marker, len);
            return;
        }
        if is_rule_or_refdef(rest) {
            self.keep(rest);
            return;
        }
        let prefix_len = prefix_re().find(rest).map_or(0, |m| m.end());
        self.keep(&rest[..prefix_len]);
        let body = &rest[prefix_len..];
        let table = rest.trim_start().starts_with('|');
        self.scan(body, table);
    }

    /// Where the open multi-line construct ends within `line`, if it does.
    /// Resets `self.open` when it does.
    fn close_open(&mut self, line: &str) -> Option<usize> {
        let end = match &self.open {
            Open::None => return Some(0),
            Open::Fence(marker, len) => {
                let t = line.trim();
                let closes = t.len() >= *len && t.chars().all(|c| c == *marker);
                closes.then_some(line.len())
            }
            Open::Comment => line.find("-->").map(|i| i + 3),
            Open::Tag(quote) => tag_end(line, *quote).ok(),
            Open::Raw(name) => {
                let lower = line.to_ascii_lowercase();
                let needle = format!("</{name}");
                lower.find(&needle).map(|i| lower[i..].find('>').map_or(line.len(), |j| i + j + 1))
            }
            Open::Shortcode(closer) => line.find(closer).map(|i| i + closer.len()),
        };
        if end.is_some() {
            self.open = Open::None;
        }
        end
    }

    /// Scan inline content, emitting text, atoms and hard boundaries.
    fn scan(&mut self, s: &str, table: bool) {
        let bytes = s.as_bytes();
        let mut i = 0;
        let mut run_start = 0;
        while i < s.len() {
            let c = bytes[i];
            let rest = &s[i..];
            // Every arm either advances `i` past a construct it emitted (after
            // flushing the prose before it) or past plain prose.
            let consumed = match c {
                b'\\' if bytes.get(i + 1).is_some_and(u8::is_ascii_punctuation) => {
                    Some((2, Emit::Atom))
                }
                b'`' => {
                    let k = rest.bytes().take_while(|b| *b == b'`').count();
                    match find_closing_ticks(&rest[k..], k) {
                        Some(end) => Some((k + end, Emit::Atom)),
                        None => {
                            i += k;
                            continue;
                        }
                    }
                }
                b'<' => match angle(rest) {
                    Angle::Text => {
                        i += 1;
                        continue;
                    }
                    Angle::Autolink(end) => Some((end, Emit::Atom)),
                    Angle::Comment(end) => Some((end, Emit::Keep)),
                    Angle::OpenComment => {
                        self.text(&s[run_start..i]);
                        self.keep(rest);
                        self.open = Open::Comment;
                        return;
                    }
                    Angle::OpenTag(quote) => {
                        self.text(&s[run_start..i]);
                        self.keep(rest);
                        self.open = Open::Tag(quote);
                        return;
                    }
                    Angle::Tag { end, name, closing, self_closing } => {
                        self.text(&s[run_start..i]);
                        match self.tag(rest, &rest[..end], &name, closing, self_closing) {
                            Some(n) => {
                                i += n;
                                run_start = i;
                                continue;
                            }
                            None => return, // rest of the line kept; construct left open
                        }
                    }
                },
                b'!' if rest.starts_with("![") => Some((2, Emit::Atom)),
                b'[' if rest.starts_with("[^") => {
                    Some((rest.find(']').map_or(1, |j| j + 1), Emit::Atom))
                }
                b'[' => Some((1, Emit::Atom)),
                b']' => Some((link_target_len(rest), Emit::Atom)),
                b'{' if rest.starts_with("{{") || rest.starts_with("{%") => {
                    let closer = if rest.starts_with("{{") { "}}" } else { "%}" };
                    match rest.find(closer) {
                        Some(j) => Some((j + 2, Emit::Atom)),
                        None => {
                            self.text(&s[run_start..i]);
                            self.keep(rest);
                            self.open = Open::Shortcode(closer);
                            return;
                        }
                    }
                }
                b'{' => anchor_re().find(rest).map(|m| (m.end(), Emit::Atom)),
                b'&' => entity_re().find(rest).map(|m| (m.end(), Emit::Atom)),
                b'h' if is_url_start(s, i) => Some((url_len(rest), Emit::Atom)),
                b'*' | b'_' => {
                    let k = rest.bytes().take_while(|b| *b == c).count();
                    if emphasis_marker(s, i, k, c) {
                        Some((k, Emit::Atom))
                    } else {
                        i += k;
                        continue;
                    }
                }
                b'|' if table => Some((1, Emit::Keep)),
                _ => None,
            };
            match consumed {
                Some((n, emit)) => {
                    self.text(&s[run_start..i]);
                    match emit {
                        Emit::Atom => self.atom(&rest[..n]),
                        Emit::Keep => self.keep(&rest[..n]),
                    }
                    i += n;
                    run_start = i;
                }
                None => i += rest.chars().next().map_or(1, char::len_utf8),
            }
        }
        self.text(&s[run_start..]);
    }

    /// Emit a complete tag. Returns how many bytes of `rest` were consumed, or
    /// `None` when the rest of the line was kept and a construct left open.
    fn tag(
        &mut self,
        rest: &str,
        tag: &str,
        name: &str,
        closing: bool,
        self_closing: bool,
    ) -> Option<usize> {
        let lower = name.to_ascii_lowercase();
        let name = lower.as_str();
        let after = &rest[tag.len()..];
        if !closing && !self_closing && RAW_TAGS.contains(&name) {
            self.keep(tag);
            let lowered = after.to_ascii_lowercase();
            return match lowered.find(&format!("</{name}")) {
                Some(j) => {
                    let close_end = lowered[j..].find('>').map_or(after.len(), |k| j + k + 1);
                    self.keep(&after[..close_end]);
                    Some(tag.len() + close_end)
                }
                None => {
                    self.keep(after);
                    self.open = Open::Raw(name.to_string());
                    None
                }
            };
        }
        if !closing && !self_closing && CODE_TAGS.contains(&name) {
            let lowered = after.to_ascii_lowercase();
            if let Some(j) = lowered.find(&format!("</{name}")) {
                let close_end = lowered[j..].find('>').map_or(after.len(), |k| j + k + 1);
                self.atom(&rest[..tag.len() + close_end]);
                return Some(tag.len() + close_end);
            }
        }
        if !closing && self.split_attr_text(tag) {
            return Some(tag.len());
        }
        if INLINE_TAGS.contains(&name) || CODE_TAGS.contains(&name) {
            self.atom(tag);
        } else {
            self.keep(tag);
        }
        Some(tag.len())
    }

    /// Emit `tag` as Keep pieces around its reader-facing attribute values,
    /// which become segments of their own (a hard boundary in the sentence).
    /// Returns false, emitting nothing, when the tag has no such value.
    fn split_attr_text(&mut self, tag: &str) -> bool {
        let values: Vec<(usize, usize, char)> = attr_text_re()
            .captures_iter(tag)
            .filter_map(|c| {
                let (m, quote) = match (c.get(1), c.get(2)) {
                    (Some(m), _) => (m, '"'),
                    (_, Some(m)) => (m, '\''),
                    _ => return None,
                };
                let templated = m.as_str().contains("{{") || m.as_str().contains("{%");
                (has_prose(m.as_str()) && !templated).then_some((m.start(), m.end(), quote))
            })
            .collect();
        if values.is_empty() {
            return false;
        }
        self.flush();
        let mut cursor = 0;
        for (start, end, quote) in values {
            self.push_keep(&tag[cursor..start]);
            self.pieces.push(Piece::Text(Segment {
                items: vec![Item::Text(tag[start..end].to_string())],
                quote: Some(quote),
            }));
            cursor = end;
        }
        self.push_keep(&tag[cursor..]);
        true
    }
}

/// What to do with a recognised construct.
#[derive(Clone, Copy)]
enum Emit {
    /// Inline: carried through the model as a token.
    Atom,
    /// Hard boundary: copied from the source.
    Keep,
}

/// A code fence opener: marker char and run length.
fn fence_of(line: &str) -> Option<(char, usize)> {
    let t = line.trim_start();
    let marker = t.chars().next().filter(|c| *c == '`' || *c == '~')?;
    let len = t.chars().take_while(|c| *c == marker).count();
    (len >= 3).then_some((marker, len))
}

/// Horizontal rules, setext underlines, table separators and link reference
/// definitions carry no prose.
fn is_rule_or_refdef(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() {
        return false;
    }
    let rule_chars = t.chars().all(|c| matches!(c, '|' | ':' | '-' | ' ' | '\t' | '*' | '_' | '='));
    let marks = t.chars().filter(|c| matches!(c, '-' | '*' | '_' | '=')).count();
    (rule_chars && marks >= 3) || refdef_re().is_match(line)
}

/// End of the run of exactly `k` backticks that closes an inline code span.
fn find_closing_ticks(after_open: &str, k: usize) -> Option<usize> {
    let bytes = after_open.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let run = bytes[i..].iter().take_while(|b| **b == b'`').count();
            if run == k {
                return Some(i + run);
            }
            i += run;
        } else {
            i += 1;
        }
    }
    None
}

/// Length of a `]` plus its link target: `](url "title")` or `][ref]`.
fn link_target_len(rest: &str) -> usize {
    let bytes = rest.as_bytes();
    match bytes.get(1) {
        Some(b'(') => {
            let mut depth = 0usize;
            for (j, b) in bytes.iter().enumerate().skip(1) {
                match b {
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            return j + 1;
                        }
                    }
                    _ => {}
                }
            }
            1
        }
        Some(b'[') => rest[1..].find(']').map_or(1, |j| j + 2),
        _ => 1,
    }
}

fn is_url_start(s: &str, i: usize) -> bool {
    let prev_ok = s[..i].chars().next_back().is_none_or(|c| !c.is_alphanumeric());
    prev_ok && (s[i..].starts_with("http://") || s[i..].starts_with("https://"))
}

fn url_len(rest: &str) -> usize {
    let end = rest
        .find(|c: char| c.is_whitespace() || matches!(c, ')' | '<' | '>' | '"' | '\'' | '['))
        .unwrap_or(rest.len());
    rest[..end].trim_end_matches(['.', ',', ';', ':', '!', '?']).len()
}

/// Whether a run of `*` or `_` is an emphasis marker rather than prose.
fn emphasis_marker(s: &str, i: usize, k: usize, c: u8) -> bool {
    if k >= 2 {
        return true;
    }
    let prev = s[..i].chars().next_back();
    let next = s[i + 1..].chars().next();
    let intraword =
        prev.is_some_and(char::is_alphanumeric) && next.is_some_and(char::is_alphanumeric);
    if c == b'_' && intraword {
        return false; // snake_case
    }
    prev.is_none_or(char::is_whitespace) != next.is_none_or(char::is_whitespace)
}

/// Classify `<...` at the start of `rest`.
fn angle(rest: &str) -> Angle {
    if rest.starts_with("<!--") {
        return match rest.find("-->") {
            Some(j) => Angle::Comment(j + 3),
            None => Angle::OpenComment,
        };
    }
    if let Some(m) = autolink_re().find(rest) {
        return Angle::Autolink(m.end());
    }
    let bytes = rest.as_bytes();
    let closing = bytes.get(1) == Some(&b'/');
    let name_start = 1 + usize::from(closing);
    if !bytes.get(name_start).is_some_and(u8::is_ascii_alphabetic) {
        return Angle::Text;
    }
    let name_len =
        bytes[name_start..].iter().take_while(|b| b.is_ascii_alphanumeric() || **b == b'-').count();
    let after_name = name_start + name_len;
    match bytes.get(after_name) {
        None | Some(b'>' | b'/' | b' ' | b'\t') => {}
        _ => return Angle::Text,
    }
    let name = rest[name_start..after_name].to_string();
    match tag_end(&rest[after_name..], None) {
        Ok(end) => {
            let end = after_name + end;
            Angle::Tag { end, name, closing, self_closing: rest[..end].ends_with("/>") }
        }
        Err(quote) => Angle::OpenTag(quote),
    }
}

/// The byte after the `>` that closes a tag's attributes, honouring quotes
/// (`onclick="a>b"`). `Err` carries the quote still open at end of line.
fn tag_end(s: &str, mut quote: Option<char>) -> Result<usize, Option<char>> {
    for (i, c) in s.char_indices() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '>' => return Ok(i + 1),
            None => {}
        }
    }
    Err(quote)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Render with every segment left as its source: must equal the input.
    fn roundtrip(src: &str) -> String {
        let doc = Doc::split(src);
        let same: Vec<String> = doc.segments().map(Segment::source).collect();
        doc.render(&same)
    }

    fn texts(src: &str) -> Vec<String> {
        Doc::split(src).segments().map(Segment::packed).collect()
    }

    /// A fake engine that uppercases prose and copies the tokens verbatim.
    fn translate_upper(src: &str) -> String {
        let doc = Doc::split(src);
        let out: Vec<String> = doc
            .segments()
            .map(|s| {
                let t = s.packed();
                let toks: Vec<&str> = token_re().find_iter(&t).map(|m| m.as_str()).collect();
                let mut r = String::new();
                for (i, p) in token_re().split(&t).enumerate() {
                    r.push_str(&p.to_uppercase());
                    if let Some(tk) = toks.get(i) {
                        r.push_str(tk);
                    }
                }
                s.restore(&r).expect("restore")
            })
            .collect();
        doc.render(&out)
    }

    fn upper(requests: &[Request<'_>]) -> Vec<Answer> {
        requests.iter().map(|r| Ok((r.text.to_uppercase(), true))).collect()
    }

    /// Run against a real site: `TRANSLATE_CORPUS_DIR=<site>/content cargo test
    /// --bin zola real_corpus -- --ignored --nocapture`. Every default-language
    /// page must split losslessly, and a worst-case "translation" (a model that
    /// drops every token) must still leave the page's skeleton byte-identical.
    #[test]
    #[ignore = "needs TRANSLATE_CORPUS_DIR"]
    fn real_corpus_splits_losslessly_and_survives_a_hostile_model() {
        let dir = std::env::var("TRANSLATE_CORPUS_DIR").expect("TRANSLATE_CORPUS_DIR");
        let mut files = Vec::new();
        super::super::translate::walk_md(std::path::Path::new(&dir), &mut files).unwrap();
        let (mut pages, mut segs, mut max_atoms, mut max_len) = (0, 0, 0, 0);
        for f in files {
            let name = f.file_name().unwrap().to_string_lossy().into_owned();
            if name != "index.md" && name != "_index.md" {
                continue;
            }
            let src = std::fs::read_to_string(&f).unwrap();
            let body = match src.strip_prefix("+++") {
                Some(r) => r.split_once("\n+++").map_or(r, |(_, b)| b).to_string(),
                None => src.clone(),
            };
            let doc = Doc::split(&body);
            let same: Vec<String> = doc.segments().map(Segment::source).collect();
            assert_eq!(doc.render(&same), body, "lossy: {}", f.display());
            let skeleton = doc.render(&vec![String::new(); doc.segments().count()]);
            let hostile: Vec<String> = doc
                .segments()
                .map(|s| s.restore_fragments(&vec!["X".to_string(); s.fragments().len()]))
                .collect();
            let rendered = doc.render(&hostile);
            assert_eq!(
                tag_skeleton(&rendered),
                tag_skeleton(&body),
                "skeleton changed: {}",
                f.display()
            );
            let _ = skeleton;
            pages += 1;
            for s in doc.segments() {
                segs += 1;
                max_atoms = max_atoms.max(s.atom_count());
                max_len = max_len.max(s.packed().len());
            }
        }
        eprintln!("pages={pages} segments={segs} max_atoms/segment={max_atoms} max_len={max_len}");
        assert!(pages > 0);
    }

    /// Every tag, in order, as written.
    fn tag_skeleton(text: &str) -> Vec<String> {
        let re = Regex::new(r"</?[A-Za-z][^<>]*>").unwrap();
        re.find_iter(text)
            .map(|m| attr_text_re().replace_all(m.as_str(), "ATTR").into_owned())
            .collect()
    }

    #[test]
    fn split_then_render_is_lossless() {
        let samples = [
            "# Title\n\nSome **bold** text with [a link](https://x.io/a_b) and `code`.\n",
            "<div class=\"a\"><h3>Hi there</h3><p>Body <strong>text</strong> here.</p></div>\n",
            "| A | B |\n|---|---|\n| one two | three four |\n",
            "```rust\nlet x = 1; // not prose\n```\n\nAfter the fence.\r\n",
            "- item one\n- item two\n\n1. first\n2. second\n",
            "{{ shortcode(a=\"b\") }}\n\n<!-- a comment -->\n\nText.",
            "<script>\nvar x = 'hello world';\n</script>\n<p>After</p>",
            "<a href=\"/x\"\n   class=\"btn\">Click here</a>\n",
            "Price is <20 min and a < b & more &amp; less.\n",
            "Unclosed `tick and **star and [bracket and <b>tag\n",
        ];
        for s in samples {
            assert_eq!(roundtrip(s), s, "lossy split for {s:?}");
        }
    }

    #[test]
    fn block_tags_and_attributes_never_reach_the_model() {
        let all =
            texts("<div class=\"hero\" id=\"top\"><h3 data-x=\"y\">Find talent fast</h3></div>\n");
        assert_eq!(all, vec!["Find talent fast".to_string()]);
    }

    #[test]
    fn inline_tags_become_tokens_inside_one_segment() {
        let all = texts("<p>Screen <strong>every</strong> resume in seconds.</p>\n");
        assert_eq!(all.len(), 1);
        assert!(all[0].contains("XHTML0000X") && all[0].contains("XHTML0001X"));
        assert!(!all[0].contains('<'), "no tag leaks into the request: {:?}", all[0]);
    }

    #[test]
    fn link_targets_and_urls_are_protected() {
        let all = texts("See [our pricing](/pricing/ \"Plans\") or https://curriculo.me/x now.\n");
        assert_eq!(all.len(), 1);
        assert!(!all[0].contains("/pricing/") && !all[0].contains("curriculo.me"), "{:?}", all[0]);
    }

    #[test]
    fn code_fences_inline_code_and_shortcodes_are_kept() {
        let src = "Run `zola build` now.\n\n```sh\nmake all now please\n```\n\n{{ cta(text=\"Book a demo\") }}\n";
        assert_eq!(texts(src), vec!["Run  XHTML0000X  now.".to_string()]);
    }

    #[test]
    fn reader_facing_attribute_values_are_segments_and_the_rest_of_the_tag_is_kept() {
        let src = "<input type=\"text\" id=\"in-role\" placeholder=\"e.g. Senior Backend Engineer\" value=\"Senior\" />\n";
        let doc = Doc::split(src);
        assert_eq!(texts(src), vec!["e.g. Senior Backend Engineer".to_string()]);
        let out = doc.render(&["p. ej. Ingeniero Backend Sénior".to_string()]);
        assert_eq!(
            out,
            "<input type=\"text\" id=\"in-role\" placeholder=\"p. ej. Ingeniero Backend Sénior\" value=\"Senior\" />\n"
        );
    }

    #[test]
    fn a_translated_attribute_cannot_break_out_of_its_quotes() {
        let doc = Doc::split("<img src=\"/a.png\" alt=\"A chart of hires\">\n");
        let out = doc.render(&["Un \"gráfico\" de contrataciones".to_string()]);
        assert_eq!(out, "<img src=\"/a.png\" alt=\"Un &quot;gráfico&quot; de contrataciones\">\n");
    }

    #[test]
    fn templated_or_wordless_attribute_values_are_left_alone() {
        assert!(texts("<a href=\"/x\" title=\"{{ page.title }}\">Go</a>\n").len() == 1);
        assert!(texts("<img alt=\"{{ title }}\" src=\"/a.png\">\n").is_empty());
        assert!(texts("<div title=\"42\"></div>\n").is_empty());
    }

    #[test]
    fn script_style_and_svg_content_is_kept() {
        let src = "<style>\n.a { color: red; }\n</style>\n<svg viewBox=\"0 0 1 1\"><text>Label here</text></svg>\n<p>Real copy</p>\n";
        assert_eq!(texts(src), vec!["Real copy".to_string()]);
    }

    #[test]
    fn quoted_gt_inside_an_attribute_does_not_end_the_tag() {
        let src = "<button onclick=\"if(a>b){go()}\">Start free trial</button>\n";
        assert_eq!(texts(src), vec!["Start free trial".to_string()]);
    }

    #[test]
    fn table_pipes_and_separator_rows_are_kept() {
        let src = "| Feature | Price |\n|---|---|\n| Screening tool | Free forever |\n";
        assert_eq!(texts(src), vec!["Feature", "Price", "Screening tool", "Free forever"]);
    }

    #[test]
    fn list_and_heading_markers_are_kept() {
        let src =
            "## Why teams switch\n\n- Faster screening\n1. Clear scoring\n> Quoted advice here\n";
        assert_eq!(
            texts(src),
            vec!["Why teams switch", "Faster screening", "Clear scoring", "Quoted advice here"]
        );
    }

    #[test]
    fn bare_lt_and_snake_case_stay_prose() {
        let all = texts("Hire in <20 min using snake_case_names today.\n");
        assert_eq!(all, vec!["Hire in <20 min using snake_case_names today.".to_string()]);
    }

    #[test]
    fn numbers_and_symbols_only_lines_are_not_segments() {
        assert!(texts("**2026** — 42%\n").is_empty());
    }

    #[test]
    fn restore_puts_every_construct_back_in_order() {
        let src = "<p>Screen <strong>every</strong> resume, see [pricing](/p/).</p>\n";
        assert_eq!(
            translate_upper(src),
            "<p>SCREEN <strong>EVERY</strong> RESUME, SEE [PRICING](/p/).</p>\n"
        );
    }

    #[test]
    fn restore_rejects_missing_repeated_and_reordered_tokens() {
        let doc = Doc::split("a <b>x</b> and <i>y</i> end\n");
        let seg = doc.segments().next().unwrap();
        assert!(seg.restore("a XHTML0000X x XHTML0001X").is_err(), "missing");
        assert!(
            seg.restore("a XHTML0000X XHTML0000X x XHTML0001X XHTML0002X XHTML0003X").is_err(),
            "repeated"
        );
        assert!(
            seg.restore("XHTML0001X x XHTML0000X y XHTML0002X XHTML0003X").is_err(),
            "reordered"
        );
        assert!(seg.restore("a x y end").is_err(), "all dropped");
    }

    #[test]
    fn restore_accepts_lowercased_tokens_and_restores_source_spacing() {
        let doc = Doc::split("Use <b>bold</b>, then stop.\n");
        let seg = doc.segments().next().unwrap();
        let got = seg.restore("Utilisez xhtml0000x gras xhtml0001x , puis arrêtez.").unwrap();
        assert_eq!(got, "Utilisez <b>gras</b>, puis arrêtez.");
    }

    #[test]
    fn fragments_carry_no_tokens_and_rebuild_with_source_constructs() {
        let doc = Doc::split("Read [the guide](/g/) before you start.\n");
        let seg = doc.segments().next().unwrap();
        assert_eq!(seg.fragments(), vec!["Read", "the guide", "before you start."]);
        let out = seg.restore_fragments(&[
            "Lisez".into(),
            "le guide".into(),
            "avant de commencer.".into(),
        ]);
        assert_eq!(out, "Lisez [le guide](/g/) avant de commencer.");
    }

    #[test]
    fn a_model_that_drops_every_token_still_cannot_damage_structure() {
        let src = "<div><p>Screen <strong>every</strong> resume, see <a href=\"/p/\">pricing</a>.</p></div>\n";
        let doc = Doc::split(src);
        let segs: Vec<(usize, &Segment)> = doc.segments().map(|s| (0, s)).collect();
        let mut calls = 0;
        let mut send = |reqs: &[Request<'_>]| {
            calls += 1;
            let mut answers = upper(reqs);
            if calls == 1 {
                // Pass 1 loses every token; pass 2 (fragments) is clean.
                for a in &mut answers {
                    if let Ok((t, _)) = a {
                        *t = token_re().replace_all(t, "").into_owned();
                    }
                }
            }
            answers
        };
        let results = translate_segments(&segs, &mut send);
        let translated: Vec<String> = results.into_iter().map(|r| r.expect("segment")).collect();
        assert_eq!(calls, 2, "fell back to fragments once");
        assert_eq!(
            doc.render(&translated),
            "<div><p>SCREEN <strong>EVERY</strong> RESUME, SEE <a href=\"/p/\">PRICING</a>.</p></div>\n"
        );
    }

    #[test]
    fn an_untouched_short_name_is_accepted_but_untouched_prose_fails_the_segment() {
        let doc = Doc::split("Curriculo ATS\n\nThis is a long sentence in English.\n");
        let segs: Vec<(usize, &Segment)> = doc.segments().map(|s| (7, s)).collect();
        let mut send = |reqs: &[Request<'_>]| -> Vec<Answer> {
            reqs.iter().map(|r| Ok((r.text.to_string(), false))).collect()
        };
        let results = translate_segments(&segs, &mut send);
        assert_eq!(results[0], Ok("Curriculo ATS".to_string()));
        assert!(results[1].as_ref().unwrap_err().contains("untranslated"));
    }

    #[test]
    fn transport_errors_fail_only_their_segment() {
        let doc = Doc::split("First sentence here.\n\nSecond sentence here.\n");
        let segs: Vec<(usize, &Segment)> = doc.segments().map(|s| (0, s)).collect();
        let mut send = |reqs: &[Request<'_>]| -> Vec<Answer> {
            let mut answers = upper(reqs);
            answers[0] = Err("endpoint down".into());
            answers
        };
        let results = translate_segments(&segs, &mut send);
        assert_eq!(results[0], Err("endpoint down".to_string()));
        assert_eq!(results[1], Ok("SECOND SENTENCE HERE.".to_string()));
    }

    #[test]
    fn an_echoed_prose_segment_is_retried_not_accepted() {
        let doc = Doc::split("This sentence must really be translated.\n");
        let segs: Vec<(usize, &Segment)> = doc.segments().map(|s| (0, s)).collect();
        let mut calls = 0;
        let mut send = |reqs: &[Request<'_>]| -> Vec<Answer> {
            calls += 1;
            if calls == 1 {
                reqs.iter().map(|r| Ok((r.text.to_string(), true))).collect()
            } else {
                upper(reqs)
            }
        };
        let results = translate_segments(&segs, &mut send);
        assert_eq!(results[0], Ok("THIS SENTENCE MUST REALLY BE TRANSLATED.".to_string()));
    }
}
