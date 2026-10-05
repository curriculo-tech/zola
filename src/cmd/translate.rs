//! `zola translate` — generate co-located translation siblings via OpenRouter.
//!
//! For each default-language page or section that has a body or `[extra]` copy,
//! for each non-default language in config.toml: the sibling `<slug>.<lang>.md`
//! is *fresh* iff its `extra.source_hash` equals sha256 of the default page's
//! translatable fields (title, description, body, `[extra]` copy leaves, see
//! [`extra_copy_leaves`]). Output that changes the HTML tags or comes back
//! untranslated is never written. Missing/stale siblings are (re)generated through
//! OpenRouter, written with `extra.source_hash` set, and brand-token-gated
//! (INV-5: a glossary token present in the source must survive verbatim, else
//! the file is NOT written and counts as a failure).
//!
//! Network lives ONLY here — `zola build` stays offline. The OpenRouter call
//! is behind the [`LlmClient`] trait so unit tests mock it without touching
//! the network. With `TRANSLATE_URL` set, pages go through the batched
//! endpoint driver instead: one language is drained at a time, each request
//! carries at most [`BATCH_MAX_PAGES`] pages and [`BATCH_MAX_TEXT_BYTES`] bytes
//! of text, and a per-input `ok` flag fails only the page owning that text.
//!
//! Field/transport contract ported from landing-website
//! `backend/cms/openrouter.py` (ADR-003: `openai/gpt-4o-mini`, JSON-object
//! response, system-prompt → keys). Fields are the Zola page set
//! {title, description, body}, not the CMS's five-field set.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use errors::{Result, anyhow, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
/// Umbrella ADR-003: gpt-4o-mini only, no local models.
const MODEL: &str = "openai/gpt-4o-mini";
/// INV-5: brand tokens that must survive translation verbatim when present in source.
/// "Curriculo ATS" is the product name in the brand guide; "CurriculoATS" is
/// the retired one-word form, kept so older sources still survive verbatim.
const GLOSSARY: &[&str] = &["Curriculo", "Curriculo ATS", "CurriculoATS"];
const MAX_TOKENS: u32 = 16384;
/// Bodies larger than this are translated in H2-sized chunks so each OpenRouter
/// call stays under the JSON output cap (the old 113 KB pillar hit truncation).
const BODY_CHUNK_CHARS: usize = 6_000;
/// Endpoint batching caps: at most this many distinct pages per request.
const BATCH_MAX_PAGES: usize = 4;
/// …and at most this many bytes of `texts` per request. Bytes, not chars —
/// the endpoint's request limit is a wire limit.
const BATCH_MAX_TEXT_BYTES: usize = 16 * 1024;
const FM_DELIM: &str = "+++";

/// Target language code → human name for the system prompt.
fn lang_name(code: &str) -> Option<&'static str> {
    Some(match code {
        "es" => "Spanish",
        "pt" => "Portuguese",
        "zh" => "Chinese (Simplified)",
        "ko" => "Korean",
        "ja" => "Japanese",
        "fr" => "French",
        "ar" => "Arabic",
        _ => return None,
    })
}

/// Translatable fields of a default-language page.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Translatable {
    pub title: String,
    pub description: String,
    pub body: String,
    /// Copy-like string leaves of `[extra]` as (path, text), e.g.
    /// (`v12.cards[1].title`, "Every score is argued"). A translation keeps
    /// the paths and replaces the text. See [`extra_copy_leaves`].
    pub extra: Vec<(String, String)>,
}

/// `[extra]` keys that never carry copy: ids, routes, hashes, machine data.
const EXTRA_SKIP_KEYS: &[&str] = &[
    "source_hash",
    "content_hash",
    "canonical",
    "source_url",
    "template",
    "slug",
    "id",
    "app",
    "author",
    "date",
    "updated",
    "path",
    "url",
    "href",
    "src",
    "image",
    "img",
    "shot",
    "icon",
    "glyph",
    "class",
    "lang",
    "jsonld",
    "noindex",
];
/// Key suffixes with the same meaning as [`EXTRA_SKIP_KEYS`].
const EXTRA_SKIP_SUFFIXES: &[&str] =
    &["_url", "_href", "_src", "_path", "_id", "_icon", "_class", "_image", "_img", "_hash"];

/// LLM translation client. A trait so unit tests inject a mock without a network.
pub trait LlmClient {
    fn translate(&self, fields: &Translatable, lang: &str, key: &str) -> Result<Translatable>;
}

/// Live OpenRouter client (blocking reqwest). Pool built once per run.
pub struct OpenRouterClient {
    client: reqwest::blocking::Client,
}

impl OpenRouterClient {
    pub fn new() -> Result<Self> {
        Ok(Self { client: http_client(Duration::from_secs(120))? })
    }
}

impl Default for OpenRouterClient {
    fn default() -> Self {
        Self::new().expect("reqwest client with default TLS")
    }
}

impl LlmClient for OpenRouterClient {
    fn translate(&self, fields: &Translatable, lang: &str, key: &str) -> Result<Translatable> {
        translate_fields_openrouter(&self.client, fields, lang, key)
    }
}

/// Bulk translation client for the self-hosted endpoint: one request
/// translates many texts at once. A trait so unit tests inject a mock without
/// a network, exactly like [`LlmClient`].
pub trait BatchTranslateClient {
    /// Translate `texts` into `lang`; returns one (translation, ok) pair per
    /// input, in input order. `ok == false` marks that input as returned
    /// untranslated — the caller must not write it.
    fn translate_texts(&self, texts: &[&str], lang: &str) -> Result<Vec<(String, bool)>>;
}

/// Client for a self-hosted translation endpoint.
///
/// Selected by setting `TRANSLATE_URL`; when it is unset the OpenRouter client
/// above is used and behaviour is unchanged. Deliberately knows nothing about
/// any particular engine or deployment — it speaks one small JSON contract:
///
/// ```text
/// POST $TRANSLATE_URL
///   {"texts": ["...", "..."], "target_language": "ko", "source_language": "en",
///    "preserve_terms": ["Acme Widgets"]}
/// → {"translations": ["..."], "ok": [true, true]}
/// ```
///
/// Requests are bulk: the driver packs whole pages in — at most
/// [`BATCH_MAX_PAGES`] distinct pages and [`BATCH_MAX_TEXT_BYTES`] bytes of
/// text per request — and drains one language at a time. Translations come
/// back in request order, one per input; a per-input false `ok` flag means
/// that text came back untranslated, and only the page owning it is failed.
/// `preserve_terms` (#23 review) carries the caller's own untranslatables —
/// brand and product names that must survive verbatim — so the endpoint masks
/// them out of the engine and restores them after (curriculo-ai #1023/#1133).
/// Sourced from `TRANSLATE_PRESERVE_TERMS` (comma-separated) via
/// [`preserve_terms_from_env`], and only included in the body when non-empty.
///
/// Note the endpoint ignores terms shorter than 3 characters, and the INV-5
/// [`glossary_ok`] gate still checks only the built-in [`GLOSSARY`] — caller
/// terms are trusted to the endpoint's mask/restore machinery.
///
/// Failure envelopes are hard failures, never passthroughs: the endpoint may
/// answer HTTP 200 with a scalar `"ok": false` or a mirrored error code
/// (e.g. `"code": 1003`), with `translations` carrying the ORIGINAL source
/// text. Writing that would stamp English content as a fresh translation, so
/// any of those shapes is a whole-batch error and none of its pages are
/// written.
pub struct TranslateApiClient {
    url: String,
    /// Built once per run and reused across chunks/requests: a blocking client
    /// owns a connection pool, so rebuilding it per chunk throws the pool (and
    /// keep-alive) away on every page.
    client: reqwest::blocking::Client,
    /// Caller-supplied untranslatables sent as `preserve_terms` on every
    /// request. Empty means the field is omitted entirely.
    preserve_terms: Vec<String>,
}

impl TranslateApiClient {
    pub fn new(
        url: impl Into<String>,
        timeout: Duration,
        preserve_terms: Vec<String>,
    ) -> Result<Self> {
        Ok(Self { url: url.into(), client: http_client(timeout)?, preserve_terms })
    }
}

/// Shared client constructor. A generous timeout by default: a self-hosted CPU
/// model is far slower than a hosted LLM, and a page body is the whole request.
/// Override with `TRANSLATE_TIMEOUT` (seconds, floor 30 — see [`timeout_from_env`]).
fn http_client(timeout: Duration) -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder().timeout(timeout).build()?)
}

/// `TRANSLATE_TIMEOUT` in seconds: default 300, clamped to a ≥30s floor (a
/// typo like `3` would otherwise time out every real request). Non-numeric
/// values are a config error and fail loudly rather than silently defaulting.
fn timeout_from_env() -> Result<Duration> {
    const DEFAULT_SECS: u64 = 300;
    const MIN_SECS: u64 = 30;
    let Ok(raw) = env::var("TRANSLATE_TIMEOUT") else {
        return Ok(Duration::from_secs(DEFAULT_SECS));
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(Duration::from_secs(DEFAULT_SECS));
    }
    let secs: u64 = raw
        .parse()
        .map_err(|_| anyhow!("TRANSLATE_TIMEOUT must be a number of seconds, got {raw:?}"))?;
    if secs < MIN_SECS {
        log::warn!(
            "translate: TRANSLATE_TIMEOUT={secs}s below the {MIN_SECS}s floor; using {MIN_SECS}s"
        );
        return Ok(Duration::from_secs(MIN_SECS));
    }
    Ok(Duration::from_secs(secs))
}

/// `TRANSLATE_PRESERVE_TERMS`: comma-separated terms the caller needs back
/// verbatim (its own brand/product names), forwarded to the endpoint as
/// `preserve_terms`. Blank segments are dropped, everything else is kept
/// verbatim after a trim — the endpoint does its own cap/filtering (100 terms
/// of ≤100 chars, terms under 3 chars ignored), so this side stays dumb.
fn preserve_terms_from_env() -> Vec<String> {
    env::var("TRANSLATE_PRESERVE_TERMS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

impl BatchTranslateClient for TranslateApiClient {
    fn translate_texts(&self, texts: &[&str], lang: &str) -> Result<Vec<(String, bool)>> {
        if texts.is_empty() {
            return Ok(Vec::new()); // the driver never packs an empty batch
        }
        // NLLB drops wrapping `<p>` and whole `<table>` trees. Pack tags as
        // fixed-width tokens the model copies. OpenRouter does not pack —
        // its prompt already says keep tags. Distinct from curriculo-ai
        // glossary `⟦N⟧` (two maskers on that token collided).
        let packed: Vec<(String, Vec<HtmlTag>)> =
            texts.iter().copied().map(pack_html_tags).collect();
        let packed_refs: Vec<&str> = packed.iter().map(|(s, _)| s.as_str()).collect();
        let payload = build_batch_request(&packed_refs, lang, &self.preserve_terms);
        let resp = self
            .client
            .post(&self.url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(serde_json::to_vec(&payload)?)
            .send()?;
        let status = resp.status();
        let text = resp.text()?;
        if !status.is_success() {
            bail!("translate endpoint HTTP {status}: {}", take200(&text));
        }
        let mut out = parse_batch_translate_response(&text, texts.len())?;
        for (i, ((_, tags), (translation, ok))) in packed.iter().zip(out.iter_mut()).enumerate() {
            if !*ok {
                continue;
            }
            match unpack_html_tags(translation, tags) {
                Ok(restored) => *translation = restored,
                Err(e) => {
                    // Soft-fail this string: driver skips the page, OpenRouter
                    // retry can pick it up. Never write leftover XHTML tokens.
                    log::error!("translate: HTML pack restore failed: {e}");
                    *translation = texts[i].to_string();
                    *ok = false;
                }
            }
        }
        Ok(out)
    }
}

/// `XHTML0003X` — 4-digit index so `XHTML0001X` is not a prefix of `XHTML0010X`.
fn html_pack_token(index: usize) -> String {
    format!("XHTML{index:04}X")
}

fn html_pack_token_re() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| regex::Regex::new(r"(?i)XHTML(\d{4})X").unwrap())
}

struct HtmlTag {
    tag: String,
    /// Whitespace immediately before/after the tag in the source. Restored
    /// verbatim so packing's extra spaces around the token do not leak
    /// (`</strong> .` / CJK `用 <strong>`).
    ws_before: String,
    ws_after: String,
}

/// Replace each HTML tag with a padded ` XHTML0003X ` token. Leaves bare
/// `<` (`<20 min`) and `<!` comments alone — same rule as [`tag_counts`].
/// Surrounding spaces are recorded so unpack can restore the original
/// adjacency (no extra space before `.`, none inside CJK).
fn pack_html_tags(text: &str) -> (String, Vec<HtmlTag>) {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut tags = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            let mut j = i + 1;
            if j < bytes.len() && bytes[j] == b'/' {
                j += 1;
            }
            if j < bytes.len() && bytes[j].is_ascii_alphabetic() {
                if let Some(rel) = text[i + 1..].find('>') {
                    let end = i + 1 + rel + 1;
                    let ws_before = {
                        let n = text[..i]
                            .chars()
                            .rev()
                            .take_while(|c| c.is_whitespace())
                            .collect::<String>();
                        n.chars().rev().collect()
                    };
                    let ws_after: String = text[end..].chars().take_while(|c| c.is_whitespace()).collect();
                    tags.push(HtmlTag {
                        tag: text[i..end].to_string(),
                        ws_before,
                        ws_after,
                    });
                    out.push(' ');
                    out.push_str(&html_pack_token(tags.len() - 1));
                    out.push(' ');
                    i = end;
                    continue;
                }
            }
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    (out, tags)
}

fn placeholder_hits(text: &str) -> Vec<(usize, usize, usize)> {
    html_pack_token_re()
        .captures_iter(text)
        .filter_map(|c| {
            let m = c.get(0)?;
            let idx: usize = c.get(1)?.as_str().parse().ok()?;
            Some((m.start(), m.end(), idx))
        })
        .collect()
}

/// Restore tags. Fails (caller marks ok=false) when:
/// - placeholders are missing, repeated, or out of original order
///   (ja/ko/ar can swap `<strong>` / `</strong>` and still match tag counts)
/// - any `XHTML####X` token remains after restore (engine echoed a spare)
fn unpack_html_tags(text: &str, tags: &[HtmlTag]) -> Result<String> {
    let hits = placeholder_hits(text);
    let expected: Vec<usize> = (0..tags.len()).collect();
    let got: Vec<usize> = hits.iter().map(|h| h.2).collect();
    if got != expected {
        bail!(
            "HTML placeholders missing, repeated, or reordered (expected {expected:?}, got {got:?})"
        );
    }
    let mut out = text.to_string();
    for (start, end, idx) in hits.into_iter().rev() {
        let tag = &tags[idx];
        let mut lo = start;
        let mut hi = end;
        while lo > 0 && out.as_bytes()[lo - 1].is_ascii_whitespace() {
            lo -= 1;
        }
        while hi < out.len() && out.as_bytes()[hi].is_ascii_whitespace() {
            hi += 1;
        }
        let put = format!("{}{}{}", tag.ws_before, tag.tag, tag.ws_after);
        out.replace_range(lo..hi, &put);
    }
    if !placeholder_hits(&out).is_empty() {
        bail!("HTML placeholder left after restore");
    }
    Ok(out)
}

/// Build one bulk request body. `preserve_terms` is included only when
/// non-empty: an empty array is the server default (curriculo-ai #1023), and
/// omitting it keeps the body byte-identical for callers with no glossary.
fn build_batch_request(texts: &[&str], lang: &str, preserve_terms: &[String]) -> Value {
    let mut payload = json!({
        "texts": texts,
        "target_language": lang,
        "source_language": "en",
    });
    if !preserve_terms.is_empty() {
        payload["preserve_terms"] = json!(preserve_terms);
    }
    payload
}

/// Parse a bulk response into one (translation, ok) pair per sent text, in
/// input order. Envelope semantics ported from the old single-page parser; the
/// one deliberate change: a per-string false `ok` flag is SOFT here — the flag
/// rides along on the pair and only the page owning that text is failed.
fn parse_batch_translate_response(text: &str, sent_count: usize) -> Result<Vec<(String, bool)>> {
    let data: Value = serde_json::from_str(text)
        .map_err(|e| anyhow!("translate endpoint non-JSON response: {e}"))?;
    // #1003-style failure envelopes arrive as HTTP 200 with the ORIGINAL source
    // text in `translations`. Any of them means "not translated" — writing the
    // file anyway would mark English content as a fresh translation, so they are
    // hard errors. `ok` may be a bool, a per-string bool array (false = that
    // string came back as source), or null/absent on older deployments.
    // A per-string false flag is SOFT: the flag rides on that index's pair and
    // the driver fails only the page owning the text.
    let mut soft = vec![true; sent_count];
    if let Some(ok) = data.get("ok") {
        match ok {
            Value::Bool(false) => {
                bail!(
                    "translate endpoint reported ok:false — source text returned untranslated, not writing"
                )
            }
            Value::Array(flags) => {
                if flags.len() != sent_count {
                    bail!(
                        "translate endpoint: malformed envelope: ok array length {} != {} sent text(s)",
                        flags.len(),
                        sent_count
                    );
                }
                for (i, f) in flags.iter().enumerate() {
                    // A non-bool entry is an envelope defect, not a "false".
                    match f.as_bool() {
                        Some(true) => {}
                        Some(false) => soft[i] = false,
                        None => bail!(
                            "translate endpoint: malformed envelope: ok[{i}] is not a boolean"
                        ),
                    }
                }
            }
            _ => {}
        }
    }
    // Some gateways mirror the error status in the body of a 200 response
    // (e.g. {"code": 1003}). 0 and 200 mean success; anything else does not.
    for key in ["code", "status", "error_code"] {
        // i64 first so numeric negatives ("code": -1) are not dropped: as_u64 is
        // None on signed JSON numbers, as_str is None on numbers. Do not fold
        // i64 → u64 (try_from drops negatives again). u64 → i64 is only for
        // values that did not fit as_i64.
        let code = data.get(key).and_then(|v| {
            v.as_i64()
                .or_else(|| v.as_u64().and_then(|n| i64::try_from(n).ok()))
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        });
        if let Some(code) = code {
            if code != 0 && code != 200 {
                bail!("translate endpoint error code {code} in `{key}` — not writing");
            }
        }
    }
    let arr = data["translations"].as_array().ok_or_else(|| {
        anyhow!("translate endpoint: no `translations` array: {}", take160(&data.to_string()))
    })?;
    // A short array would silently shift every text onto the wrong page, so the
    // length is a hard error rather than something to paper over.
    if arr.len() != sent_count {
        bail!(
            "translate endpoint returned {} translation(s) for {} text(s)",
            arr.len(),
            sent_count
        );
    }
    let mut out = Vec::with_capacity(sent_count);
    for (i, value) in arr.iter().enumerate() {
        // A non-string entry (null/list/object) used to coerce to "" and write
        // a blank title/description/body. Skip-with-warning is not an option
        // here: there is nothing sane to write for that text, so the request
        // fails and the existing siblings (if any) are left untouched.
        let s = value.as_str().ok_or_else(|| {
            anyhow!("translate endpoint: translations[{i}] is not a string ({}) — not writing blanks", json_kind(value))
        })?.to_string();
        out.push((s, soft[i]));
    }
    Ok(out)
}

fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// One OpenRouter JSON-object call for a (possibly partial) page payload.
fn openrouter_translate_once(
    client: &reqwest::blocking::Client,
    fields: &Translatable,
    lang: &str,
    key: &str,
    body_only: bool,
) -> Result<Translatable> {
    // Any code works: a language outside the named set is described by its code,
    // so adding a language to config.toml is enough to translate into it.
    let name =
        lang_name(lang).map(str::to_string).unwrap_or_else(|| format!("the language {lang}"));
    let extra_texts: Vec<&str> = if body_only {
        Vec::new()
    } else {
        fields.extra.iter().map(|(_, text)| text.as_str()).collect()
    };
    let system = if body_only {
        format!(
            "You translate marketing web page BODY markdown into {name}. Translate ONLY human-readable text. \
            Preserve these brand tokens verbatim, untranslated: {}. \
            Return a single JSON object with exactly these keys: title, description, body. \
            Leave title and description as empty strings. No prose.",
            GLOSSARY.join(", ")
        )
    } else {
        format!(
            "You translate marketing web content into {name}. Translate ONLY human-readable text. \
            Preserve these brand tokens verbatim, untranslated: {}. \
            Return a single JSON object with exactly these keys: title, description, body, extra. \
            `extra` is an array of short UI strings: return an array of the same length, \
            translated item for item in the same order. Keep HTML tags exactly as they are. No prose.",
            GLOSSARY.join(", ")
        )
    };
    let user_title = if body_only { "" } else { fields.title.as_str() };
    let user_desc = if body_only { "" } else { fields.description.as_str() };
    let payload = json!({
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json!({
                "title": user_title, "description": user_desc, "body": fields.body,
                "extra": extra_texts,
            }).to_string()},
        ],
    });
    let body = serde_json::to_vec(&payload)?;
    let resp = client
        .post(OPENROUTER_URL)
        .bearer_auth(key)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()?;
    let status = resp.status();
    let text = resp.text()?;
    if !status.is_success() {
        bail!("OpenRouter HTTP {status}: {}", take200(&text));
    }
    let data: Value =
        serde_json::from_str(&text).map_err(|e| anyhow!("OpenRouter non-JSON response: {e}"))?;
    let content = data["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow!("OpenRouter: unexpected shape: {}", take160(&data.to_string())))?;
    let out: Value = serde_json::from_str(content).map_err(|_| {
        anyhow!("model returned non-JSON (likely truncated at output cap): {}", take160(content))
    })?;
    let extra =
        if body_only { Vec::new() } else { parse_extra_reply(&out["extra"], &fields.extra)? };
    Ok(Translatable {
        title: out["title"].as_str().unwrap_or("").to_string(),
        description: out["description"].as_str().unwrap_or("").to_string(),
        body: out["body"].as_str().unwrap_or("").to_string(),
        extra,
    })
}

/// Pair the model's `extra` array back onto the source paths. A missing,
/// short or non-string array is an error, never a silent shift or a blank.
fn parse_extra_reply(reply: &Value, source: &[(String, String)]) -> Result<Vec<(String, String)>> {
    if source.is_empty() {
        return Ok(Vec::new());
    }
    let items = reply
        .as_array()
        .ok_or_else(|| anyhow!("model reply has no `extra` array ({} expected)", source.len()))?;
    if items.len() != source.len() {
        bail!("model reply `extra` has {} item(s), expected {}", items.len(), source.len());
    }
    source
        .iter()
        .zip(items)
        .map(|((path, _), item)| {
            let text = item.as_str().ok_or_else(|| {
                anyhow!("model reply `extra` item for {path} is {}", json_kind(item))
            })?;
            Ok((path.clone(), text.to_string()))
        })
        .collect()
}

/// Split a long markdown body on H2 boundaries for chunked translation.
fn chunk_body(body: &str) -> Vec<String> {
    let body = body.trim();
    if body.is_empty() || body.len() <= BODY_CHUNK_CHARS {
        return vec![body.to_string()];
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for (i, line) in body.lines().enumerate() {
        let is_h2 = line.starts_with("## ");
        if is_h2 && i > 0 && !current.is_empty() && current.len() >= BODY_CHUNK_CHARS / 2 {
            chunks.push(current.trim_end().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
        if current.len() >= BODY_CHUNK_CHARS {
            chunks.push(current.trim_end().to_string());
            current.clear();
        }
    }
    if !current.is_empty() {
        chunks.push(current.trim_end().to_string());
    }
    if chunks.is_empty() { vec![body.to_string()] } else { chunks }
}

fn translate_fields_openrouter(
    client: &reqwest::blocking::Client,
    fields: &Translatable,
    lang: &str,
    key: &str,
) -> Result<Translatable> {
    translate_fields_impl(
        |f, body_only| openrouter_translate_once(client, f, lang, key, body_only),
        fields,
    )
}

/// Translate title/description once; chunk the body when it exceeds [`BODY_CHUNK_CHARS`].
fn translate_fields<C: LlmClient>(
    client: &C,
    fields: &Translatable,
    lang: &str,
    key: &str,
) -> Result<Translatable> {
    translate_fields_impl(
        |f, body_only| {
            let payload = if body_only {
                Translatable { body: f.body.clone(), ..Translatable::default() }
            } else {
                f.clone()
            };
            client.translate(&payload, lang, key)
        },
        fields,
    )
}

fn translate_fields_impl<F>(mut translate_one: F, fields: &Translatable) -> Result<Translatable>
where
    F: FnMut(&Translatable, bool) -> Result<Translatable>,
{
    let chunks = chunk_body(&fields.body);
    if chunks.len() == 1 {
        return translate_one(fields, false);
    }
    // Title, description and [extra] ride with the first chunk only.
    let head = Translatable { body: chunks[0].clone(), ..fields.clone() };
    let first = translate_one(&head, false)?;
    let mut body_out = first.body;
    for chunk in chunks.iter().skip(1) {
        let partial = Translatable { body: chunk.clone(), ..Translatable::default() };
        let t = translate_one(&partial, true)?;
        if !body_out.is_empty() && !t.body.is_empty() {
            body_out.push_str("\n\n");
        }
        body_out.push_str(&t.body);
    }
    Ok(Translatable {
        title: first.title,
        description: first.description,
        body: body_out,
        extra: first.extra,
    })
}

/// sha256 over the translatable fields, NUL-separated for an unambiguous boundary.
/// `[extra]` copy is appended only when present, so a page without any keeps
/// the hash it had before `[extra]` was translated and is not re-translated.
pub fn source_hash(t: &Translatable) -> String {
    let mut h = Sha256::new();
    h.update(t.title.as_bytes());
    h.update(b"\x00");
    h.update(t.description.as_bytes());
    h.update(b"\x00");
    h.update(t.body.as_bytes());
    for (path, text) in &t.extra {
        h.update(b"\x00");
        h.update(path.as_bytes());
        h.update(b"\x1f");
        h.update(text.as_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Every translated text in field order, for whole-page checks.
fn all_texts(t: &Translatable) -> impl Iterator<Item = &str> {
    [t.title.as_str(), t.description.as_str(), t.body.as_str()]
        .into_iter()
        .chain(t.extra.iter().map(|(_, text)| text.as_str()))
}

/// INV-5 post-check: every glossary token in the source must appear verbatim in
/// the translation. Failure ⇒ the caller must NOT write the file.
pub fn glossary_ok(en: &Translatable, t: &Translatable) -> Result<()> {
    let en_blob = all_texts(en).collect::<Vec<_>>().join(" ");
    let t_blob = all_texts(t).collect::<Vec<_>>().join(" ");
    for token in GLOSSARY {
        if en_blob.contains(token) && !t_blob.contains(token) {
            bail!("glossary: brand token {token:?} lost in translation");
        }
    }
    Ok(())
}

/// Post-check: the translation carries the same HTML tags as the source, and no
/// new bare `<` (a `<` not opening a tag, e.g. "<20 min"). A lost `</a>` or a
/// stray `<5` makes the HTML minifier drop `</article></main>` and the footer
/// renders inside the article, so such output is never written.
/// Also rejects an empty field for a non-empty source (a blank title or CTA),
/// and a change in the count of code fences, shortcodes or markdown link
/// targets, which a model that rewrites code or URLs leaves behind.
pub fn markup_ok(en: &Translatable, t: &Translatable) -> Result<()> {
    let pairs = [("title", &en.title, &t.title), ("description", &en.description, &t.description)]
        .into_iter()
        .chain(std::iter::once(("body", &en.body, &t.body)))
        .map(|(field, a, b)| (field.to_string(), a, b))
        .chain(en.extra.iter().zip(&t.extra).map(|((p, a), (_, b))| (format!("extra.{p}"), a, b)));
    for (field, source, translated) in pairs {
        if !source.trim().is_empty() && translated.trim().is_empty() {
            bail!("markup: {field} came back empty");
        }
    }
    for mark in MARKUP_MARKS {
        let (a, b) = (count_in(en, mark), count_in(t, mark));
        if a != b {
            bail!("markup: `{mark}` count changed in translation ({a}→{b})");
        }
    }
    let source = tag_counts(&all_texts(en).collect::<Vec<_>>().join("\n"));
    let translated = tag_counts(&all_texts(t).collect::<Vec<_>>().join("\n"));
    if source.tags != translated.tags {
        let diff: Vec<String> = source
            .tags
            .keys()
            .chain(translated.tags.keys())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .filter(|k| source.tags.get(*k) != translated.tags.get(*k))
            .map(|k| {
                let (name, close) = k;
                let tag = if *close { format!("</{name}>") } else { format!("<{name}>") };
                format!(
                    "{tag} {}→{}",
                    source.tags.get(k).unwrap_or(&0),
                    translated.tags.get(k).unwrap_or(&0)
                )
            })
            .collect();
        bail!("markup: HTML tags changed in translation ({})", diff.join(", "));
    }
    if translated.bare_lt > source.bare_lt {
        bail!("markup: translation adds a bare `<` that would open a bogus tag");
    }
    Ok(())
}

/// Markdown and Tera syntax a translation must carry over unchanged.
const MARKUP_MARKS: &[&str] = &["```", "{{", "{%", "]("];

fn count_in(t: &Translatable, mark: &str) -> usize {
    all_texts(t).map(|s| s.matches(mark).count()).sum()
}

struct TagCounts {
    /// (lowercased tag name, is closing tag) → count.
    tags: std::collections::BTreeMap<(String, bool), usize>,
    /// `<` not followed by a tag name, `/` or `!`.
    bare_lt: usize,
}

fn tag_counts(text: &str) -> TagCounts {
    let mut tags = std::collections::BTreeMap::new();
    let mut bare_lt = 0;
    let bytes = text.as_bytes();
    for (i, _) in text.match_indices('<') {
        let rest = &bytes[i + 1..];
        let (close, rest) = match rest.first() {
            Some(b'/') => (true, &rest[1..]),
            _ => (false, rest),
        };
        match rest.first() {
            Some(b'!') if !close => continue, // comment or doctype
            Some(c) if c.is_ascii_alphabetic() => {
                let name: String = rest
                    .iter()
                    .take_while(|c| c.is_ascii_alphanumeric() || **c == b'-')
                    .map(|c| (*c as char).to_ascii_lowercase())
                    .collect();
                *tags.entry((name, close)).or_insert(0) += 1;
            }
            _ => bare_lt += 1,
        }
    }
    TagCounts { tags, bare_lt }
}

/// An unchanged string reads as untranslated prose when it has this many
/// lowercase words ("Compare every major ATS")...
const PASSTHROUGH_MIN_LOWER: usize = 2;
/// ...or this many words in all, for Title Case headlines. Below both, it can
/// legitimately read the same in both languages: "FAQ", names such as
/// "Maria Fernanda Silva", "Curriculo ATS REST API".
const PASSTHROUGH_MIN_WORDS: usize = 5;

/// (lowercase-initial words, all words) of three or more letters that a
/// translation would change: brand tokens and acronyms (`ATS`) do not count.
fn translatable_words(text: &str) -> (usize, usize) {
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| w.chars().count() >= 3)
        .filter(|w| !GLOSSARY.contains(w) && !w.chars().all(char::is_uppercase))
        .collect();
    let lower = words.iter().filter(|w| w.starts_with(char::is_lowercase)).count();
    (lower, words.len())
}

fn reads_as_prose(text: &str) -> bool {
    let (lower, all) = translatable_words(text);
    lower >= PASSTHROUGH_MIN_LOWER || all >= PASSTHROUGH_MIN_WORDS
}

/// Post-check for clients without a per-text `ok` flag: a body or string that
/// reads as prose (see [`reads_as_prose`]) and comes back byte-identical was
/// not translated, and stamping it fresh would keep it English forever.
pub fn not_passthrough(en: &Translatable, t: &Translatable) -> Result<()> {
    let pairs = [("title", &en.title, &t.title), ("description", &en.description, &t.description)]
        .into_iter()
        .chain(std::iter::once(("body", &en.body, &t.body)))
        .map(|(field, a, b)| (field.to_string(), a, b))
        .chain(
            en.extra
                .iter()
                .zip(&t.extra)
                .map(|((path, a), (_, b))| (format!("extra.{path}"), a, b)),
        );
    for (field, source, translated) in pairs {
        if reads_as_prose(source) && source.trim() == translated.trim() {
            bail!("passthrough: {field} came back untranslated");
        }
    }
    Ok(())
}

/// Copy-like string leaves of a page's `[extra]` table, in document order.
/// Skips keys that name ids, routes, hashes and machine data, and values that
/// look like URLs, paths, slugs or embedded JSON.
pub fn extra_copy_leaves(fm: &toml::Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if let Some(extra) = fm.get("extra") {
        collect_leaves(extra, "", &mut out);
    }
    out
}

fn collect_leaves(node: &toml::Value, path: &str, out: &mut Vec<(String, String)>) {
    match node {
        toml::Value::Table(table) => {
            for (key, value) in table {
                // A quoted key with `.` or `[` cannot round-trip through a
                // dotted path, so it is never sent rather than written back
                // to the wrong place.
                if is_skipped_key(key) || key.contains(['.', '[', ']']) {
                    continue;
                }
                let child = if path.is_empty() { key.clone() } else { format!("{path}.{key}") };
                collect_leaves(value, &child, out);
            }
        }
        toml::Value::Array(items) => {
            for (i, value) in items.iter().enumerate() {
                collect_leaves(value, &format!("{path}[{i}]"), out);
            }
        }
        toml::Value::String(text) if !path.is_empty() && is_copy(text) => {
            out.push((path.to_string(), text.clone()));
        }
        _ => {}
    }
}

fn is_skipped_key(key: &str) -> bool {
    EXTRA_SKIP_KEYS.contains(&key) || EXTRA_SKIP_SUFFIXES.iter().any(|s| key.ends_with(s))
}

/// True when a string reads as human copy rather than an identifier.
fn is_copy(text: &str) -> bool {
    let text = text.trim();
    if !text.chars().any(char::is_alphabetic) {
        return false; // "01", "$50", "/ 04"
    }
    const MACHINE_PREFIXES: &[&str] =
        &["/", "./", "../", "#", "{", "[", "http://", "https://", "mailto:", "tel:"];
    if MACHINE_PREFIXES.iter().any(|p| text.starts_with(p)) {
        return false;
    }
    // One token in lowercase ASCII with a separator is a slug, key or file
    // name: "impact_scoring", "interface-essential-crown", "01-welcome.jpg".
    let one_token = !text.contains(char::is_whitespace);
    let slug_like = text.chars().all(|c| {
        c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-' | '.' | '/')
    });
    if one_token && slug_like && text.contains(['_', '-', '.', '/']) {
        return false;
    }
    // A language tag: "en-US", "pt-BR".
    if let Some((lang, region)) = text.split_once('-') {
        let is_lang = (2..=3).contains(&lang.len()) && lang.chars().all(|c| c.is_ascii_lowercase());
        let is_region = (2..=4).contains(&region.len())
            && region.chars().all(|c| c.is_ascii_alphanumeric())
            && region.chars().any(|c| c.is_ascii_uppercase());
        if one_token && is_lang && is_region {
            return false;
        }
    }
    // CSS values: "translateY(10px) rotate(3deg)", "var(--accent)".
    let css_call = text.as_bytes().windows(3).any(|w| {
        w[0].is_ascii_alphabetic() && w[1] == b'(' && (w[2].is_ascii_digit() || w[2] == b'-')
    });
    !css_call
}

/// Read the string at `path` (`a.b[2].c`) inside `extra`.
fn get_leaf<'a>(extra: &'a toml::Value, path: &str) -> Option<&'a str> {
    let mut node = extra;
    for part in path_parts(path) {
        node = match (node, part) {
            (toml::Value::Table(t), PathPart::Key(k)) => t.get(k)?,
            (toml::Value::Array(a), PathPart::Index(i)) => a.get(i)?,
            _ => return None,
        };
    }
    node.as_str()
}

/// Write `text` at `path` (`a.b[2].c`) inside `extra`. False when the path
/// does not resolve to a string, so the caller never stamps a sibling that
/// still carries the English leaf.
fn set_leaf(extra: &mut toml::Value, path: &str, text: &str) -> bool {
    let mut node = extra;
    for part in path_parts(path) {
        let next = match (node, part) {
            (toml::Value::Table(t), PathPart::Key(k)) => t.get_mut(k),
            (toml::Value::Array(a), PathPart::Index(i)) => a.get_mut(i),
            _ => None,
        };
        match next {
            Some(n) => node = n,
            None => return false,
        }
    }
    if !node.is_str() {
        return false;
    }
    *node = toml::Value::String(text.to_string());
    true
}

enum PathPart<'a> {
    Key(&'a str),
    Index(usize),
}

fn path_parts(path: &str) -> Vec<PathPart<'_>> {
    let mut parts = Vec::new();
    for segment in path.split('.') {
        let (key, indexes) = segment.split_once('[').unwrap_or((segment, ""));
        if !key.is_empty() {
            parts.push(PathPart::Key(key));
        }
        for index in indexes.split('[').filter(|s| !s.is_empty()) {
            if let Ok(i) = index.trim_end_matches(']').parse() {
                parts.push(PathPart::Index(i));
            }
        }
    }
    parts
}

/// Entry point from `main.rs`. Reads `OPENROUTER_API_KEY` (fail-fast when absent
/// and not a dry-run), then delegates to [`translate_with`].
///
/// Opt-in alternative: when `TRANSLATE_URL` is set the run is served by
/// [`translate_with_endpoint`] (batched, no API key read) instead. Unset —
/// which is the default for every existing site — and nothing below this
/// block changes.
pub fn translate(
    root_dir: &Path,
    config_file: &Path,
    max: Option<usize>,
    dry_run: bool,
) -> Result<()> {
    if let Some(url) = env::var("TRANSLATE_URL").ok().filter(|s| !s.is_empty()) {
        let timeout = timeout_from_env()?;
        let preserve_terms = preserve_terms_from_env();
        log::info!(
            "translate: using translation endpoint at {url} (timeout {timeout:?}, {} preserve term(s)",
            preserve_terms.len()
        );
        let client = TranslateApiClient::new(url, timeout, preserve_terms)?;
        return translate_with_endpoint(root_dir, config_file, max, dry_run, &client);
    }
    let key = if dry_run {
        String::new()
    } else {
        env::var("OPENROUTER_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| anyhow!("OPENROUTER_API_KEY not set — translate needs it"))?
    };
    translate_with(root_dir, config_file, max, dry_run, &key, &OpenRouterClient::new()?)
}

/// Testable core; takes the key and LLM client as parameters so tests inject a
/// mock without a network or a real key.
pub fn translate_with<C: LlmClient>(
    root_dir: &Path,
    config_file: &Path,
    max: Option<usize>,
    dry_run: bool,
    key: &str,
    client: &C,
) -> Result<()> {
    let (_default_lang, langs) = read_langs(config_file)?;
    if langs.is_empty() {
        log::info!("translate: no non-default languages configured; nothing to do");
        return Ok(());
    }
    let lang_set: HashSet<&str> = langs.iter().map(|s| s.as_str()).collect();

    let content_dir = root_dir.join("content");
    let mut pages: Vec<PathBuf> = Vec::new();
    walk_md(&content_dir, &mut pages)?;
    pages.sort();

    let cap = max.unwrap_or(usize::MAX);
    let mut calls = 0usize; // API calls made this run (success + fail)
    let mut failures = 0usize;
    let mut written = 0usize;
    let mut skipped_fresh = 0usize;
    let mut skipped_cap = 0usize;

    for page in &pages {
        // Only default-language, non-section pages.
        let name = page.file_name().unwrap().to_string_lossy().into_owned();
        if !is_default_page(&name, &lang_set) {
            continue;
        }
        let (en_fm, en_body) = match parse_page(page) {
            Ok(v) => v,
            Err(e) => {
                failures += 1;
                log::error!("translate: {}: parse failed: {e}", page.display());
                continue;
            }
        };
        let en = page_source(&en_fm, &en_body);
        // ponytail: nothing meaningful to translate without a body or [extra]
        // copy (title-only stubs). They get picked up once content is authored.
        if en.body.is_empty() && en.extra.is_empty() {
            continue;
        }
        let hash = source_hash(&en);

        for lang in &langs {
            let sibling = sibling_path(page, lang);
            if let Ok((sib_fm, _)) = parse_page(&sibling) {
                if extra_hash(&sib_fm).as_deref() == Some(hash.as_str()) {
                    skipped_fresh += 1;
                    continue; // fresh — skip
                }
            }
            // missing or stale
            if dry_run {
                log::info!("translate [dry-run]: {} → {lang} (stale/missing)", page.display());
                continue;
            }
            if calls >= cap {
                skipped_cap += 1;
                continue; // --max reached; resumes next run
            }
            calls += 1;
            match client.translate(&en, lang, &key).and_then(|t| {
                glossary_ok(&en, &t)?;
                markup_ok(&en, &t)?;
                not_passthrough(&en, &t)?;
                Ok(t)
            }) {
                Ok(t) => {
                    write_sibling(&sibling, &en_fm, &t, &hash)?;
                    written += 1;
                    log::info!("translate: {} → {lang}", page.display());
                }
                Err(e) => {
                    failures += 1;
                    log::error!("translate: {} → {lang} FAILED: {e}", page.display());
                    // file intentionally NOT written on failure
                }
            }
        }
    }

    log::info!(
        "translate: written={written} fresh={skipped_fresh} capped={skipped_cap} calls={calls} failures={failures}"
    );
    if failures > 0 {
        bail!("translate completed with {failures} failure(s)");
    }
    Ok(())
}

/// One (page, lang) unit of work for [`translate_with_endpoint`].
struct BatchJob {
    /// Default-language page; logs read its display path.
    page: PathBuf,
    lang: String,
    en: Translatable,
    /// Frontmatter template copied into the sibling.
    en_fm: toml::Value,
    /// sha256 of `en` — the freshness stamp written into the sibling.
    hash: String,
    sibling: PathBuf,
}

/// Where a flattened request text came from, for positional reassembly.
enum BatchSlot {
    Title,
    Description,
    /// Index into the job's `en.extra`.
    Extra(usize),
    Body(usize),
}

/// A flattened request text: its owning job (index into the current language's
/// job list), its slot, and the text itself.
struct FlatText {
    job: usize,
    slot: BatchSlot,
    text: String,
}

/// Endpoint driver: same walk/parse/hash-gate rules as [`translate_with`], but
/// jobs are batched — one language drained at a time, at most
/// [`BATCH_MAX_PAGES`] pages and [`BATCH_MAX_TEXT_BYTES`] text bytes per
/// request — so a site-wide run is a handful of requests instead of one per
/// page. Testable core; takes the client as a parameter so tests inject a
/// mock without a network or sockets.
fn translate_with_endpoint<C: BatchTranslateClient>(
    root_dir: &Path,
    config_file: &Path,
    max: Option<usize>,
    dry_run: bool,
    client: &C,
) -> Result<()> {
    let (_default_lang, langs) = read_langs(config_file)?;
    if langs.is_empty() {
        log::info!("translate: no non-default languages configured; nothing to do");
        return Ok(());
    }
    let lang_set: HashSet<&str> = langs.iter().map(|s| s.as_str()).collect();

    let content_dir = root_dir.join("content");
    let mut pages: Vec<PathBuf> = Vec::new();
    walk_md(&content_dir, &mut pages)?;
    pages.sort();

    let mut jobs: Vec<BatchJob> = Vec::new();
    let mut failures = 0usize;
    let mut skipped_fresh = 0usize;
    let mut skipped_cap = 0usize;

    // Collection: page-major, same walk/parse/skip rules as translate_with.
    for page in &pages {
        // Only default-language, non-section pages.
        let name = page.file_name().unwrap().to_string_lossy().into_owned();
        if !is_default_page(&name, &lang_set) {
            continue;
        }
        let (en_fm, en_body) = match parse_page(page) {
            Ok(v) => v,
            Err(e) => {
                failures += 1;
                log::error!("translate: {}: parse failed: {e}", page.display());
                continue;
            }
        };
        let en = page_source(&en_fm, &en_body);
        // ponytail: nothing meaningful to translate without a body or [extra]
        // copy (title-only stubs). They get picked up once content is authored.
        if en.body.is_empty() && en.extra.is_empty() {
            continue;
        }
        let hash = source_hash(&en);

        for lang in &langs {
            let sibling = sibling_path(page, lang);
            if let Ok((sib_fm, _)) = parse_page(&sibling) {
                if extra_hash(&sib_fm).as_deref() == Some(hash.as_str()) {
                    skipped_fresh += 1;
                    continue; // fresh — skip
                }
            }
            // missing or stale
            if dry_run {
                log::info!("translate [dry-run]: {} → {lang} (stale/missing)", page.display());
                continue;
            }
            jobs.push(BatchJob {
                page: page.clone(),
                lang: lang.clone(),
                en: en.clone(),
                en_fm: en_fm.clone(),
                hash: hash.clone(),
                sibling,
            });
        }
    }

    // Cap on jobs, page-major — the same iteration-order cut translate_with
    // applies per (page, lang); the rest resumes next run.
    let cap = max.unwrap_or(usize::MAX);
    if jobs.len() > cap {
        skipped_cap = jobs.len() - cap;
        jobs.truncate(cap);
    }
    let calls = jobs.len(); // jobs attempted this run (success + fail)
    let mut written = 0usize;

    // One language at a time: drain all of a language's jobs before the next,
    // so a partial run leaves whole languages consistent.
    for lang in &langs {
        let lang_jobs: Vec<&BatchJob> = jobs.iter().filter(|j| j.lang == *lang).collect();
        if lang_jobs.is_empty() {
            continue;
        }

        // Flatten: per job, title → description → [extra] copy → body chunks,
        // in field order (empty fields are not sent). Every job has ≥1 text —
        // a job has a body or [extra] copy by the skip above.
        let mut flat: Vec<FlatText> = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(lang_jobs.len());
        for (ji, job) in lang_jobs.iter().enumerate() {
            let start = flat.len();
            if !job.en.title.is_empty() {
                flat.push(FlatText { job: ji, slot: BatchSlot::Title, text: job.en.title.clone() });
            }
            if !job.en.description.is_empty() {
                flat.push(FlatText {
                    job: ji,
                    slot: BatchSlot::Description,
                    text: job.en.description.clone(),
                });
            }
            for (ei, (_, text)) in job.en.extra.iter().enumerate() {
                flat.push(FlatText { job: ji, slot: BatchSlot::Extra(ei), text: text.clone() });
            }
            if !job.en.body.is_empty() {
                for (ci, chunk) in chunk_body(&job.en.body).iter().enumerate() {
                    flat.push(FlatText { job: ji, slot: BatchSlot::Body(ci), text: chunk.clone() });
                }
            }
            ranges.push((start, flat.len()));
        }

        // Pack greedily in order: a batch closes when one more text would push
        // it past the page or byte cap. A lone over-cap text (a monster chunk)
        // rides alone in its own batch rather than being dropped.
        let mut batches: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let mut cur_jobs: HashSet<usize> = HashSet::new();
        let mut cur_bytes = 0usize;
        for (i, ft) in flat.iter().enumerate() {
            let adds_page = !cur_jobs.contains(&ft.job);
            if !cur.is_empty()
                && (cur_jobs.len() + usize::from(adds_page) > BATCH_MAX_PAGES
                    || cur_bytes + ft.text.len() > BATCH_MAX_TEXT_BYTES)
            {
                batches.push(std::mem::take(&mut cur));
                cur_jobs.clear();
                cur_bytes = 0;
            }
            cur.push(i);
            cur_jobs.insert(ft.job);
            cur_bytes += ft.text.len();
        }
        if !cur.is_empty() {
            batches.push(cur);
        }

        // One request in flight at a time: a self-hosted CPU endpoint serves a
        // batch no faster for being asked concurrently.
        let mut results: Vec<Option<(String, bool)>> = vec![None; flat.len()];
        let mut failed = vec![false; lang_jobs.len()];
        for batch in &batches {
            let texts: Vec<&str> = batch.iter().map(|&i| flat[i].text.as_str()).collect();
            match client.translate_texts(&texts, lang) {
                Err(e) => {
                    // Envelope/HTTP failure: every job owning a text in this
                    // batch is unwritten and counted, then the run moves on.
                    for &i in batch {
                        let ji = flat[i].job;
                        if !failed[ji] {
                            failed[ji] = true;
                            failures += 1;
                            log::error!(
                                "translate: {} → {lang} FAILED: {e}",
                                lang_jobs[ji].page.display()
                            );
                        }
                    }
                }
                Ok(pairs) => {
                    for (&i, pair) in batch.iter().zip(pairs) {
                        results[i] = Some(pair);
                    }
                }
            }
        }

        // Assemble survivors in job order: slots back onto fields, body chunks
        // rejoined with the same conditional translate_fields_impl uses.
        for (ji, job) in lang_jobs.iter().enumerate() {
            if failed[ji] {
                continue; // already counted + logged above
            }
            let (start, end) = ranges[ji];
            let mut t = Translatable::default();
            let mut ok_all = true;
            let mut body_seen = 0usize;
            for i in start..end {
                // A missing pair means a misbehaving client short-changed the
                // batch; treat it like a false flag, never as a blank field.
                let Some((s, ok)) = &results[i] else {
                    ok_all = false;
                    break;
                };
                if !*ok {
                    ok_all = false; // this text came back untranslated
                }
                match flat[i].slot {
                    BatchSlot::Title => t.title = s.clone(),
                    BatchSlot::Description => t.description = s.clone(),
                    BatchSlot::Extra(ei) => t.extra.push((job.en.extra[ei].0.clone(), s.clone())),
                    BatchSlot::Body(ci) => {
                        // Chunks reassemble in request order — the slot index
                        // is that invariant, so it is checked, not assumed.
                        debug_assert_eq!(ci, body_seen, "body chunks out of request order");
                        body_seen += 1;
                        if !t.body.is_empty() && !s.is_empty() {
                            t.body.push_str("\n\n");
                        }
                        t.body.push_str(s);
                    }
                }
            }
            if !ok_all {
                failures += 1;
                log::error!(
                    "translate: {} → {lang} FAILED: endpoint reported ok=false — source text returned untranslated, not writing",
                    job.page.display()
                );
                continue;
            }
            if let Err(e) = glossary_ok(&job.en, &t).and_then(|()| markup_ok(&job.en, &t)) {
                failures += 1;
                log::error!("translate: {} → {lang} FAILED: {e}", job.page.display());
                continue; // file intentionally NOT written on failure
            }
            write_sibling(&job.sibling, &job.en_fm, &t, &job.hash)?;
            written += 1;
            log::info!("translate: {} → {lang}", job.page.display());
        }
    }

    log::info!(
        "translate: written={written} fresh={skipped_fresh} capped={skipped_cap} calls={calls} failures={failures}"
    );
    if failures > 0 {
        bail!("translate completed with {failures} failure(s)");
    }
    Ok(())
}

/// `zola translate --adopt`: stamp existing siblings that already hold a
/// complete translation of the current source (written by hand, say) as fresh,
/// so a later run does not replace them. No network and no key. A sibling is
/// adopted only when every `[extra]` copy path is present and the same checks a
/// machine translation must pass hold: glossary, markup, not left in English.
/// Nothing but `extra.source_hash` changes in an adopted file.
pub fn adopt(root_dir: &Path, config_file: &Path) -> Result<()> {
    let (mut adopted, mut rejected, mut fresh) = (0usize, 0usize, 0usize);
    for_each_sibling(root_dir, config_file, |sib| {
        if sib.is_fresh() {
            fresh += 1;
            return Ok(());
        }
        match sib.check() {
            Ok(()) => {
                sib.write_hash(Some(sib.hash))?;
                adopted += 1;
                log::info!("translate --adopt: {} adopted", sib.path.display());
            }
            Err(e) => {
                rejected += 1;
                log::warn!("translate --adopt: {} not adopted: {e}", sib.path.display());
            }
        }
        Ok(())
    })?;
    log::info!("translate --adopt: adopted={adopted} rejected={rejected} fresh={fresh}");
    Ok(())
}

/// `zola translate --recheck`: run the output checks on siblings already
/// stamped fresh and remove the stamp from any that fail (English left in
/// place, HTML tags lost, brand token dropped, `[extra]` copy missing), so the
/// next `zola translate` run regenerates them. No network and no key; copy is
/// never changed. Fails when it cleared anything, so CI surfaces it.
pub fn recheck(root_dir: &Path, config_file: &Path) -> Result<()> {
    let (mut cleared, mut passed) = (0usize, 0usize);
    for_each_sibling(root_dir, config_file, |sib| {
        if !sib.is_fresh() {
            return Ok(()); // stale or unstamped: translate already redoes it
        }
        match sib.check() {
            Ok(()) => passed += 1,
            Err(e) => {
                sib.write_hash(None)?;
                cleared += 1;
                log::warn!("translate --recheck: {} cleared: {e}", sib.path.display());
            }
        }
        Ok(())
    })?;
    log::info!("translate --recheck: cleared={cleared} passed={passed}");
    if cleared > 0 {
        bail!(
            "translate --recheck cleared {cleared} sibling(s); run `zola translate` to redo them"
        );
    }
    Ok(())
}

/// An existing sibling of a default-language page, with its source.
struct Sibling<'a> {
    path: &'a Path,
    en: &'a Translatable,
    hash: &'a str,
    fm: toml::Value,
    body: String,
}

impl Sibling<'_> {
    fn is_fresh(&self) -> bool {
        extra_hash(&self.fm).as_deref() == Some(self.hash)
    }

    /// The checks a machine translation must pass before it is written.
    fn check(&self) -> Result<()> {
        let t = sibling_translation(self.en, &self.fm, &self.body)?;
        glossary_ok(self.en, &t)?;
        markup_ok(self.en, &t)?;
        not_passthrough(self.en, &t)
    }

    /// Set (`Some`) or remove (`None`) `extra.source_hash`; nothing else changes.
    /// The `source_hash` line is edited in place so hand-written front matter
    /// keeps its order and comments; the front matter is re-serialized only
    /// when the line edit does not parse back to the expected value.
    fn write_hash(&self, hash: Option<&str>) -> Result<()> {
        let mut fm = self.fm.clone();
        match hash {
            Some(h) => stamp_hash(&mut fm, h),
            None => {
                if let Some(et) = fm.get_mut("extra").and_then(|e| e.as_table_mut()) {
                    et.remove("source_hash");
                }
            }
        }
        let text = fs::read_to_string(self.path)?;
        let edited =
            rewrite_hash_line(&text, hash).filter(|t| front_matter_of(t).as_ref() == Some(&fm));
        let out = match edited {
            Some(t) => t,
            None => {
                let fm_str =
                    toml::to_string(&fm).map_err(|e| anyhow!("serialize frontmatter: {e}"))?;
                format!("+++\n{fm_str}+++\n{}", self.body)
            }
        };
        fs::write(self.path, out)?;
        Ok(())
    }
}

/// `text` with the `[extra]` `source_hash = "..."` line set to `hash` (added
/// under `[extra]`, or in a new `[extra]` table, when missing) or removed
/// (`None`). Line endings and every other line are kept. `None` when the
/// front matter delimiters are not found.
fn rewrite_hash_line(text: &str, hash: Option<&str>) -> Option<String> {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    if lines.first()?.trim() != FM_DELIM {
        return None;
    }
    let close = 1 + lines[1..].iter().position(|l| l.trim() == FM_DELIM)?;
    let eol = if lines[0].ends_with("\r\n") { "\r\n" } else { "\n" };
    let (mut table, mut header, mut existing) = (String::new(), None, None);
    for (i, line) in lines.iter().enumerate().take(close).skip(1) {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            table = trimmed.trim_matches(|c| c == '[' || c == ']').trim().to_string();
            if table == "extra" {
                header = Some(i);
            }
        } else if table == "extra"
            && trimmed.split('=').next().map(str::trim) == Some("source_hash")
        {
            existing = Some(i);
        }
    }
    let new_line = hash.map(|h| format!("source_hash = \"{h}\"{eol}"));
    let mut out: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
    match (existing, new_line) {
        (Some(i), Some(l)) => out[i] = l,
        (Some(i), None) => {
            out.remove(i);
        }
        (None, Some(l)) => match header {
            Some(h) => out.insert(h + 1, l),
            None => out.insert(close, format!("[extra]{eol}{l}")),
        },
        (None, None) => {}
    }
    Some(out.concat())
}

/// The parsed TOML front matter of a page's full text.
fn front_matter_of(text: &str) -> Option<toml::Value> {
    let mut parts = text.splitn(3, FM_DELIM);
    parts.next()?;
    toml::from_str(parts.next()?).ok()
}

/// Visit every existing sibling of every translatable default-language page.
fn for_each_sibling<F>(root_dir: &Path, config_file: &Path, mut visit: F) -> Result<()>
where
    F: FnMut(&Sibling<'_>) -> Result<()>,
{
    let (_default_lang, langs) = read_langs(config_file)?;
    let lang_set: HashSet<&str> = langs.iter().map(|s| s.as_str()).collect();
    let mut pages: Vec<PathBuf> = Vec::new();
    walk_md(&root_dir.join("content"), &mut pages)?;
    pages.sort();
    for page in &pages {
        let name = page.file_name().unwrap().to_string_lossy().into_owned();
        if !is_default_page(&name, &lang_set) {
            continue;
        }
        let (en_fm, en_body) = match parse_page(page) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("translate: skipping {}: {e}", page.display());
                continue;
            }
        };
        let en = page_source(&en_fm, &en_body);
        if en.body.is_empty() && en.extra.is_empty() {
            continue;
        }
        let hash = source_hash(&en);
        for lang in &langs {
            let path = sibling_path(page, lang);
            if !path.exists() {
                continue; // missing: translate creates it
            }
            let (fm, body) = match parse_page(&path) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!("translate: skipping {}: {e}", path.display());
                    continue;
                }
            };
            visit(&Sibling { path: &path, en: &en, hash: &hash, fm, body })?;
        }
    }
    Ok(())
}

/// The sibling's text at every path of the English source, or an error naming
/// the first `[extra]` path it lacks.
fn sibling_translation(
    en: &Translatable,
    sib_fm: &toml::Value,
    sib_body: &str,
) -> Result<Translatable> {
    let empty = toml::Value::Table(toml::value::Table::new());
    let sib_extra = sib_fm.get("extra").unwrap_or(&empty);
    let extra = en
        .extra
        .iter()
        .map(|(path, _)| {
            get_leaf(sib_extra, path)
                .map(|text| (path.clone(), text.to_string()))
                .ok_or_else(|| anyhow!("missing extra.{path}"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Translatable {
        title: get_str(sib_fm, "title"),
        description: get_str(sib_fm, "description"),
        body: sib_body.trim().to_string(),
        extra,
    })
}

fn stamp_hash(fm: &mut toml::Value, hash: &str) {
    if let Some(table) = fm.as_table_mut() {
        let extra =
            table.entry("extra").or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
        if let Some(et) = extra.as_table_mut() {
            et.insert("source_hash".into(), toml::Value::String(hash.to_string()));
        }
    }
}

// ---- helpers ----

fn read_langs(config_file: &Path) -> Result<(String, Vec<String>)> {
    let text = fs::read_to_string(config_file).map_err(|e| anyhow!("read config: {e}"))?;
    let cfg: toml::Value = toml::from_str(&text).map_err(|e| anyhow!("parse config: {e}"))?;
    let default = cfg.get("default_language").and_then(|v| v.as_str()).unwrap_or("en").to_string();
    let mut langs: Vec<String> = cfg
        .get("languages")
        .and_then(|v| v.as_table())
        .map(|t| t.keys().filter(|k| *k != &default).cloned().collect())
        .unwrap_or_default();
    langs.sort();
    Ok((default, langs))
}

fn walk_md(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let rd = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(anyhow!("read dir {}: {e}", dir.display())),
    };
    for entry in rd {
        let path = entry?.path();
        if path.is_dir() {
            walk_md(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            out.push(path);
        }
    }
    Ok(())
}

/// True if `file_name` (`index.md`, `_index.md`, `index.es.md`, …) is a
/// default-language page or section, not a per-language translation.
/// Sections count: a homepage or section landing keeps its copy in `_index.md`.
fn is_default_page(file_name: &str, lang_set: &HashSet<&str>) -> bool {
    let Some(stem) = file_name.strip_suffix(".md") else {
        return false;
    };
    !lang_set.iter().any(|l| stem.ends_with(&format!(".{l}")))
}

/// The translatable fields of a default-language page.
fn page_source(fm: &toml::Value, body: &str) -> Translatable {
    Translatable {
        title: get_str(fm, "title"),
        description: get_str(fm, "description"),
        body: body.trim().to_string(),
        extra: extra_copy_leaves(fm),
    }
}

/// Split a `+++`-delimited page into (frontmatter, body markdown).
fn parse_page(path: &Path) -> Result<(toml::Value, String)> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let first = lines.next().unwrap_or("");
    if first.trim() != FM_DELIM {
        bail!(
            "{}: expected `{}` frontmatter (TOML). `zola translate` handles `+++` pages only.",
            path.display(),
            FM_DELIM
        );
    }
    let mut fm_buf = String::new();
    let mut body_buf = String::new();
    let mut closed = false;
    for line in lines {
        if !closed {
            if line.trim() == FM_DELIM {
                closed = true;
            } else {
                fm_buf.push_str(line);
                fm_buf.push('\n');
            }
        } else {
            body_buf.push_str(line);
            body_buf.push('\n');
        }
    }
    if !closed {
        bail!("{}: frontmatter not terminated by `{}`", path.display(), FM_DELIM);
    }
    let fm: toml::Value = toml::from_str(&fm_buf)
        .map_err(|e| anyhow!("{}: frontmatter parse: {e}", path.display()))?;
    Ok((fm, body_buf))
}

fn get_str(fm: &toml::Value, key: &str) -> String {
    fm.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string()
}

fn extra_hash(fm: &toml::Value) -> Option<String> {
    Some(fm.get("extra")?.get("source_hash")?.as_str()?.to_string())
}

/// `<dir>/<stem>.<lang>.md` sibling of a default page `<dir>/<stem>.md`.
fn sibling_path(page: &Path, lang: &str) -> PathBuf {
    let stem = page.file_stem().unwrap().to_string_lossy().into_owned();
    page.with_file_name(format!("{stem}.{lang}.md"))
}

/// Write the sibling from the English front matter with the translated fields
/// and `[extra]` copy set at their paths. Top-level `[extra]` keys the English
/// page does not have (a locale's own `noindex`, say) are kept from the
/// existing sibling; everything else follows the English layout. An explicit
/// `path` gets the sibling's language prefix and `aliases` are dropped: both
/// are URLs, and a copied one makes two pages claim it and fails the build.
fn write_sibling(path: &Path, en_fm: &toml::Value, t: &Translatable, hash: &str) -> Result<()> {
    let existing_extra = parse_page(path)
        .ok()
        .and_then(|(fm, _)| fm.get("extra").and_then(|e| e.as_table().cloned()))
        .unwrap_or_default();
    let lang = sibling_lang(path)?;
    let mut fm = en_fm.clone();
    if let Some(table) = fm.as_table_mut() {
        table.remove("aliases");
        if let Some(en_path) = table.get("path").and_then(|p| p.as_str()) {
            let localized = localized_path(en_path, &lang);
            table.insert("path".into(), toml::Value::String(localized));
        }
        if !t.title.is_empty() || table.contains_key("title") {
            table.insert("title".into(), toml::Value::String(t.title.clone()));
        }
        if !t.description.is_empty() || table.contains_key("description") {
            table.insert("description".into(), toml::Value::String(t.description.clone()));
        }
        let extra =
            table.entry("extra").or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
        for (leaf, text) in &t.extra {
            if !set_leaf(extra, leaf, text) {
                bail!("{}: extra.{leaf} has no string to translate into", path.display());
            }
        }
        if let Some(et) = extra.as_table_mut() {
            for (key, value) in existing_extra {
                et.entry(key).or_insert(value);
            }
            et.insert("source_hash".into(), toml::Value::String(hash.to_string()));
        }
    }
    let fm_str = toml::to_string(&fm).map_err(|e| anyhow!("serialize frontmatter: {e}"))?;
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(path, format!("+++\n{fm_str}+++\n\n{}\n", t.body.trim_end()))?;
    Ok(())
}

/// `fr` from `<dir>/index.fr.md`.
fn sibling_lang(path: &Path) -> Result<String> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.rsplit_once('.'))
        .map(|(_, lang)| lang.to_string())
        .ok_or_else(|| anyhow!("{}: not a `<stem>.<lang>.md` sibling", path.display()))
}

/// `engineering/post` → `fr/engineering/post`, keeping a leading `/`.
fn localized_path(en_path: &str, lang: &str) -> String {
    let rest = en_path.trim_start_matches('/');
    if en_path.starts_with('/') { format!("/{lang}/{rest}") } else { format!("{lang}/{rest}") }
}

fn take200(s: &str) -> String {
    s.chars().take(200).collect()
}
fn take160(s: &str) -> String {
    s.chars().take(160).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

    /// Unique temp dir per test (no tempfile dep): lives under std temp.
    struct Fixture {
        root: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
            let root = env::temp_dir().join(format!("zola-translate-test-{id}-{}", process_id()));
            fs::create_dir_all(&root).unwrap();
            // config.toml: default en + es, fr
            fs::write(
                root.join("config.toml"),
                "base_url = \"https://x/\"\ndefault_language = \"en\"\n[languages.es]\n[languages.fr]\n",
            )
            .unwrap();
            Fixture { root }
        }
        fn write_page(&self, rel: &str, fm: &str, body: &str) {
            let p = self.root.join(rel);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, format!("+++\n{fm}+++\n\n{body}")).unwrap();
        }
        fn page_body(&self, rel: &str) -> String {
            fs::read_to_string(self.root.join(rel)).unwrap()
        }
        fn config(&self) -> PathBuf {
            self.root.join("config.toml")
        }
        fn root(&self) -> &Path {
            &self.root
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[cfg(not(target_pointer_width = "32"))]
    fn process_id() -> usize {
        std::process::id() as usize
    }
    #[cfg(target_pointer_width = "32")]
    fn process_id() -> usize {
        std::process::id()
    }

    /// Mock that prepends "[lang]" and counts calls via interior mutability.
    // ── TranslateApiClient wire format ────────────────────────────────

    fn t(title: &str, description: &str, body: &str) -> Translatable {
        Translatable {
            title: title.into(),
            description: description.into(),
            body: body.into(),
            ..Translatable::default()
        }
    }

    #[test]
    fn request_carries_texts_languages_and_omits_empty_glossary() {
        let payload = build_batch_request(&["T", "D", "B"], "ko", &[]);
        assert_eq!(payload["texts"], json!(["T", "D", "B"]));
        assert_eq!(payload["target_language"], "ko");
        assert_eq!(payload["source_language"], "en");
        // Empty glossary ⇒ field omitted, body byte-identical to pre-#23-review
        // shape (an empty array is the server default, curriculo-ai #1023).
        assert!(payload.get("preserve_terms").is_none());
    }

    #[test]
    fn request_carries_preserve_terms_when_supplied() {
        // #23 review / curriculo-ai #1023: the caller's untranslatables ride the
        // same request as `texts`, so the endpoint can mask them out of the
        // engine and restore them verbatim.
        let terms = vec!["Acme Widgets".to_string(), "Zephyr Analytics".to_string()];
        let payload = build_batch_request(&["T", "D", "B"], "ko", &terms);
        assert_eq!(payload["preserve_terms"], json!(terms));
        assert_eq!(payload["target_language"], "ko");
    }

    #[test]
    fn body_only_chunk_sends_only_the_body() {
        // Continuation chunks carry no title/description (empty fields are
        // never sent); those empty strings would spend an engine call on
        // nothing.
        let payload = build_batch_request(&["B"], "ja", &[]);
        assert_eq!(payload["texts"], json!(["B"]));
    }

    #[test]
    fn response_pairs_map_back_in_input_order() {
        let out = parse_batch_translate_response(r#"{"translations":["本文"]}"#, 1).unwrap();
        assert_eq!(out, vec![("本文".to_string(), true)]);
    }

    #[test]
    fn short_response_is_an_error_not_a_silent_shift() {
        // Two translations for three texts would otherwise shift every later
        // page's fields onto the wrong slot and write a plausible-looking,
        // wrong page.
        let err = parse_batch_translate_response(r#"{"translations":["Titre","Description"]}"#, 3)
            .unwrap_err();
        assert!(format!("{err}").contains("2 translation(s) for 3"));
    }

    #[test]
    fn long_body_is_chunked_by_translate_fields_impl() {
        // Regression: chunking lives inside each client path, so a client that
        // skips it sends a whole pillar page as one request. Counts the calls
        // translate_fields_impl makes for an over-cap body.
        let body = (0..40)
            .map(|i| format!("## Heading {i}\n\n{}", "word ".repeat(120)))
            .collect::<Vec<_>>()
            .join("\n\n");
        assert!(body.len() > BODY_CHUNK_CHARS, "fixture must exceed the cap");
        let mut calls = 0usize;
        let out = translate_fields_impl(
            |f, _body_only| {
                calls += 1;
                Ok(f.clone())
            },
            &t("T", "D", &body),
        )
        .unwrap();
        assert!(calls > 1, "expected the body to be chunked, got {calls} call(s)");
        assert_eq!(out.title, "T");
    }

    #[test]
    fn missing_translations_array_is_an_error() {
        let err = parse_batch_translate_response(r#"{"oops":true}"#, 1).unwrap_err();
        assert!(format!("{err}").contains("translations"));
    }

    #[test]
    fn ok_false_is_a_hard_failure_not_source_passthrough() {
        // #1003: HTTP 200 + ok:false means `translations` carries the ORIGINAL
        // source text. Accepting it would write English into `<slug>.ko.md`
        // stamped fresh (source_hash set) — exactly the bug this guards.
        let err = parse_batch_translate_response(r#"{"ok":false,"translations":["T","D","B"]}"#, 3)
            .unwrap_err();
        assert!(format!("{err}").contains("ok:false"), "got: {err}");
    }

    #[test]
    fn ok_array_false_flag_is_soft_and_per_input() {
        // Bulk semantics: a false flag marks ONLY that input as untranslated.
        // The driver fails the page owning it; the rest of the batch writes.
        let out =
            parse_batch_translate_response(r#"{"ok":[true,false],"translations":["T","B"]}"#, 2)
                .unwrap();
        assert_eq!(out[0], ("T".to_string(), true));
        assert_eq!(out[1], ("B".to_string(), false));
    }

    #[test]
    fn ok_array_shorter_than_sent_is_malformed() {
        // Finding #3: an all-true but short ok array used to pass; the missing
        // flags must reject the envelope rather than silently covering unsent texts.
        let err =
            parse_batch_translate_response(r#"{"ok":[true],"translations":["T","D","B"]}"#, 3)
                .unwrap_err();
        assert!(
            format!("{err}").contains("malformed") || format!("{err}").contains("length"),
            "got: {err}"
        );
    }

    #[test]
    fn ok_array_with_a_non_bool_entry_is_malformed() {
        let err = parse_batch_translate_response(r#"{"ok":[true,1],"translations":["T","B"]}"#, 2)
            .unwrap_err();
        assert!(format!("{err}").contains("malformed"), "got: {err}");
    }

    #[test]
    fn ok_all_true_or_absent_still_passes() {
        // Older deployments send no `ok` at all; newer ones send all-true.
        // Both must keep working — guard against over-tightening the #1003 fix.
        for body in [r#"{"ok":[true],"translations":["本文"]}"#, r#"{"translations":["本文"]}"#]
        {
            let out = parse_batch_translate_response(body, 1).unwrap();
            assert_eq!(out, vec![("本文".to_string(), true)]);
        }
    }

    #[test]
    fn mirrored_error_code_1003_is_a_hard_failure() {
        for body in [
            r#"{"code":1003,"translations":["T"]}"#,
            r#"{"status":1003,"translations":["T"]}"#,
            r#"{"error_code":"1003","translations":["T"]}"#,
        ] {
            let err = parse_batch_translate_response(body, 1).unwrap_err();
            assert!(format!("{err}").contains("1003"), "body {body} -> {err}");
        }
    }

    #[test]
    fn numeric_negative_error_code_is_a_hard_failure() {
        // Finding #4: as_u64/as_str both miss JSON numbers like -1.
        let err =
            parse_batch_translate_response(r#"{"code":-1,"translations":["T"]}"#, 1).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("-1") || msg.contains("error code"), "got: {err}");
    }

    #[test]
    fn non_string_translation_is_an_error_not_a_blank() {
        // Batch-level: one bad entry sinks the whole request — there is nothing
        // sane to write for that slot's page, and a partial write would need a
        // partial-response contract the endpoint does not have.
        for body in [
            r#"{"translations":[null,"D"]}"#,
            r#"{"translations":[["T"],"D"]}"#,
            r#"{"translations":[{"v":"T"},"D"]}"#,
        ] {
            let err = parse_batch_translate_response(body, 2).unwrap_err();
            assert!(
                format!("{err}").contains("translations[0] is not a string"),
                "body {body} -> {err}"
            );
        }
    }

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn translate_timeout_env_default_floor_and_garbage() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::remove_var("TRANSLATE_TIMEOUT") };
        assert_eq!(timeout_from_env().unwrap(), Duration::from_secs(300));
        unsafe { env::set_var("TRANSLATE_TIMEOUT", "900") };
        assert_eq!(timeout_from_env().unwrap(), Duration::from_secs(900));
        unsafe { env::set_var("TRANSLATE_TIMEOUT", "3") };
        assert_eq!(timeout_from_env().unwrap(), Duration::from_secs(30));
        unsafe { env::set_var("TRANSLATE_TIMEOUT", "  ") };
        assert_eq!(timeout_from_env().unwrap(), Duration::from_secs(300));
        unsafe { env::set_var("TRANSLATE_TIMEOUT", "soon") };
        assert!(timeout_from_env().is_err());
        unsafe { env::remove_var("TRANSLATE_TIMEOUT") };
    }

    #[test]
    fn preserve_terms_env_parses_trims_and_drops_blanks() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { env::remove_var("TRANSLATE_PRESERVE_TERMS") };
        assert!(preserve_terms_from_env().is_empty());
        unsafe { env::set_var("TRANSLATE_PRESERVE_TERMS", " Acme Widgets ,, Zephyr ") };
        assert_eq!(
            preserve_terms_from_env(),
            vec!["Acme Widgets".to_string(), "Zephyr".to_string()]
        );
        unsafe { env::set_var("TRANSLATE_PRESERVE_TERMS", " , , ") };
        assert!(preserve_terms_from_env().is_empty());
        unsafe { env::remove_var("TRANSLATE_PRESERVE_TERMS") };
    }

    /// Minimal keep-alive HTTP/1.1 server for exercising [`TranslateApiClient`]
    /// over a real socket. Serves one JSON body per request, built from the
    /// request's `texts` length; counts sockets and requests so tests can
    /// assert on connection reuse.
    struct TinyServer {
        url: String,
        conns: Arc<AtomicUsize>,
        reqs: Arc<AtomicUsize>,
    }

    impl TinyServer {
        fn start(make_body: fn(usize) -> String) -> Self {
            use std::io::Write;
            use std::net::TcpListener;
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let conns = Arc::new(AtomicUsize::new(0));
            let reqs = Arc::new(AtomicUsize::new(0));
            let (c2, r2) = (conns.clone(), reqs.clone());
            std::thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let mut stream = stream;
                    c2.fetch_add(1, Ordering::SeqCst);
                    while let Some(body) = read_request_body(&mut stream) {
                        let n = serde_json::from_str::<Value>(&body)
                            .ok()
                            .and_then(|v| v["texts"].as_array().map(|a| a.len()))
                            .unwrap_or(0);
                        r2.fetch_add(1, Ordering::SeqCst);
                        let out = make_body(n);
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{}",
                            out.len(),
                            out
                        );
                        if stream.write_all(resp.as_bytes()).is_err() {
                            break;
                        }
                    }
                }
            });
            TinyServer { url: format!("http://{addr}/translate"), conns, reqs }
        }

        /// Echo `texts` back as `translations`. Packed `XHTML0003X` tokens
        /// survive; the client must unpack them. Optional `strip_real_tags`
        /// mimics raw NLLB dropping `<p>` / `<table>` — after pack there are
        /// none, so unpack still restores structure.
        fn echo(strip_real_tags: bool) -> Self {
            use std::io::Write;
            use std::net::TcpListener;
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let conns = Arc::new(AtomicUsize::new(0));
            let reqs = Arc::new(AtomicUsize::new(0));
            let (c2, r2) = (conns.clone(), reqs.clone());
            std::thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let mut stream = stream;
                    c2.fetch_add(1, Ordering::SeqCst);
                    while let Some(body) = read_request_body(&mut stream) {
                        r2.fetch_add(1, Ordering::SeqCst);
                        let texts: Vec<String> = serde_json::from_str::<Value>(&body)
                            .ok()
                            .and_then(|v| v["texts"].as_array().cloned())
                            .unwrap_or_default()
                            .into_iter()
                            .filter_map(|t| t.as_str().map(str::to_string))
                            .map(|t| {
                                if strip_real_tags {
                                    TAG_STRIP.replace_all(&t, "").into_owned()
                                } else {
                                    t
                                }
                            })
                            .collect();
                        let out = json!({ "ok": true, "translations": texts }).to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{}",
                            out.len(),
                            out
                        );
                        if stream.write_all(resp.as_bytes()).is_err() {
                            break;
                        }
                    }
                }
            });
            TinyServer { url: format!("http://{addr}/translate"), conns, reqs }
        }

        fn echo_map(map: fn(&str) -> String) -> Self {
            use std::io::Write;
            use std::net::TcpListener;
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let conns = Arc::new(AtomicUsize::new(0));
            let reqs = Arc::new(AtomicUsize::new(0));
            let (c2, r2) = (conns.clone(), reqs.clone());
            std::thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let mut stream = stream;
                    c2.fetch_add(1, Ordering::SeqCst);
                    while let Some(body) = read_request_body(&mut stream) {
                        r2.fetch_add(1, Ordering::SeqCst);
                        let texts: Vec<String> = serde_json::from_str::<Value>(&body)
                            .ok()
                            .and_then(|v| v["texts"].as_array().cloned())
                            .unwrap_or_default()
                            .into_iter()
                            .filter_map(|t| t.as_str().map(str::to_string))
                            .map(|t| map(&t))
                            .collect();
                        let out = json!({ "ok": true, "translations": texts }).to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{}",
                            out.len(),
                            out
                        );
                        if stream.write_all(resp.as_bytes()).is_err() {
                            break;
                        }
                    }
                }
            });
            TinyServer { url: format!("http://{addr}/translate"), conns, reqs }
        }
    }

    static TAG_STRIP: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?>").unwrap()
    });

    /// Read one request (headers + Content-Length body) off the stream; None on EOF.
    fn read_request_body(stream: &mut std::net::TcpStream) -> Option<String> {
        use std::io::Read;
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 8192];
        let hdr_end = loop {
            if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break p + 4;
            }
            let n = stream.read(&mut chunk).ok()?;
            if n == 0 {
                return None;
            }
            buf.extend_from_slice(&chunk[..n]);
        };
        let headers = String::from_utf8_lossy(&buf[..hdr_end]).to_ascii_uppercase();
        let cl: usize = headers
            .lines()
            .find_map(|l| l.strip_prefix("CONTENT-LENGTH:"))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0);
        while buf.len() < hdr_end + cl {
            let n = stream.read(&mut chunk).ok()?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
        }
        Some(String::from_utf8_lossy(&buf[hdr_end..]).into_owned())
    }

    #[test]
    fn ok_false_over_http_writes_no_sibling() {
        // #1003 end-to-end: HTTP 200 + ok:false + source-text echo must NOT
        // produce `.es.md`/`.fr.md` stamped as fresh translations.
        let fx = Fixture::new();
        fx.write_page("content/post/index.md", en_page_fm(), "Body one.\n");
        let server = TinyServer::start(|n| {
            json!({"ok": false, "translations": vec!["passthrough"; n]}).to_string()
        });
        let client = TranslateApiClient::new(&server.url, Duration::from_secs(30), vec![]).unwrap();
        let res = translate_with_endpoint(fx.root(), &fx.config(), None, false, &client);
        assert!(res.is_err(), "ok:false must surface as a failure");
        assert!(!fx.root().join("content/post/index.es.md").exists(), "must not write es");
        assert!(!fx.root().join("content/post/index.fr.md").exists(), "must not write fr");
        assert!(server.reqs.load(Ordering::SeqCst) >= 2, "both langs attempted");
    }

    #[test]
    fn mirrored_1003_over_http_writes_no_sibling() {
        let fx = Fixture::new();
        fx.write_page("content/post/index.md", en_page_fm(), "Body one.\n");
        let server = TinyServer::start(|n| {
            json!({"code": 1003, "translations": vec!["passthrough"; n]}).to_string()
        });
        let client = TranslateApiClient::new(&server.url, Duration::from_secs(30), vec![]).unwrap();
        let res = translate_with_endpoint(fx.root(), &fx.config(), None, false, &client);
        assert!(res.is_err(), "mirrored error code 1003 must surface as a failure");
        assert!(!fx.root().join("content/post/index.es.md").exists(), "must not write es");
        assert!(!fx.root().join("content/post/index.fr.md").exists(), "must not write fr");
    }

    #[test]
    fn http_client_is_built_once_and_reused_across_batches() {
        // An over-16 KiB body spans several batches, each its own request; a
        // client built per request would open a TCP connection per batch. The
        // pooled client built once per run must keep them on ONE connection.
        let fx = Fixture::new();
        // 24 × ~1 KB sections ≈ 24 KB body → 4+ chunks → over the 16 KiB
        // batch cap → 2+ batches per language.
        let body = (0..24)
            .map(|i| format!("## H{i}\n\n{}", "word ".repeat(200)))
            .collect::<Vec<_>>()
            .join("\n\n");
        fx.write_page("content/post/index.md", en_page_fm(), &body);
        let server = TinyServer::start(|n| {
            json!({
                "ok": vec![true; n],
                "translations": (0..n).map(|i| format!("Curriculo {i}")).collect::<Vec<_>>(),
            })
            .to_string()
        });
        let client = TranslateApiClient::new(&server.url, Duration::from_secs(30), vec![]).unwrap();
        translate_with_endpoint(fx.root(), &fx.config(), None, false, &client).unwrap();
        let reqs = server.reqs.load(Ordering::SeqCst);
        let conns = server.conns.load(Ordering::SeqCst);
        assert!(reqs > 2, "expected several batched requests, got {reqs}");
        assert_eq!(conns, 1, "all requests must share one pooled connection, got {conns}");
        assert!(fx.root().join("content/post/index.es.md").exists());
    }

    /// Rewrite the fixture config to a single non-default language (es).
    fn one_lang_config(fx: &Fixture) {
        fs::write(
            fx.root().join("config.toml"),
            "base_url = \"https://x/\"\ndefault_language = \"en\"\n[languages.es]\n",
        )
        .unwrap();
    }

    /// Recording mock for the endpoint driver: remembers every call's texts
    /// and lang, and answers from a scripted list of outcomes drained in
    /// order. When the script runs dry it echoes the input with ok=true
    /// (echo keeps glossary tokens, so happy-path runs pass INV-5).
    struct MockBatch {
        calls: Mutex<Vec<(Vec<String>, String)>>,
        script: Mutex<Vec<Result<Vec<(String, bool)>, String>>>,
    }

    impl MockBatch {
        fn new() -> Self {
            MockBatch { calls: Mutex::new(Vec::new()), script: Mutex::new(Vec::new()) }
        }

        /// Script one successful response: (translation, ok) per text.
        fn push_ok_flags(&self, pairs: Vec<(&str, bool)>) {
            self.script
                .lock()
                .unwrap()
                .push(Ok(pairs.into_iter().map(|(s, ok)| (s.to_string(), ok)).collect()));
        }

        /// Script one whole-batch failure (envelope/HTTP error).
        fn push_err(&self, msg: &str) {
            self.script.lock().unwrap().push(Err(msg.to_string()));
        }

        fn calls(&self) -> Vec<(Vec<String>, String)> {
            self.calls.lock().unwrap().clone()
        }

        fn texts_of(&self, i: usize) -> Vec<String> {
            self.calls()[i].0.clone()
        }

        fn langs(&self) -> Vec<String> {
            self.calls().into_iter().map(|(_, l)| l).collect()
        }
    }

    impl BatchTranslateClient for MockBatch {
        fn translate_texts(&self, texts: &[&str], lang: &str) -> Result<Vec<(String, bool)>> {
            self.calls
                .lock()
                .unwrap()
                .push((texts.iter().map(|s| s.to_string()).collect(), lang.to_string()));
            let mut script = self.script.lock().unwrap();
            if script.is_empty() {
                return Ok(texts.iter().map(|t| (t.to_string(), true)).collect());
            }
            script.remove(0).map_err(|e| anyhow!("{e}"))
        }
    }

    #[test]
    fn page_cap_splits_five_small_pages_into_two_requests() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        for i in 1..=5 {
            fx.write_page(
                &format!("content/p{i}/index.md"),
                &format!("title = \"P{i}\"\ndescription = \"D{i}\"\n"),
                &format!("Body {i}.\n"),
            );
        }
        let mock = MockBatch::new();
        translate_with_endpoint(fx.root(), &fx.config(), None, false, &mock).unwrap();
        let calls = mock.calls();
        assert_eq!(calls.len(), 2, "5 pages under a 4-page cap ⇒ 2 requests");
        assert_eq!(calls[0].1, "es");
        // First request: pages 1-4 flattened in field order (title/desc/body).
        assert_eq!(
            calls[0].0,
            [
                "P1", "D1", "Body 1.", "P2", "D2", "Body 2.", "P3", "D3", "Body 3.", "P4", "D4",
                "Body 4."
            ]
        );
        // Second request: just page 5.
        assert_eq!(calls[1].0, ["P5", "D5", "Body 5."]);
        for i in 1..=5 {
            assert!(fx.root().join(format!("content/p{i}/index.es.md")).exists(), "page {i}");
        }
    }

    #[test]
    fn byte_cap_keeps_every_request_under_16kib() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        // 8188-byte bodies (one chunk each): a page's texts weigh 8192 bytes,
        // so pages 1-2 land at exactly 16384 — the cap, allowed — and page 3's
        // title overflows it. ⇒ clean 2+1 page split, 6+3 texts.
        for i in 1..=3 {
            fx.write_page(
                &format!("content/p{i}/index.md"),
                &format!("title = \"P{i}\"\ndescription = \"D{i}\"\n"),
                &format!("Curriculo {}\n", "x".repeat(8178)),
            );
        }
        let mock = MockBatch::new();
        translate_with_endpoint(fx.root(), &fx.config(), None, false, &mock).unwrap();
        let calls = mock.calls();
        assert_eq!(calls.len(), 2, "expected the 2+1 page split");
        for (texts, lang) in &calls {
            let bytes: usize = texts.iter().map(|t| t.len()).sum();
            assert!(bytes <= BATCH_MAX_TEXT_BYTES, "{lang} request of {bytes} bytes over cap");
        }
        // Batch 1 sits exactly at the cap: two whole pages (title/desc/body
        // each), 16384 bytes.
        assert_eq!(calls[0].0.len(), 6, "first request carries pages 1-2 (3 texts each)");
        assert_eq!(calls[0].0[0], "P1");
        assert_eq!(calls[0].0[3], "P2");
        assert_eq!(calls[0].0[4], "D2");
        let bytes0: usize = calls[0].0.iter().map(|t| t.len()).sum();
        assert_eq!(bytes0, BATCH_MAX_TEXT_BYTES);
        assert_eq!(calls[1].0.first().map(String::as_str), Some("P3"));
        assert_eq!(calls[1].0.len(), 3, "second request carries page 3");
    }

    #[test]
    fn oversized_page_spans_batches_and_reassembles_in_order() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        // One ~24 KB body: its chunks alone exceed the 16 KiB request cap, so
        // the page spans batches. The echo mock makes the written body equal to
        // the chunk join — order and separators both checked.
        let body = (0..24)
            .map(|i| format!("## H{i}\n\n{}", "x".repeat(1000)))
            .collect::<Vec<_>>()
            .join("\n\n");
        fx.write_page("content/post/index.md", "title = \"T\"\ndescription = \"D\"\n", &body);
        let mock = MockBatch::new();
        translate_with_endpoint(fx.root(), &fx.config(), None, false, &mock).unwrap();
        let calls = mock.calls();
        assert!(calls.len() >= 2, "over-cap page must span batches, got {}", calls.len());
        let sibling = fx.root().join("content/post/index.es.md");
        let (fm, written) = parse_page(&sibling).unwrap();
        let expected = chunk_body(&body).join("\n\n");
        assert_eq!(written.trim(), expected, "chunks must rejoin in request order");
        // …and the hash stamp matches the en fields, so the pair is now fresh.
        let en = Translatable {
            title: "T".into(),
            description: "D".into(),
            body: body.trim().into(),
            ..Translatable::default()
        };
        assert_eq!(extra_hash(&fm).as_deref(), Some(source_hash(&en).as_str()));
    }

    #[test]
    fn languages_drain_one_at_a_time_es_before_fr() {
        let fx = Fixture::new(); // default fixture config: es + fr
        for i in 1..=5 {
            fx.write_page(
                &format!("content/p{i}/index.md"),
                &format!("title = \"P{i}\"\n"),
                &format!("Body {i}.\n"),
            );
        }
        let mock = MockBatch::new();
        translate_with_endpoint(fx.root(), &fx.config(), None, false, &mock).unwrap();
        // 5 pages > the 4-page cap ⇒ 2 requests per language; all of es first.
        assert_eq!(mock.langs(), vec!["es", "es", "fr", "fr"]);
    }

    #[test]
    fn flattening_order_and_positional_reassembly() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        fx.write_page("content/post/index.md", "title = \"T\"\ndescription = \"D\"\n", "Body.\n");
        let mock = MockBatch::new();
        mock.push_ok_flags(vec![("t-es", true), ("d-es", true), ("b-es", true)]);
        translate_with_endpoint(fx.root(), &fx.config(), None, false, &mock).unwrap();
        // One call: title → description → body, in field order.
        assert_eq!(mock.texts_of(0), ["T", "D", "Body."]);
        let (fm, body) = parse_page(&fx.root().join("content/post/index.es.md")).unwrap();
        assert_eq!(fm.get("title").and_then(|v| v.as_str()), Some("t-es"));
        assert_eq!(fm.get("description").and_then(|v| v.as_str()), Some("d-es"));
        assert_eq!(body.trim(), "b-es");
    }

    #[test]
    fn one_false_ok_flag_fails_only_its_page() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        for i in 1..=4 {
            fx.write_page(
                &format!("content/p{i}/index.md"),
                &format!("title = \"P{i}\"\ndescription = \"D{i}\"\n"),
                &format!("Body {i}.\n"),
            );
        }
        let mock = MockBatch::new();
        // All four pages fit one request (12 texts); page 2's three slots come
        // back flagged untranslated.
        mock.push_ok_flags(vec![
            ("x0", true),
            ("x1", true),
            ("x2", true),
            ("x3", false),
            ("x4", false),
            ("x5", false),
            ("x6", true),
            ("x7", true),
            ("x8", true),
            ("x9", true),
            ("x10", true),
            ("x11", true),
        ]);
        let res = translate_with_endpoint(fx.root(), &fx.config(), None, false, &mock);
        let err = res.unwrap_err().to_string();
        assert!(err.contains("failure"), "got: {err}");
        // Page 2: no file (which also means no source_hash stamp). Pages
        // 1, 3, 4: written and stamped with the hash of their en fields.
        assert!(
            !fx.root().join("content/p2/index.es.md").exists(),
            "flagged page must not be written"
        );
        for i in [1, 3, 4] {
            let (fm, _) = parse_page(&fx.root().join(format!("content/p{i}/index.es.md"))).unwrap();
            let en = Translatable {
                title: format!("P{i}"),
                description: format!("D{i}"),
                body: format!("Body {i}."),
                ..Translatable::default()
            };
            assert_eq!(extra_hash(&fm).as_deref(), Some(source_hash(&en).as_str()), "page {i}");
        }
        assert_eq!(mock.calls().len(), 1, "all four pages fit one request");
    }

    #[test]
    fn envelope_error_fails_the_whole_batch_but_not_the_run() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        // 5459-byte bodies (one chunk each): a page weighs 5461 bytes, so
        // three pages land at 16383 and page 4's title overflows ⇒ batch 1 =
        // pages 1-3 exactly, batch 2 = pages 4-5.
        for i in 1..=5 {
            fx.write_page(
                &format!("content/p{i}/index.md"),
                &format!("title = \"P{i}\"\n"),
                &format!("Curriculo {}\n", "y".repeat(5449)),
            );
        }
        let mock = MockBatch::new();
        mock.push_err("translate endpoint reported ok:false — source text returned untranslated");
        // Batch 2 falls through to the echo default and writes pages 4-5.
        let res = translate_with_endpoint(fx.root(), &fx.config(), None, false, &mock);
        assert!(res.is_err(), "3 failed jobs must fail the run");
        for i in 1..=3 {
            assert!(
                !fx.root().join(format!("content/p{i}/index.es.md")).exists(),
                "page {i} must not be written"
            );
        }
        for i in 4..=5 {
            assert!(
                fx.root().join(format!("content/p{i}/index.es.md")).exists(),
                "page {i} should still be written"
            );
        }
        assert_eq!(mock.calls().len(), 2, "second batch must still be attempted");
    }

    #[test]
    fn max_cap_truncates_page_major_and_resumes() {
        let fx = Fixture::new();
        for i in 1..=2 {
            fx.write_page(
                &format!("content/p{i}/index.md"),
                &format!("title = \"P{i}\"\n"),
                &format!("Body {i}.\n"),
            );
        }
        // Jobs are page-major (p1-es, p1-fr, p2-es, p2-fr); --max 1 keeps the
        // first only — the same cut translate_with makes per (page, lang).
        let m1 = MockBatch::new();
        translate_with_endpoint(fx.root(), &fx.config(), Some(1), false, &m1).unwrap();
        assert_eq!(m1.calls().len(), 1);
        assert_eq!(m1.calls()[0].1, "es");
        assert_eq!(m1.texts_of(0), ["P1", "Body 1."]);
        assert!(fx.root().join("content/p1/index.es.md").exists());
        for p in ["content/p1/index.fr.md", "content/p2/index.es.md", "content/p2/index.fr.md"] {
            assert!(!fx.root().join(p).exists(), "{p} must stay untouched");
        }

        // Resume: exactly the 3 remaining jobs translate; es drained before fr.
        let m2 = MockBatch::new();
        translate_with_endpoint(fx.root(), &fx.config(), None, false, &m2).unwrap();
        assert_eq!(m2.langs(), vec!["es", "fr"], "es (only p2 left) before fr (p1, p2)");
        for p in ["content/p1/index.fr.md", "content/p2/index.es.md", "content/p2/index.fr.md"] {
            assert!(fx.root().join(p).exists(), "{p} must be written on resume");
        }
    }

    #[test]
    fn endpoint_dry_run_makes_no_calls_and_writes_nothing() {
        let fx = Fixture::new();
        fx.write_page("content/post/index.md", en_page_fm(), "Body one.\n");
        let mock = MockBatch::new();
        translate_with_endpoint(fx.root(), &fx.config(), None, true, &mock).unwrap();
        assert!(mock.calls().is_empty(), "dry-run must not hit the endpoint");
        assert!(!fx.root().join("content/post/index.es.md").exists());
        assert!(!fx.root().join("content/post/index.fr.md").exists());
    }

    #[test]
    fn fresh_endpoint_sibling_skips_the_call() {
        let fx = Fixture::new();
        fx.write_page("content/post/index.md", en_page_fm(), "Body one.\n");
        // Pre-stamp a fresh es sibling (correct source_hash) ⇒ not a job.
        let en = Translatable {
            title: "Hello Curriculo".into(),
            description: "A desc".into(),
            body: "Body one.".into(),
            ..Translatable::default()
        };
        let (fm, _) = parse_page(&fx.root().join("content/post/index.md")).unwrap();
        write_sibling(&fx.root().join("content/post/index.es.md"), &fm, &en, &source_hash(&en))
            .unwrap();
        let mock = MockBatch::new();
        translate_with_endpoint(fx.root(), &fx.config(), None, false, &mock).unwrap();
        let calls = mock.calls();
        assert_eq!(calls.len(), 1, "only fr remains");
        assert_eq!(calls[0].1, "fr");
    }

    struct EchoClient {
        calls: Cell<usize>,
    }
    impl LlmClient for EchoClient {
        fn translate(&self, f: &Translatable, lang: &str, _key: &str) -> Result<Translatable> {
            self.calls.set(self.calls.get() + 1);
            Ok(Translatable {
                title: format!("[{lang}] {}", f.title),
                description: format!("[{lang}] {}", f.description),
                body: format!("[{lang}] {}", f.body),
                extra: f.extra.iter().map(|(p, s)| (p.clone(), format!("[{lang}] {s}"))).collect(),
            })
        }
    }

    /// Mock that drops the "Curriculo" token → glossary must reject.
    struct DroppingClient;
    impl LlmClient for DroppingClient {
        fn translate(&self, f: &Translatable, _lang: &str, _key: &str) -> Result<Translatable> {
            Ok(Translatable {
                title: f.title.replace("Curriculo", "Brand"),
                description: f.description.replace("Curriculo", "Brand"),
                body: f.body.replace("Curriculo", "Brand"),
                extra: f
                    .extra
                    .iter()
                    .map(|(p, s)| (p.clone(), s.replace("Curriculo", "Brand")))
                    .collect(),
            })
        }
    }

    /// Mock that always fails.
    struct FailClient;
    impl LlmClient for FailClient {
        fn translate(&self, _: &Translatable, _: &str, _: &str) -> Result<Translatable> {
            bail!("boom")
        }
    }

    fn en_page_fm() -> &'static str {
        "title = \"Hello Curriculo\"\ndescription = \"A desc\"\n[date]\ntaxonomies = {tags = [\"x\"]}\n"
    }

    #[test]
    fn hash_gate_fresh_skip_and_stale_regen() {
        let fx = Fixture::new();
        fx.write_page("content/post/index.md", en_page_fm(), "Body one.\n");

        let c = EchoClient { calls: Cell::new(0) };
        // run 1: both langs stale → 2 calls, both written
        translate_with(fx.root(), &fx.config(), None, false, "test-key", &c).unwrap();
        assert_eq!(c.calls.get(), 2);
        let es = fx.page_body("content/post/index.es.md");
        assert!(es.contains("[es] Hello Curriculo"));
        assert!(es.contains("source_hash"));
        assert!(es.contains("Curriculo")); // brand preserved by EchoClient

        // run 2: source unchanged → both fresh → no calls
        translate_with(fx.root(), &fx.config(), None, false, "test-key", &c).unwrap();
        assert_eq!(c.calls.get(), 2, "fresh siblings must not re-translate");

        // mutate en body → both stale → 2 more calls
        fx.write_page("content/post/index.md", en_page_fm(), "Body two.\n");
        translate_with(fx.root(), &fx.config(), None, false, "test-key", &c).unwrap();
        assert_eq!(c.calls.get(), 4, "stale siblings must regenerate");
    }

    #[test]
    fn glossary_rejection_does_not_write() {
        let fx = Fixture::new();
        fx.write_page("content/post/index.md", en_page_fm(), "Curriculo body.\n");

        let res = translate_with(fx.root(), &fx.config(), None, false, "test-key", &DroppingClient);
        assert!(res.is_err(), "glossary failure must surface as non-zero exit");
        assert!(!fx.root().join("content/post/index.es.md").exists(), "file must NOT be written");
        assert!(!fx.root().join("content/post/index.fr.md").exists());
    }

    #[test]
    fn max_cap_resumes_next_run() {
        let fx = Fixture::new();
        fx.write_page("content/post/index.md", en_page_fm(), "Body.\n");

        // cap at 1 call: only one of {es, fr} translated
        let c1 = EchoClient { calls: Cell::new(0) };
        translate_with(fx.root(), &fx.config(), Some(1), false, "test-key", &c1).unwrap();
        assert_eq!(c1.calls.get(), 1);
        let one_written = fx.root().join("content/post/index.es.md").exists()
            ^ fx.root().join("content/post/index.fr.md").exists();
        assert!(one_written, "exactly one sibling written under --max 1");

        // resume: the remaining lang is still missing → translated now
        let c2 = EchoClient { calls: Cell::new(0) };
        translate_with(fx.root(), &fx.config(), None, false, "test-key", &c2).unwrap();
        assert_eq!(c2.calls.get(), 1, "only the previously-capped lang remains");
        assert!(fx.root().join("content/post/index.es.md").exists());
        assert!(fx.root().join("content/post/index.fr.md").exists());
    }

    #[test]
    fn non_zero_exit_on_failures() {
        let fx = Fixture::new();
        fx.write_page("content/post/index.md", en_page_fm(), "Body.\n");
        let res = translate_with(fx.root(), &fx.config(), None, false, "test-key", &FailClient);
        assert!(res.is_err());
        assert!(res.unwrap_err().to_string().contains("failure"));
    }

    #[test]
    fn empty_body_skipped_no_key_needed_for_dry_run() {
        let fx = Fixture::new();
        // title-only page: empty body → skipped by translate
        fx.write_page("content/stub/index.md", "title = \"Stub\"\n", "\n");
        let c = EchoClient { calls: Cell::new(0) };
        // dry_run, no key in env for this process slice: must NOT error
        translate_with(fx.root(), &fx.config(), None, true, "test-key", &c).unwrap();
        assert_eq!(c.calls.get(), 0);
        assert!(!fx.root().join("content/stub/index.es.md").exists());
    }

    #[test]
    fn source_hash_is_deterministic_and_field_scoped() {
        let a = t("t", "d", "b");
        let b = t("t", "d", "b");
        let c = t("t", "d", "B");
        assert_eq!(source_hash(&a), source_hash(&b));
        assert_ne!(source_hash(&a), source_hash(&c));
    }

    #[test]
    fn glossary_passes_when_token_preserved() {
        let en = t("Welcome to Curriculo", "", "");
        let ok = Translatable { title: "Bienvenue sur Curriculo".into(), ..en.clone() };
        let lost = Translatable { title: "Bienvenue sur Brand".into(), ..en.clone() };
        assert!(glossary_ok(&en, &ok).is_ok());
        assert!(glossary_ok(&en, &lost).is_err());
    }

    #[test]
    fn chunk_body_splits_on_h2_when_large() {
        let h2 = "## Section\n\n";
        let para = "word ".repeat(800); // ~4k chars each
        let body = format!("{h2}{para}\n{h2}{para}\n{h2}{para}");
        let chunks = chunk_body(&body);
        assert!(chunks.len() >= 2, "expected multiple chunks, got {}", chunks.len());
        let joined = chunks.join("\n\n");
        assert!(joined.contains("## Section"));
    }

    /// Mock that counts calls — chunked bodies should invoke translate >1 time.
    struct CountingClient {
        calls: Cell<usize>,
    }
    impl LlmClient for CountingClient {
        fn translate(&self, f: &Translatable, lang: &str, _key: &str) -> Result<Translatable> {
            self.calls.set(self.calls.get() + 1);
            Ok(Translatable {
                title: if f.title.is_empty() {
                    String::new()
                } else {
                    format!("[{lang}] {}", f.title)
                },
                description: if f.description.is_empty() {
                    String::new()
                } else {
                    format!("[{lang}] {}", f.description)
                },
                body: format!("[{lang}] {}", f.body),
                extra: f.extra.iter().map(|(p, s)| (p.clone(), format!("[{lang}] {s}"))).collect(),
            })
        }
    }

    #[test]
    fn large_body_uses_multiple_translate_calls() {
        let h2 = "## Part\n\n";
        let para = "Curriculo ".repeat(1200);
        let body = format!("{h2}{para}\n{h2}{para}\n{h2}{para}");
        let en = Translatable {
            title: "Big Curriculo page".into(),
            description: "desc".into(),
            body,
            ..Translatable::default()
        };
        let c = CountingClient { calls: Cell::new(0) };
        let out = translate_fields(&c, &en, "es", "k").unwrap();
        assert!(c.calls.get() > 1, "chunked body should call translate more than once");
        assert!(out.body.contains("[es]"));
        assert!(out.title.contains("Curriculo"));
    }

    // ── [extra] copy, sections, output checks ──────────────────────────

    fn extra_fm() -> &'static str {
        "title = \"Home Curriculo\"\ndescription = \"Desc\"\n\
         [extra]\ncanonical = \"https://x/\"\njsonld = '{\"@context\":\"https://schema.org\"}'\n\
         [extra.hero]\ncta = \"Start Free\"\nlede = \"Curriculo reads the work.\"\nicon = \"interface-essential-crown\"\n\
         [[extra.cards]]\nn = \"01\"\nlink = \"impact_scoring\"\ntitle = \"It reads the actual work\"\n\
         items = [\"Portfolios and repositories\", \"Case studies\"]\nshot = \"/arb/01.jpg\"\n"
    }

    fn fm_of(fx: &Fixture, rel: &str) -> toml::Value {
        parse_page(&fx.root().join(rel)).unwrap().0
    }

    #[test]
    fn extra_copy_leaves_keep_copy_and_skip_ids_routes_and_json() {
        let fm: toml::Value = toml::from_str(extra_fm()).unwrap();
        let paths: Vec<String> = extra_copy_leaves(&fm).into_iter().map(|(p, _)| p).collect();
        assert_eq!(
            paths,
            vec![
                "cards[0].items[0]",
                "cards[0].items[1]",
                "cards[0].title",
                "hero.cta",
                "hero.lede",
            ],
            "canonical, jsonld, icon, n, slug link and shot must never be sent"
        );
    }

    #[test]
    fn extra_copy_is_translated_and_written_at_its_paths() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        fx.write_page("content/home/index.md", extra_fm(), "Body Curriculo.\n");
        let c = EchoClient { calls: Cell::new(0) };
        translate_with(fx.root(), &fx.config(), None, false, "k", &c).unwrap();

        let es = fm_of(&fx, "content/home/index.es.md");
        let extra = &es["extra"];
        assert_eq!(extra["hero"]["cta"].as_str(), Some("[es] Start Free"));
        assert_eq!(extra["cards"][0]["items"][1].as_str(), Some("[es] Case studies"));
        assert_eq!(extra["cards"][0]["title"].as_str(), Some("[es] It reads the actual work"));
        // identifiers stay as the English page has them
        assert_eq!(extra["hero"]["icon"].as_str(), Some("interface-essential-crown"));
        assert_eq!(extra["cards"][0]["link"].as_str(), Some("impact_scoring"));
        assert_eq!(extra["cards"][0]["shot"].as_str(), Some("/arb/01.jpg"));
        assert!(extra["jsonld"].as_str().unwrap().starts_with('{'));
        assert!(extra.get("source_hash").is_some());
    }

    #[test]
    fn section_index_with_only_extra_copy_is_translated() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        fx.write_page("content/_index.md", extra_fm(), "\n");
        let c = EchoClient { calls: Cell::new(0) };
        translate_with(fx.root(), &fx.config(), None, false, "k", &c).unwrap();
        assert_eq!(c.calls.get(), 1);
        let es = fm_of(&fx, "content/_index.es.md");
        assert_eq!(es["extra"]["hero"]["lede"].as_str(), Some("[es] Curriculo reads the work."));
        assert_eq!(es["title"].as_str(), Some("[es] Home Curriculo"));
    }

    #[test]
    fn title_only_section_is_still_skipped() {
        let fx = Fixture::new();
        fx.write_page("content/blog/_index.md", "title = \"Blog\"\n", "\n");
        let c = EchoClient { calls: Cell::new(0) };
        translate_with(fx.root(), &fx.config(), None, false, "k", &c).unwrap();
        assert_eq!(c.calls.get(), 0);
        assert!(!fx.root().join("content/blog/_index.es.md").exists());
    }

    #[test]
    fn editing_extra_copy_marks_siblings_stale() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        fx.write_page("content/home/index.md", extra_fm(), "Body Curriculo.\n");
        let c = EchoClient { calls: Cell::new(0) };
        translate_with(fx.root(), &fx.config(), None, false, "k", &c).unwrap();
        translate_with(fx.root(), &fx.config(), None, false, "k", &c).unwrap();
        assert_eq!(c.calls.get(), 1, "unchanged page is fresh");

        fx.write_page(
            "content/home/index.md",
            &extra_fm().replace("Start Free", "Start now"),
            "Body Curriculo.\n",
        );
        translate_with(fx.root(), &fx.config(), None, false, "k", &c).unwrap();
        assert_eq!(c.calls.get(), 2, "an [extra] edit must regenerate the sibling");
    }

    #[test]
    fn page_without_extra_copy_keeps_the_previous_hash() {
        // Pre-[extra] formula: sha256(title \0 description \0 body).
        let mut h = Sha256::new();
        h.update(b"T\x00D\x00B");
        let old: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(source_hash(&t("T", "D", "B")), old);
        let with_extra = Translatable { extra: vec![("a".into(), "x".into())], ..t("T", "D", "B") };
        assert_ne!(source_hash(&with_extra), old);
    }

    #[test]
    fn sibling_only_extra_keys_survive_a_retranslation() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        fx.write_page("content/home/index.md", extra_fm(), "Body Curriculo.\n");
        fx.write_page(
            "content/home/index.es.md",
            "title = \"old\"\n[extra]\nnoindex = true\nsource_hash = \"stale\"\n",
            "old body\n",
        );
        let c = EchoClient { calls: Cell::new(0) };
        translate_with(fx.root(), &fx.config(), None, false, "k", &c).unwrap();
        let es = fm_of(&fx, "content/home/index.es.md");
        assert_eq!(es["extra"]["noindex"].as_bool(), Some(true));
        assert_ne!(es["extra"]["source_hash"].as_str(), Some("stale"));
    }

    #[test]
    fn markup_ok_rejects_a_lost_close_tag_and_a_new_bare_lt() {
        let en = t("T", "", "<p>Read <a href=\"/x\">this</a> now.</p>");
        let good = t("T", "", "<p>Lisez <a href=\"/x\">ceci</a> maintenant.</p>");
        let lost = t("T", "", "<p>Lisez <a href=\"/x\">ceci maintenant.</p>");
        let bare = t("T", "", "<p>Lisez <a href=\"/x\">ceci</a> en <20 min.</p>");
        assert!(markup_ok(&en, &good).is_ok());
        let err = markup_ok(&en, &lost).unwrap_err().to_string();
        assert!(err.contains("</a>"), "{err}");
        assert!(markup_ok(&en, &bare).is_err());
        // a bare `<` already in the source may stay
        let src_lt = t("T", "", "Setup in <20 min.");
        assert!(markup_ok(&src_lt, &t("T", "", "Listo en <20 min.")).is_ok());
    }

    #[test]
    fn pack_html_tags_round_trips_and_skips_bare_lt() {
        let src = "<p>Hire faster with <strong>Curriculo ATS</strong>.</p> Setup in <20 min.";
        let (packed, tags) = pack_html_tags(src);
        let names: Vec<&str> = tags.iter().map(|t| t.tag.as_str()).collect();
        assert_eq!(names, vec!["<p>", "<strong>", "</strong>", "</p>"]);
        assert!(packed.contains("XHTML0000X"));
        assert!(packed.contains("XHTML0003X"));
        assert!(!packed.contains("<p>"));
        assert!(packed.contains("<20 min"));
        let restored = unpack_html_tags(&packed, &tags).unwrap();
        assert_eq!(tag_counts(src).tags, tag_counts(&restored).tags);
        assert!(restored.contains("<20 min"));
        assert!(markup_ok(&t("T", "", src), &t("T", "", &restored)).is_ok());
    }

    #[test]
    fn unpack_html_tags_does_not_eat_index_ten_when_restoring_one() {
        // 12 tags: unpadded XHTML1X is a prefix of XHTML10X.
        let src = (0..12).map(|i| format!("<td>{i}</td>")).collect::<String>();
        let (packed, tags) = pack_html_tags(&src);
        assert!(packed.contains("XHTML0001X"));
        assert!(packed.contains("XHTML0010X"));
        assert!(packed.contains("XHTML0011X"));
        let restored = unpack_html_tags(&packed, &tags).unwrap();
        assert_eq!(tag_counts(&src).tags, tag_counts(&restored).tags);
        assert_eq!(tags.len(), 24, "open+close per cell");
        let lower = packed.replace("XHTML0010X", "xhtml0010x");
        assert_eq!(
            tag_counts(&src).tags,
            tag_counts(&unpack_html_tags(&lower, &tags).unwrap()).tags
        );
    }

    #[test]
    fn unpack_html_tags_survives_nllb_dropping_real_tags() {
        // What raw NLLB does to a table: eat the tags, keep inner words.
        // Packed form has no real tags, so a tag-dropping model still
        // round-trips structure after unpack.
        let src = "<table><tr><td>Plan</td><td>Price</td></tr></table>";
        let (packed, tags) = pack_html_tags(src);
        let stripped = packed.replace('<', "").replace('>', "");
        let restored = unpack_html_tags(&stripped, &tags).unwrap();
        assert_eq!(tag_counts(src).tags, tag_counts(&restored).tags);
        assert!(markup_ok(&t("T", "", src), &t("T", "", &restored)).is_ok());
    }

    #[test]
    fn unpack_rejects_leftover_placeholder() {
        let src = "<p>Ready to hire?</p>";
        let (packed, tags) = pack_html_tags(src);
        let dup = format!("{packed} XHTML0001X");
        let err = unpack_html_tags(&dup, &tags).unwrap_err().to_string();
        assert!(err.contains("reordered") || err.contains("repeated") || err.contains("missing"), "{err}");
    }

    #[test]
    fn unpack_rejects_reordered_placeholders() {
        // ja/ko/ar can swap open/close while keeping counts.
        let src = "Click <strong>here</strong>";
        let (packed, tags) = pack_html_tags(src);
        let swapped = packed.replace("XHTML0000X", "TMP").replace("XHTML0001X", "XHTML0000X").replace("TMP", "XHTML0001X");
        let err = unpack_html_tags(&swapped, &tags).unwrap_err().to_string();
        assert!(err.contains("reordered") || err.contains("got"), "{err}");
    }

    #[test]
    fn unpack_restores_original_spacing() {
        let src = "Hire faster with <strong>Curriculo ATS</strong>.";
        let (packed, tags) = pack_html_tags(src);
        let restored = unpack_html_tags(&packed, &tags).unwrap();
        assert_eq!(restored, src, "no space before the full stop");
        let cjk = "用<strong>Curriculo ATS</strong>更快";
        let (p2, t2) = pack_html_tags(cjk);
        assert_eq!(unpack_html_tags(&p2, &t2).unwrap(), cjk);
    }

    #[test]
    fn translate_api_client_restores_html_when_endpoint_echoes_packed_tokens() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        fx.write_page(
            "content/post/index.md",
            "title = \"Hire with Curriculo ATS\"\ndescription = \"Ready to hire with precision?\"\n",
            "<p>See the <a href=\"/ai-resume-builder/\">AI resume builder</a>.</p>\n<table><tr><td>Plan</td><td>Price</td></tr></table>\n",
        );
        let server = TinyServer::echo(true);
        let client = TranslateApiClient::new(&server.url, Duration::from_secs(30), vec![]).unwrap();
        translate_with_endpoint(fx.root(), &fx.config(), None, false, &client).unwrap();
        let es = fx.page_body("content/post/index.es.md");
        assert!(es.contains("<p>"), "{es}");
        assert!(es.contains("</p>"), "{es}");
        assert!(es.contains("<table>"), "{es}");
        assert!(es.contains("</table>"), "{es}");
        assert!(es.contains("<a href=\"/ai-resume-builder/\">"), "{es}");
        assert!(!es.contains("XHTML"), "tokens must be unpacked: {es}");
        assert!(server.reqs.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn translate_api_client_plain_extra_has_no_pack_tokens() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        fx.write_page("content/_index.md", extra_fm(), "\n");
        let server = TinyServer::echo(false);
        let client = TranslateApiClient::new(&server.url, Duration::from_secs(30), vec![]).unwrap();
        translate_with_endpoint(fx.root(), &fx.config(), None, false, &client).unwrap();
        let es = fx.page_body("content/_index.es.md");
        assert!(!es.contains("XHTML"), "{es}");
        assert!(es.contains("Start Free") || es.contains("Case studies"), "{es}");
    }

    #[test]
    fn translate_api_client_rejects_leftover_placeholder_and_writes_nothing() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        fx.write_page(
            "content/post/index.md",
            "title = \"T\"\ndescription = \"D\"\n",
            "<p>Ready to hire?</p>\n",
        );
        let server = TinyServer::echo_map(|t| format!("{t} XHTML0001X"));
        let client = TranslateApiClient::new(&server.url, Duration::from_secs(30), vec![]).unwrap();
        let res = translate_with_endpoint(fx.root(), &fx.config(), None, false, &client);
        assert!(res.is_err(), "leftover placeholder must fail the page");
        assert!(!fx.root().join("content/post/index.es.md").exists());
        let es = fx.root().join("content/post/index.es.md");
        if es.exists() {
            let body = fs::read_to_string(&es).unwrap();
            assert!(!body.contains("XHTML"), "must not publish leftover tokens: {body}");
        }
    }

    /// Mock that loses a closing tag.
    struct TagDroppingClient;
    impl LlmClient for TagDroppingClient {
        fn translate(&self, f: &Translatable, lang: &str, _key: &str) -> Result<Translatable> {
            Ok(Translatable {
                title: format!("[{lang}] {}", f.title),
                description: format!("[{lang}] {}", f.description),
                body: format!("[{lang}] {}", f.body.replace("</a>", "")),
                extra: f.extra.clone(),
            })
        }
    }

    #[test]
    fn broken_markup_is_not_written() {
        let fx = Fixture::new();
        fx.write_page("content/post/index.md", en_page_fm(), "See <a href=\"/x\">Curriculo</a>.\n");
        let res = translate_with(fx.root(), &fx.config(), None, false, "k", &TagDroppingClient);
        assert!(res.is_err());
        assert!(!fx.root().join("content/post/index.es.md").exists());
    }

    /// Mock that returns the source unchanged.
    struct IdentityClient;
    impl LlmClient for IdentityClient {
        fn translate(&self, f: &Translatable, _lang: &str, _key: &str) -> Result<Translatable> {
            Ok(f.clone())
        }
    }

    #[test]
    fn untranslated_body_is_not_stamped_fresh() {
        let fx = Fixture::new();
        fx.write_page(
            "content/post/index.md",
            en_page_fm(),
            "Curriculo scores every candidate with written reasons.\n",
        );
        let res = translate_with(fx.root(), &fx.config(), None, false, "k", &IdentityClient);
        assert!(res.unwrap_err().to_string().contains("failure"));
        assert!(!fx.root().join("content/post/index.es.md").exists());
    }

    #[test]
    fn short_strings_may_read_the_same_in_both_languages() {
        let en = Translatable { extra: vec![("faq".into(), "FAQ".into())], ..t("Blog", "", "") };
        assert!(not_passthrough(&en, &en.clone()).is_ok());
        // brand tokens, acronyms and names are not words a translation changes
        let api = t("Curriculo ATS REST API", "Curriculo ATS + Google Workspace", "");
        assert!(not_passthrough(&api, &api.clone()).is_ok());
        let names = t("Maria Fernanda Silva", "Google Cloud Platform", "");
        assert!(not_passthrough(&names, &names.clone()).is_ok());
        for prose in
            ["Compare every major ATS", "Curriculo ATS vs Lever: Which AI ATS Is Right for You?"]
        {
            let en = t(prose, "", "");
            assert!(not_passthrough(&en, &en.clone()).is_err(), "{prose}");
        }
    }

    #[test]
    fn non_copy_values_are_never_sent() {
        for value in ["en-US", "pt-BR", "translateY(10px) rotate(3deg)", "var(--accent)"] {
            assert!(!is_copy(value), "{value}");
        }
        for value in ["Portfolio", "Score (0–100)", "Start Free", "portfolio"] {
            assert!(is_copy(value), "{value}");
        }
        let fm: toml::Value =
            toml::from_str("[extra.faq]\n\"v1.0\" = \"Is it free?\"\nq = \"Is it free?\"\n")
                .unwrap();
        let paths: Vec<String> = extra_copy_leaves(&fm).into_iter().map(|(p, _)| p).collect();
        assert_eq!(paths, vec!["faq.q"], "a dotted key cannot round-trip, so it is not sent");
    }

    #[test]
    fn markup_ok_rejects_empty_fields_and_changed_code_or_links() {
        let en = t("Title", "", "See [docs](/docs/) and `{{ x }}`.\n```\ncode\n```");
        let ok = t("Titre", "", "Voir [docs](/docs/) et `{{ x }}`.\n```\ncode\n```");
        assert!(markup_ok(&en, &ok).is_ok());
        assert!(markup_ok(&en, &Translatable { title: String::new(), ..ok.clone() }).is_err());
        let link_lost = Translatable { body: ok.body.replace("](/docs/)", ""), ..ok.clone() };
        assert!(markup_ok(&en, &link_lost).unwrap_err().to_string().contains("]("));
        let fence_lost = Translatable { body: ok.body.replace("```\ncode\n```", "code"), ..ok };
        assert!(markup_ok(&en, &fence_lost).is_err());
    }

    #[test]
    fn set_leaf_reports_a_path_that_does_not_resolve() {
        let mut extra: toml::Value = toml::from_str("[hero]\ncta = \"x\"\nn = 1\n").unwrap();
        assert!(set_leaf(&mut extra, "hero.cta", "y"));
        assert!(!set_leaf(&mut extra, "hero.missing", "y"));
        assert!(!set_leaf(&mut extra, "hero.n", "y"), "not a string");
        assert!(!set_leaf(&mut extra, "hero.cta[0]", "y"));
    }

    #[test]
    fn endpoint_sends_extra_copy_and_writes_it_back() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        fx.write_page("content/_index.md", extra_fm(), "\n");
        let mock = MockBatch::new();
        translate_with_endpoint(fx.root(), &fx.config(), None, false, &mock).unwrap();
        let sent = mock.texts_of(0);
        assert!(sent.contains(&"Start Free".to_string()));
        assert!(sent.contains(&"Case studies".to_string()));
        assert!(!sent.iter().any(|s| s.starts_with('{') || s == "impact_scoring" || s.is_empty()));
        let es = fm_of(&fx, "content/_index.es.md");
        assert_eq!(es["extra"]["cards"][0]["items"][1].as_str(), Some("Case studies"));
        assert!(es["extra"].get("source_hash").is_some());
    }

    fn hand_es_fm() -> &'static str {
        "title = \"Inicio Curriculo\"\ndescription = \"Desc es\"\n\
         [extra]\nnoindex = false\n\
         [extra.hero]\ncta = \"Empieza gratis\"\nlede = \"Curriculo lee el trabajo.\"\n\
         [[extra.cards]]\ntitle = \"Lee el trabajo real\"\nitems = [\"Portafolios y repositorios\", \"Casos\"]\n"
    }

    #[test]
    fn adopt_stamps_a_complete_hand_translation_and_translate_then_skips_it() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        fx.write_page("content/_index.md", extra_fm(), "\n");
        fx.write_page("content/_index.es.md", hand_es_fm(), "\n");
        adopt(fx.root(), &fx.config()).unwrap();

        let es = fm_of(&fx, "content/_index.es.md");
        assert_eq!(es["extra"]["hero"]["cta"].as_str(), Some("Empieza gratis"), "copy untouched");
        assert!(es["extra"].get("source_hash").is_some());

        let c = EchoClient { calls: Cell::new(0) };
        translate_with(fx.root(), &fx.config(), None, false, "k", &c).unwrap();
        assert_eq!(c.calls.get(), 0, "an adopted sibling is fresh");
    }

    #[test]
    fn adopt_refuses_a_sibling_missing_extra_copy() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        fx.write_page("content/_index.md", extra_fm(), "\n");
        fx.write_page(
            "content/_index.es.md",
            &hand_es_fm().replace("cta = \"Empieza gratis\"\n", ""),
            "\n",
        );
        adopt(fx.root(), &fx.config()).unwrap();
        assert!(fm_of(&fx, "content/_index.es.md")["extra"].get("source_hash").is_none());
    }

    #[test]
    fn adopt_refuses_a_sibling_whose_body_is_still_english() {
        let fx = Fixture::new();
        one_lang_config(&fx);
        let body = "Curriculo scores every candidate with written reasons.\n";
        fx.write_page("content/post/index.md", en_page_fm(), body);
        fx.write_page("content/post/index.es.md", "title = \"Hola Curriculo\"\n", body);
        adopt(fx.root(), &fx.config()).unwrap();
        assert!(fm_of(&fx, "content/post/index.es.md").get("extra").is_none());
    }

    #[test]
    fn recheck_clears_a_fresh_sibling_that_lost_markup_and_keeps_good_ones() {
        let fx = Fixture::new();
        let body = "See <a href=\"/x\">the scoring docs</a> for Curriculo.\n";
        fx.write_page("content/a/index.md", en_page_fm(), body);
        fx.write_page("content/b/index.md", en_page_fm(), body);
        let c = EchoClient { calls: Cell::new(0) };
        translate_with(fx.root(), &fx.config(), None, false, "k", &c).unwrap();
        // Break one sibling the way older translations did: `</a>` dropped.
        let broken = fx.root().join("content/a/index.es.md");
        let text = fs::read_to_string(&broken).unwrap().replace("</a>", "");
        fs::write(&broken, text).unwrap();

        let err = recheck(fx.root(), &fx.config()).unwrap_err().to_string();
        assert!(err.contains("cleared 1"), "{err}");
        assert!(fm_of(&fx, "content/a/index.es.md")["extra"].get("source_hash").is_none());
        assert!(fm_of(&fx, "content/b/index.es.md")["extra"].get("source_hash").is_some());
        assert!(recheck(fx.root(), &fx.config()).is_ok(), "nothing left to clear");

        let before = c.calls.get();
        translate_with(fx.root(), &fx.config(), None, false, "k", &c).unwrap();
        assert_eq!(c.calls.get() - before, 1, "only the cleared sibling is redone");
    }

    #[test]
    fn hash_line_edit_keeps_order_comments_and_line_endings() {
        let src = "+++\r\ntitle = \"Inicio\"\r\n# hand-written, keep me\r\n[extra]\r\nnoindex = false\r\n[extra.hero]\r\ncta = \"Empieza\"\r\n+++\r\nbody\r\n";
        let set = rewrite_hash_line(src, Some("abc")).unwrap();
        assert!(set.contains("[extra]\r\nsource_hash = \"abc\"\r\nnoindex = false"), "{set}");
        assert!(set.contains("# hand-written, keep me\r\n"));
        let fm = front_matter_of(&set).unwrap();
        assert_eq!(fm["extra"]["source_hash"].as_str(), Some("abc"));
        assert_eq!(fm["extra"]["hero"]["cta"].as_str(), Some("Empieza"));

        let again = rewrite_hash_line(&set, Some("def")).unwrap();
        assert_eq!(again, set.replace("abc", "def"), "an existing stamp is replaced in place");
        assert_eq!(rewrite_hash_line(&set, None).unwrap(), src, "removal restores the file");

        // no [extra] header: a new table at the end of the front matter
        let bare = "+++\ntitle = \"T\"\n[extra.hero]\ncta = \"x\"\n+++\nbody\n";
        let fm = front_matter_of(&rewrite_hash_line(bare, Some("h")).unwrap()).unwrap();
        assert_eq!(fm["extra"]["source_hash"].as_str(), Some("h"));
        assert_eq!(fm["extra"]["hero"]["cta"].as_str(), Some("x"));
        // a source_hash key in another table is not the stamp
        let other = "+++\n[extra.hero]\nsource_hash = \"keep\"\n+++\n";
        let fm = front_matter_of(&rewrite_hash_line(other, Some("h")).unwrap()).unwrap();
        assert_eq!(fm["extra"]["hero"]["source_hash"].as_str(), Some("keep"));
        assert_eq!(fm["extra"]["source_hash"].as_str(), Some("h"));
    }

    #[test]
    fn sibling_path_is_prefixed_with_its_language_and_aliases_are_dropped() {
        let fx = Fixture::new();
        fx.write_page(
            "content/eng-post/index.md",
            "title = \"Post Curriculo\"\npath = \"engineering/post\"\naliases = [\"/old-post/\"]\n",
            "Body Curriculo.\n",
        );
        fx.write_page(
            "content/eng-root/index.md",
            "title = \"Root Curriculo\"\npath = \"/engineering/root/\"\n",
            "Body Curriculo.\n",
        );
        let c = EchoClient { calls: Cell::new(0) };
        translate_with(fx.root(), &fx.config(), None, false, "k", &c).unwrap();

        let es = fm_of(&fx, "content/eng-post/index.es.md");
        assert_eq!(
            es["path"].as_str(),
            Some("es/engineering/post"),
            "two pages may not share a URL"
        );
        assert!(es.get("aliases").is_none(), "an alias is an English URL, it would collide");
        let fr = fm_of(&fx, "content/eng-root/index.fr.md");
        assert_eq!(fr["path"].as_str(), Some("/fr/engineering/root/"));
    }

    #[test]
    fn extra_reply_must_match_the_source_shape() {
        let src = vec![("a".to_string(), "One".to_string()), ("b".to_string(), "Two".to_string())];
        let ok = parse_extra_reply(&json!(["Uno", "Dos"]), &src).unwrap();
        assert_eq!(ok, vec![("a".into(), "Uno".into()), ("b".into(), "Dos".into())]);
        assert!(parse_extra_reply(&json!(["Uno"]), &src).is_err(), "short array");
        assert!(parse_extra_reply(&json!(null), &src).is_err(), "missing array");
        assert!(parse_extra_reply(&json!(["Uno", 2]), &src).is_err(), "non-string item");
        assert!(parse_extra_reply(&json!(null), &[]).unwrap().is_empty());
    }
}
