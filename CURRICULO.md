# Curriculo Zola fork

This is Curriculo’s fork of [getzola/zola](https://github.com/getzola/zola). Upstream
behavior is unchanged; we add three subcommands used by `curriculo-tech/landing-website`
(ADR-008 files-as-truth: content lives as committed markdown under the site root).

| Release pin | What it ships |
|---|---|
| `v0.23.2-curriculo.1` | `zola translate` |
| `v0.23.2-curriculo.2` | `zola graph migrate` / `graph refresh` |
| `v0.23.2-curriculo.3` | Clean migrate extraction (metadata description, boilerplate strip, asset URL skip) |
| `v0.23.2-curriculo.13` | `zola indexnow` — IndexNow ping for changed content |
| `v0.23.2-curriculo.14` | graph: `OPENROUTER_URL` / `OPENROUTER_MODEL` env overrides (OpenAI-compatible gateway) |
| `v0.23.2-curriculo.15` | `zola translate` translates `[extra]` copy and sections, rejects broken or untranslated output, `--adopt` / `--recheck` |

Landing CI pins the binary via `ZOLA_VERSION` / `ZOLA_BIN_URL` (never `latest`).

## Commands

### `zola translate`

Generate/refresh co-located `index.<lang>.md` and `_index.<lang>.md` siblings
via OpenRouter (`openai/gpt-4o-mini`), or the endpoint at `TRANSLATE_URL`.
Hash-gated on `extra.source_hash`. Needs `OPENROUTER_API_KEY` unless
`TRANSLATE_URL` is set.

```bash
zola --root <site> translate [--max N] [--dry-run]
zola --root <site> translate --adopt      # stamp good hand translations fresh, no API
```

- **What is translated:** `title`, `description`, the body, and every copy-like
  string in `[extra]` (nested tables and arrays included). Ids, routes, hashes
  and machine data are never sent: keys such as `canonical`, `source_url`,
  `jsonld`, `icon`, `id`, `slug`, `date`, `author`, any `*_url`/`*_id`/`*_hash`,
  and values that are URLs, paths, slugs or embedded JSON.
- **Sections:** `_index.md` is translated like a page. A file with neither a
  body nor `[extra]` copy (a title-only stub) is skipped.
- **Freshness:** `source_hash` covers title, description, body and `[extra]`
  copy. A page without `[extra]` copy hashes exactly as before, so upgrading
  does not re-translate it. Pages that have `[extra]` copy get a new hash and
  are re-translated once (or stamped with `--adopt`); so does every page whose
  copy selection changes when the skip lists or the glossary change.
- **Writes:** the sibling follows the English front matter with the
  translated strings at their paths. Top-level `[extra]` keys only the sibling
  has (`noindex`, say) are kept.
- **Never written:** output that loses a brand token, changes the HTML tags or
  adds a bare `<` (the cause of footers rendering inside `<main>`), or, on the
  OpenRouter path, returns prose unchanged (two or more lowercase words, or
  five or more words in all; brand tokens, acronyms and names do not count).
  Empty output for a non-empty field, and a changed count of code fences,
  `{{`/`{%` shortcodes or `](` link targets, are rejected too.
- **`--recheck`:** runs the same checks on siblings already stamped fresh and
  removes the stamp from those that fail, so the next run redoes them. Exits
  non-zero when it cleared anything.
- **Any language:** a code outside the built-in names is passed to the model
  as-is, so adding `[languages.<code>]` to config.toml is enough.
- **`--adopt`:** marks an existing sibling fresh only when it has every
  `[extra]` copy path and passes the same checks. Copy is never changed.

### `zola graph`

Topical knowledge graph: pages ↔ topics ↔ relations, committed as JSON under
`data/graph/`. See [docs/curriculo/graph.md](docs/curriculo/graph.md).

```bash
# ONCE per origin (Firecrawl + OpenRouter) — writes content/** + data/graph/**
zola --root <site> graph migrate --from https://example.com [--max N] [--force] [--dry-run]

# Forever after (OpenRouter only) — updates data/graph from local markdown
zola --root <site> graph refresh [--max N] [--dry-run]
```

**Hard rule:** Firecrawl is migrate-only. `refresh` and `build` never crawl.

### `zola indexnow`

Submit page/section URLs to [IndexNow](https://www.indexnow.org/) so
participating engines re-crawl changed content. Defaults to diffing `content/`
against `HEAD~1` (`--diff-base` to override, e.g. `origin/master` in CI); a
changed page submits its permalinks in every language. The site is loaded
read-only (like `zola check`) — nothing is written to the output directory.
Needs `INDEXNOW_KEY` (`INDEXNOW_URL` overrides the endpoint).

```bash
zola --root <site> indexnow [--diff-base <ref> | --full | --urls-file <file>] [--dry-run]
```

Network lives only here — `zola build` stays offline. Google does not
participate in IndexNow (Bing, Yandex, Seznam and Naver do).

## Operator loop

Identity (org, pillars, forbidden related-pairs) lives in the site's
`config.toml` `[extra.graph]`, not in this binary.

```bash
zola graph migrate --from https://example.com   # once
zola graph refresh
zola build --base-url https://example.com/
```

`zola build` stays offline. Firecrawl is migrate-only.

## Secrets

| Secret | Used by |
|--------|---------|
| `OPENROUTER_API_KEY` | `translate`, `graph migrate`, `graph refresh` |
| `FIRECRAWL_API_KEY` | `graph migrate` only |
| `INDEXNOW_KEY` | `zola indexnow` only |

## Related

- Design: [docs/curriculo/graph.md](docs/curriculo/graph.md)
- Landing CI: `curriculo-tech/landing-website` (`.github/workflows/master.yml`, `graph-migrate.yml`)
- Workspace design notes: `~/dev_ws/c/docs/superpowers/specs/2026-08-11-zola-graph-design.md`
