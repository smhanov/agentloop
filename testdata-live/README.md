# testdata-live — local live-test corpus

This directory holds the **local, never-committed prompt corpus** used to
exercise agentloop against a real model and real search results. Its
content is derived from the internal design spec (`specs.md` §14), which
is the single source of truth for the corpus text. Everything here except
this README is gitignored: `.gitignore` excludes `/testdata-live/*` and
re-admits only this README.

## Why the split exists

The shipped repository is domain-neutral (`specs.md` §15): no committed
file — source, tests, testdata, docs — may contain corpus terminology.
The acceptance corpus nevertheless stresses everything a search agent
must do: time-sensitive facts, structured JSON output, multi-entity
queries, and citation-style overrides. The convention resolves the
tension:

- **Committed offline tests** (each package's `*_test.go`) replicate
  every corpus category's *loop shape* with genericized queries — e.g.
  the head-to-head JSON comparison becomes two programming languages with
  the same verdict schema, the entity-identification prompt becomes
  resolving an organization's official name and country from a website
  domain. The default suite stays hermetic and needs no credentials.
- **The live corpus** (this directory, gitignored) carries the verbatim
  prompts for manual CLI runs and for the opt-in `-tags live` suite.

## Layout

    testdata-live/
      README.md               this file — the only committed entry
      prompts/*.md            one verbatim prompt per file (gitignored)
      prompts/manifest.json   per-entry metadata (gitignored)

`manifest.json` records, for every prompt file: its `category`, the
production prompt it mirrors (`mirror`), `live` or `offline` (`live` =
requires real search; `offline` = loop-behavior control answerable
without tools), and the `pass` criteria from `specs.md` §14.

## Running the corpus manually

With credentials exported (see below), feed every prompt file to the CLI
in one pass:

```
for f in testdata-live/prompts/*.md; do echo "===== $f"; go run ./examples/ask -provider zai -model glm-5.3-flash -base-url https://api.z.ai/api/coding/paas/v4 -serper-key "$SERPER_API_KEY" -prompt-file "$f" -json; done
```

`-json` prints the answer object (text, steps, tool calls, sources,
usage), which lets pass criteria that reference tool-call counts, source
lists, or JSON structure be checked mechanically. Drop `-json` for plain
text, or add `-stream -v` to watch tool activity live. The `offline`
manifest entries need no search key.

## Running the live test suite

The `search` package carries an opt-in live smoke suite behind the build
tag `live` (`specs.md` §12.10): three structural tests — a current-facts
question, a search→fetch chain, and a zero-tool control — run against a
real model plus real search. It is excluded from the default suite and
from CI, and does not count against the default suite's time budget:

```
export ZAI_API_KEY=… SERPER_API_KEY=…
go test -tags live ./search -v -timeout 10m
```

Without either credential variable set, every live test skips with a
clear message; without the tag the file is not compiled at all.

## Rules

- Never `git add` anything in this directory except this README.
- Never write key material into prompt files, `manifest.json`, captured
  run output, or plans; refer to `ZAI_API_KEY` / `SERPER_API_KEY` by
  name only.
- New corpus entries start in `specs.md` §14 (source of truth), then land
  here as prompt file + manifest entry, with a genericized loop-shape
  equivalent added to the relevant package's committed tests.
