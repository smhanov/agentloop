package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/smhanov/agentloop"
)

// newLineReader returns a function that reads one line (newline
// stripped) per call. It reports io.EOF once the input is exhausted.
func newLineReader(r io.Reader) func() (string, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	return func() (string, error) {
		if !sc.Scan() {
			if err := sc.Err(); err != nil {
				return "", err
			}
			return "", io.EOF
		}
		return sc.Text(), nil
	}
}

// runChat loops one Conversation: the first turn comes from
// -prompt/-prompt-file when given, all turns otherwise (and afterwards)
// come one per stdin line until a blank line or end of input. It returns
// the process exit code.
func (cfg *appConfig) runChat(ctx context.Context, stdin io.Reader, stdout, stderr io.Writer) int {
	handle, err := cfg.newAgent(stdin, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "ask: %v\n", err)
		return 1
	}
	// One shared conversation over the handle's core agent: construction
	// options (system prompt, tools, budgets, cache key, skills) are
	// already frozen into it and carry across turns. For the search and
	// bash presets this is the documented escape hatch, so the bash
	// action protocol does not run in chat mode.
	conv := handle.core.Conversation()

	read := newLineReader(stdin)
	query := cfg.promptText
	for {
		if query == "" {
			line, err := read()
			if err != nil {
				if !errors.Is(err, io.EOF) {
					fmt.Fprintf(stderr, "ask: read stdin: %v\n", err)
				}
				break // end of input ends the session
			}
			if strings.TrimSpace(line) == "" {
				break // blank line ends the session
			}
			query = line
		}
		turn := query
		query = ""

		res, rerr := cfg.chatTurn(ctx, conv, handle, turn, stdout, stderr)
		if rerr != nil {
			if cfg.asJSON && res != nil && res.core != nil {
				printJSON(stdout, res)
			}
			fmt.Fprintf(stderr, "ask: %v\n", rerr)
			return exitCodeFor(rerr)
		}
		if cfg.asJSON {
			printJSON(stdout, res)
		} else {
			fmt.Fprintln(stdout) // blank line between turns
		}
	}
	if !cfg.asJSON {
		printCost(stdout, conv.TotalUsage(), "total ")
	}
	return 0
}

// chatTurn runs one conversation turn and prints its answer. It returns
// the turn's result even alongside an error (a budget-exhausted turn
// still carries a partial answer).
func (cfg *appConfig) chatTurn(ctx context.Context, conv *agentloop.Conversation, handle *agentHandle, query string, stdout, stderr io.Writer) (*runResult, error) {
	if cfg.stream {
		events, err := conv.AskStream(ctx, query)
		if err != nil {
			return nil, err
		}
		p := &streamPrinter{stdout: stdout, stderr: stderr, verbose: cfg.verbose}
		core, runErr := p.consume(events)
		p.endLine()
		res := newRunResult(handle.kind, core)
		if runErr != nil {
			return res, runErr
		}
		if res == nil || res.core == nil {
			return nil, errors.New("agent returned no answer")
		}
		if !cfg.asJSON {
			printSummary(stdout, res)
		}
		return res, nil
	}

	core, err := conv.Ask(ctx, query)
	if core == nil {
		return nil, err
	}
	res := newRunResult(handle.kind, core)
	if err != nil {
		return res, err
	}
	if !cfg.asJSON {
		fmt.Fprintln(stdout, strings.TrimRight(core.Text, "\n"))
		printSummary(stdout, res)
	}
	return res, nil
}
