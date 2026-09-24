---
title: "Top 12% at the AI Chessathon: One CPU Core, Four Dead Neural Nets, and the Residual That Finally Earned Its Elo"
date: 2026-09-24T09:00:00+00:00
draft: false
description: "I entered an online tournament where AIs play chess: 465 bots, one CPU core, no GPU, no network, a 50 MB zip, and 120 seconds per game. The engine that finished 58th is a numba-jitted alpha-beta in pure Python with a 128-wide NNUE residual on top of a classical evaluation. Four neural networks died to teach me that lower training loss does not play chess."
summary: "A tournament report from the AI Chessathon: a pure-Python numba chess engine on one CPU core, four replacement neural evaluators that all lost to classical piece-square tables (one 0-30), the residual NNUE that beat them (+224 Elo, confirmed at tournament time control), a time-management bug found by post-mortem forensics mid-event, and the promotion gates that kept an AI coding agent honest. Final: 58th of 465, rating 2207."
tags:
  - chess
  - numba
  - nnue
  - python
  - alpha-beta
  - tournament
  - ai-agents
  - codex
categories:
  - engineering
keywords:
  - AI Chessathon
  - chess engine
  - NNUE
  - numba
  - alpha-beta search
  - PeSTO
  - Syzygy tablebases
  - Python chess
  - Stockfish distillation
  - chess bot tournament
---

## What I entered, and how it ended

The [AI Chessathon](https://aichessathon.com) is an online tournament where the participants are programs. You upload a zip, the platform runs your bot against everyone else's, hourly rated rounds, a qualification ladder through September 11, finals after that. When I joined, the field had 299 bots. By the time the ladder closed there were 465, with names like CheckMateGPT, Slopfish, Stonkfish, and What Even Is En Passant.

The constraints are the interesting part. Each game runs in a container with **one CPU core, 2 GB of RAM, no GPU, no network**. Python 3.12 with exactly four libraries preinstalled: `torch` (CPU), `numpy`, `python-chess`, and `numba`. Time control 120 seconds plus 0.5 per move, a 90-second initialization budget, and one rule that quietly shapes everything: **your process is suspended while the opponent thinks**. The submission is a zip, source-only, at most 50,000,000 bytes unpacked. No third-party engines in the zip; opening books and endgame tablebases explicitly allowed; training data unrestricted but neural network weights must be your own.

Twelve days later my bot finished like this:

![BitsEntangled final placement: rank 58 of 465, rating 2207, record 39-24-39, best streak 5, 39 checkmates](/aichessathon-final-placement.jpg)

| | |
|---|---|
| Final rank | #58 of 465 (top 12%) |
| Rating | 2207 |
| Record | 39 wins, 24 draws, 39 losses |
| Checkmates delivered | 39 |
| Best win streak | 5 |

A .500 record sounds middling until you look at the distribution: the field spans engines that are essentially random movers to engines that are essentially Stockfish with a hat on. Half the games won against that field puts you in the top eighth.

The last game of my ladder is a decent summary of the whole thing. Facing CheckMateGPT, my engine ground out a queen endgame, promoted a pawn, and then the log does this — iterative deepening reporting deeper into a forced line, spending about half a second a move:

```
[agent] d12 n422623 score mate in 9 ...
[agent] d1  score mate in 5 ...
[agent] d1  score mate in 4 ...
[agent] d1  score mate in 2 ...
[agent] d1 n70 score mate in 1 clock 18951ms effective 18701ms soft 1.31s hard 3.93s move h8e5
```

Mate counted down from 9 to 1 and delivered. The referee scored it, the ladder closed, and I went to write this.

## The rules are the architecture

Most of the design decisions fall straight out of the platform contract:

- **Process suspension kills pondering.** My first build had a pondering thread — search on the opponent's clock, keep the transposition table warm. Then I re-read the docs, checked the official starter's `rules.py`, and confirmed the judge freezes your process between turns. All that pondering machinery, deleted. Under these rules, all strength must come from per-move search quality.
- **90 seconds of init is a JIT budget.** Numba compiles per signature, at first call, and a cold chess search is tens of seconds of compilation. So the agent starts a daemon thread at import that warms every jitted signature — normal search, ponder-shaped search, a capture-rich position to hit the quiescence branches — bounded at 80 seconds, leaving 10 seconds of handshake headroom. If move 1 arrives while compilation is still running, a pure-Python fallback (capture-preferring, verified legal) plays instead of flagging. This is not paranoia: the platform measured my readiness at **81.3 s of the 90 s budget**. My local machine warmed in 21. Local timings are not platform timings.
- **The 50 MB cap is a budget spreadsheet.** The final zip unpacks to 44.7 MB: 39.5 MB is Syzygy tablebases (complete 3–4 piece, curated 5-piece endings), 3.0 MB is the neural network, and all the source code is under 0.2 MB. The tablebase is loaded at import and probed **only at the root** — if the position is in-table, we play DTZ-optimal moves directly, and the search's choice gets overridden if it disagrees. Zero hot-path cost.
- **Games start from curated positions.** Round 1 began from a Sicilian Dragon at move 6. There is no "book up on the Staunton" — there is only evaluation and search.

## The engine: numba all the way down

The core is deliberately unglamorous: a 64-square **mailbox** board as a numba `@jitclass` — no bitboards. Numba compiles scalar mailbox code extremely well, and the incremental make/unmake with an undo stack means no board copying in the search. Moves are packed into one int32 (`from | to<<6 | promo<<12`). The board maintains zobrist keys, piece lists, material counts, tapered eval bases, and a pawn key, all incrementally.

On top of it, a modern alpha-beta stack, jitted end to end: iterative deepening with aspiration windows (±28 cp, geometric widening), null-move pruning, reverse futility, razoring, late move reductions, SEE-ordered captures, killer moves, history heuristic, singular extensions at depth ≥ 10, mate-distance pruning, internal iterative reduction, and a correction-history table that learns a per-pawn-structure evaluation bias during the search — the same idea modern Stockfish carries. The transposition table is a struct-of-arrays `@jitclass`, because numba can't do pointers, with power-of-two masking and mate-score ply adjustment.

The hot path never touches Python. `python-chess` appears only at the edges: FEN parsing, tablebase queries, the emergency fallback. The result searches **~140–200k nodes per second** on one core — three orders of magnitude below a native engine, but the algorithmic toolbox is the same, and the time manager has to be better to compensate.

## Four neural nets died to teach me one thing

Here is the part I would underline twice. The platform ships `torch` and `onnxruntime`, the rules allow learned weights if you train them yourself, and I had 6.3 million positions annotated by Stockfish at depth 14 sitting on disk. Obviously the answer was a neural network. Obviously.

The classical evaluation was PeSTO tapered piece-square tables plus bishop pair, pawn structure, mobility, rook files, king shelter. Respectable, hand-tuned, about 1920 Elo calibrated against a Stockfish anchor.

Four times I trained a network to **replace** it. Four times the network had better held-out loss than PeSTO. Four times I ran the arena anyway, 30 games against the classical build. The ledger, because I kept one:

| Attempt | Held-out metrics | Arena vs PeSTO |
|---|---|---|
| Flat 768→16 net, sigmoid-MSE | val 0.018 prob-MSE | 32.5% |
| Same, logit loss + rescale | 78% sign agreement | 31.7% |
| HalfKAv2 NNUE, 2×32 accumulators, SCReLU | exact parity, 69% sign | **0–30** |
| Texel-tuned linear PSQT | 77.1% sign | 31.7% / 33.3% |

Zero wins out of thirty. Not "marginal" — the NNUE lost every single game, because a 2×32 net that has never seen a balanced training regime blunders in exactly the ways a search cannot forgive. A later, properly GPU-trained 32-lane net ran at 520k nps versus 978k nps for classical eval and lost all its smoke games **on time**.

The conclusions I now consider load-bearing:

1. **Held-out RMSE is insufficient.** A network must clear a measured CPU inference-cost gate and an actual playing-strength arena, or it is a curiosity.
2. Evaluation features must be side-to-move relative. Absolute features with stm-relative labels are close to unlearnable.
3. Loss in logit/centipawn space, never sigmoid-MSE, which compresses everything to ±900 cp and calls it precision.
4. PeSTO is genuinely good. Do not trust a net until it beats the tables **at chess**, not at loss.

## The one that lived: a residual, not a replacement

What finally won was giving up on replacement. The shipped evaluation is a **blend**: PeSTO does what PeSTO is good at, and a narrow 128-wide integer-quantized network — king-bucketed, board-only features, 16-bit incremental accumulators, lazy ancestor refresh up to 8 plies — contributes a *residual* correction added on top:

```
score = clamp(base + tempo + correction(pos), ±30000)
```

Trained on 6.3M positions generated by my own engine's games and labeled by Stockfish — the weights are legally mine, the teacher only ever touched training. It went through the promotion gate instead of around it: **+224 Elo over pure classical across 320 games** (95% CI [+189, +264]), then confirmed at the exact tournament time control, 160 more games, **+215 Elo** (CI [+167, +272], p ≈ 4×10⁻¹⁴). The 256-wide variant scored slightly better offline and was rejected by a cost gate: 1.28× whole-search cost. The 128-wide net is 26–27% cheaper per search than its fat sibling and keeps the node rate that pays for depth.

The engine that carried this net was the one that climbed from the middle of the pack to the top 12%. Around round 30 it sat at 22nd of 299. The final ladder had grown by 166 more bots and considerably more science.

## Mid-tournament forensics: the sink-guard

Round 66, playing White, my engine played 24.Qd2 and the evaluation started to slide. Post-mortem: the move was −370 cp worse than the teacher's choice, in a position that was already declining but not lost. The interesting part was *why*. The log showed the search had stopped at depth 15 while **13.4 seconds of already-approved clock sat unused** — the "stable move, save time" heuristic was doing its job faithfully on a sinking evaluation, committing to a shallow decision exactly when the position demanded the full budget.

So round 94's build shipped a **sink-guard**: when two consecutive own evaluations drop by ≥150 cp and at least 20 seconds remain, the soft deadline doubles — pushing the stability factors past the hard deadline so the search spends its approved budget instead of early-stopping shallow. Validated in A/B arenas (54.0% at fast TC, zero false terminations), audited across 218 in-tournament decisions with zero false triggers, and silent for all 109 rounds — the losses that remained were honest ones.

That forensics pass produced the finding I consider the most valuable single fact of the event: **of 40 losses, 34 were gradual positional bleeds** — a hundred centipawns here, a hundred there, confirmed by a Stockfish referee pass. Not blunders, not clock, not openings. The separation in this field is middlegame evaluation quality times search depth. No gimmick fixes that; only a better evaluator at more nodes does.

## Everything I did not ship

The tournament is also a record of restraint, and I am prouder of some of the no's than the yes's:

- An **opening book** distilled from teacher lines: two arenas, no gain, rejected.
- **King-pressure and pawn-safe mobility** terms: +23 and −17 Elo with confidence intervals hugging zero, rejected.
- A **K+P vs K+P endgame dataset** — 9.4M tablebase-generated slots folded into the search as a four-piece specialist: a 160-game match that came back −1.875 percentage points with a CI of [−6.45, +2.70]. No confirmed gain, no confirmed regression, therefore **not promoted**, per the pre-registered rule. That campaign cost two days.
- Post-event, a refit of the net on 1.3M fresh self-play positions: a wash, within ±2–6%. Not shipped either, because dead is dead.

Every one of these had a written plan before data existed, a fixed promotion gate, and a ledger entry after. When the final tuning sweep closed — six search-constant variants, every confidence interval spanning zero at n=100, extended to n=300, still noise — the last commit of the campaign reads `zero open threads`. I have shipped a lot of things in my life with less certainty than the things I refused to ship here.

## The agent that ran the experiments

I did not write this engine alone, and that is the other half of the story. The workhorse was **Codex**, on the machine I call perception, for something like two weeks of sessions. My job drifted from writing code to directing a research program: I chose hypotheses and made the ship/no-ship calls; the agent ran the collect-annotate-encode-fit-gate loops, the arenas, the forensics.

What made that safe was not model quality — it was **durable context and pre-registered gates**. The project keeps a `HANDOFF.md` that grew to ~200 KB: every experiment's plan, outcome, and do-not-redo verdict, so a fresh agent session starts with the entire campaign memory and none of the campaign's mistakes. No change reaches a zip without winning an arena whose pass criteria were written before the games started. The discipline caught two bad ships I would otherwise have made, and it is exactly what made an AI-written engine trustworthy enough to bet a tournament on. The gates are the product; the engine is what they produced.

Behind it, the usual suspects from this blog: the 13700K dev box, the RTX 5090 training the nets, a 32-thread machine grinding overnight self-play (31 shards, 131k labeled positions in one night), and a small VPS where the engine lives on as a playable toy — you can lose to it yourself at [chess.bitsentangled.ch](https://chess.bitsentangled.ch).

## The checklist I would hand my past self

1. Read the platform's `rules.py`, not the docs' summary. The process-suspension rule deleted a week of pondering work in one afternoon. The starter repo is the spec.
2. Numba compiles per signature, at first call. Warm every signature inside the init budget, with the shapes the real calls use, and keep a legal fallback move for the move-1 emergency.
3. Local timings are not platform timings. 21 s at home, 81 s on the judge, same code. Measure readiness through the platform's own smoke games.
4. A learned evaluator replaces nothing until it wins an arena. Held-out loss, parity tests, sign agreement — none of it plays chess.
5. The cheapest strong architecture may be the one that *adds* to classical eval instead of replacing it. Residual beats revolutionary at fixed node rate.
6. Size the net by search cost, not accuracy. A net 2% better offline at 1.28× cost is a slower engine.
7. When results confound you, read the game logs before training anything. The sink-guard was a five-line fix found by forensics, worth more than every rejected search tweak combined.
8. Write the promotion gate before the experiment, and honor it when it says no. Especially then.

## What's next

The next engine is already specced, and it sheds this one's two real constraints: single-threaded search and the 50 MB mindset. The plan is a split-brain design — a Rust lazy-SMP search on all CPU cores, and the NNUE as a GPU-resident batched inference server, because the GPU wants batches and the search wants latency: the batcher is the product. The training pipeline, the data farm, and the gate discipline carry over unchanged. Stockfish is a fifteen-year community effort and I am one person with a garage fleet, but the moat — a pipeline where every change has to earn its Elo — is already built and battle-tested at 58th of 465.

If you want to lose a lunch break to the engine, it is right here: [chess.bitsentangled.ch](https://chess.bitsentangled.ch). It plays the exact tournament build, and it has gotten quite good at grinding queen endgames.
