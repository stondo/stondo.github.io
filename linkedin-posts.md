# LinkedIn posts for the three newest blog articles
# Paste each block separately. Both hook line and body are inside each block.

---

## Follow-up: fleet-doctor — "query the endpoints" became a command (NEWEST — post this next; the Post 0 audit post is its prequel)

My chaos drill passed. It was lying.

Two days ago I wrote that my coding agent's routing config was quietly lying to me, and that the cure was to query the endpoints instead of reading the config. That line became a tool — and the tool then caught me.

fleet-doctor: one CLI over the same three-box fleet. probe checks endpoint health, model-roster drift, and credential resolution for every fallback and promotion target — the exact bug classes from the audit, now regression tests: a planted ghost promotion target exits 2, a planted stale model id exits 1, red until caught. bench produces TTFT-vs-context curves per tier and a cost-per-role table where the local rows literally read "free" and the cloud meters tick in the same table. chaos runs induced-failure drills — dry-run by default, two flags to go live, a documented recovery for every failure. And a timer runs the check every 30 minutes into the existing Grafana, with CI-friendly exit codes: 0 clean, 1 degraded, 2 broken.

Then the lesson I did not plan for. The first live stop-service drill "passed": systemctl returned 0 and the health probe stayed green throughout. False. I had stopped a same-named system-scope decoy unit; the live container belonged to a user-scope unit and never blinked. The drill became honest only when it verified what the endpoint serves instead of what the command returned. The re-run: real teardown, real connection reset, real recovery — MTTR 236 seconds measured — while the agent riding that very endpoint failed over to the 5090 mid-drill and kept working.

A green check is a claim, not a fact. Verify the endpoint. Then pull the lever and verify again.

The post it continues (now with a P.S., and one correction to my own claims — the deep box quietly turned out to be multimodal):
https://stondo.github.io/posts/omp-local-first-routing-homelab-fleet/

---

## Post 0: omp local-first routing (NEWEST — post this first, with static/omp-routing-card.png as the image)

My coding agent's config was quietly lying to me in three places. All three were invisible until I stopped reading the config and started querying the endpoints.

Setup: omp (a Pi-fork coding agent) routed over three local boxes — 2x RTX PRO 6000 brain at 512K context, an RTX 5090 workhorse running my NVFP4 27B for subagent fan-out, an RTX 4080 senses node — plus exactly two cloud API keys: z.ai GLM 5.3 and DeepSeek.

Finding 1: a context-promotion fallback pointing at Claude Opus... on a machine with no Anthropic credential. Dead on arrival, failing silently forever.

Finding 2: the agent's internal judge — the thing that classifies every prompt and detects broken streams — was supposed to use a decision-model API I have no key for. So it silently degraded to prompting my 27B to generate yes/no answers. Every single turn. On the GPU that runs my subagents.

The fix is my favorite kind: Ollaya serving an 8B decision model on the senses node, speaking the exact judgment protocol the agent already expects. Calibrated probabilities in one forward pass, 0.93 seconds on CPU. My 5090 has never been this idle.

Finding 3: the "V4.1" model I configured does not exist as an id. deepseek-flash IS V4.1-Flash. And deepseek-v4-pro has been server-routing to it for weeks — so my slow lane now auto-tracks the DeepSeek flagship, by accident.

The principle that survived the audit: local models carry the token volume, cloud keys carry the decisions. A 400K-token prefill costs me electricity, not API credits.

One more thing now live in the same agent: the Context Language Models approach — the model curates its own context file while it works. Working memory inside the session, cairnkeep (the durable memory layer I open-sourced in August) across sessions. The session forgets on purpose. The stack remembers.

Full config, receipts, and the local<->cloud fallback lattice:
https://stondo.github.io/posts/omp-local-first-routing-homelab-fleet/

---

## Post 1: Zoysh (post this first, freshest)

I ported an LLM-enabled shell to zsh so I never have to leave my terminal.

yo "find all python files modified today"

That command now exists in my prompt. Not in a chat window. Not behind a confirmation dialog. At my prompt, editable, waiting for my Enter.

The idea comes from Yosh by Fil Pizlo: Bash with an integrated LLM, built on his own memory-safe C runtime. Glorious engineering, but I live in zsh. So I ported the interaction model as a plain plugin: streaming answers, Ctrl-C that actually cancels, multi-step plans you approve step by step, scrollback context, and one rule that never bends: it never executes anything you did not press Enter on.

It defaults to your own local model endpoint. Mine talks to the GPU under my desk.

zsh plugin, GPL-3.0, works with zinit/antidote/oh-my-zsh:
https://github.com/stondo/zoysh

Full story and the prefill footgun I found in my own first release:
https://stondo.github.io/posts/zoysh-yosh-zsh-port/

---

## Post 2: OBLITERATUS NVFP4 quantization

No NVFP4 build of the uncensored Qwen3.8-27B existed. So I made one, and it took three days mostly because I kept being wrong in interesting ways.

Day 1: discovered that ModelOpt's quantize() ships FAKE quantization. The real packing is a second call nobody told me about. My "success" was a 51 GB file of bf16 pretending.

Day 2: vLLM died on a bare AssertionError, 27 restarts. Three reasonable fixes, all wrong. The monitoring lesson still stings: systemd reports crash loops as "activating".

Day 3: I stopped guessing and instrumented the loader. One run, one log file, the real answer: you cannot exclude the fused linear-attention projections from quantization on 32 GB. They must ship packed.

Result: 21.5 GB, 40 tok/s single stream, 317 tok/s with 8 concurrent subagents on one RTX 5090. Weights are public:

https://huggingface.co/Joestar79/Qwen3.8-27B-OBLITERATED-NVFP4

The full saga, with every trap documented:
https://stondo.github.io/posts/obliteratus-qwen38-27b-nvfp4-rtx5090/

---

## Post 3: AIOS three-node stack

My LLM stack costs me electricity. Three machines, distinct jobs, one router:

- 2x RTX PRO 6000: the brain. DeepSeek V4 Flash at 1M context does the architecture decisions and the hard debugging
- RTX 5090: the workhorse. A 27B handles every subagent, every review, every screenshot. 8 concurrent agents, full speed each
- RTX 4080: the senses. Embeddings, speech to text, text to speech, a Telegram voice bot, and Prometheus/Grafana watching all of it

Between them, a router I wrote: one logical model name, automatic failover, and a code exploration loop that answers with file:line citations. If the big model dies mid session, requests silently land on the 27B. I notice nothing.

The part that actually matters: opencode, pi, and qwen CLI all point at the same router, share the same memory server, and delegate by standing rules. Lessons survive context compaction and tool switches.

Full architecture, benchmarks, and the stuff that did not survive contact with reality:
https://stondo.github.io/posts/aios-three-node-agentic-coding-stack/

---

## Post 4: DeepSeek-V4-Flash-Vision uncensored (post with the attached terminal card image)

I replaced my big text-only model with an uncensored 305B that can see. It runs on two RTX PRO 6000s under my desk, and the weights were the only part that worked on the first try.

That's the beauty of how OrcaRouter built the uncensored variant: the refusal-direction edit is baked directly into DeepSeek's official mixed-precision shards. Same tensors, same layout, byte-for-byte drop-in. Which meant every failure afterwards was mine to fix. A gift, honestly — your stack is the thing you can fix.

Three fights, one afternoon each:

1. A flashinfer kernel on SM120 that had simply never met a vision token. It rejected the multimodal prefill shape outright. The fix is a ten-line dispatch patch: Triton for the vision-sized batches, CUTLASS for decode. 113 tok/s back.

2. Tool calls that corrupted exactly the arguments coding agents live on — entire files as one string — and ONLY in real sessions. Ten clean synthetic runs in a row, then your agent's write call arrives with the content parameter missing. Both bugs were known, open, unfixed upstream. That's when you check what people with your exact GPUs actually run.

3. The answer was vLLM: this whole bug family was fixed there months ago, and the RTX 6000 Pro community wiki maintains a source-locked image with the vision-relevant SM120 fixes already integrated. Swapped engines, dropped the weights in unchanged. 142 tok/s single-stream, native vision, 900K context, DSpark speculative on.

Bonus bug the swap exposed: my router's health checks proved a port was answering — never WHICH model held it. On a shared GPU lane that's a rumor, not a check. So token-miser now verifies the model name against /v1/models before calling a deployment healthy:

glm53@deep → evicted
model "glm-5.3-flash" not served
(lane serves: [DeepSeek-V4-Flash-Vision])

That feature is open-sourced. The full story — kernel archaeology, parser forensics, and why synthetic green means nothing for timing-dependent bugs:

https://stondo.github.io/posts/deepseek-v4-flash-vision-uncensored-two-rtx-pro-6000/

---

## Post 5: MiniMax H3 15s Ref2VA + the 30s watchdog (post with the video link)

My GPU sampled a 15-second AI video for 4 hours 21 minutes, finished it, and then a hardcoded 30-second timeout threw the whole render away. The MP4 is encoded once, at the very end. No partial flush. No error message that meant anything. Just a module-level constant with an opinion about how fast my GPU should be.

That was yesterday. Today the same job completed in 1h45m, and the result is genuinely fun: a 1986 OutRun arcade clip that stays pixel-art for 7 seconds, then melts into photorealistic live action mid-shot, no cuts:

https://youtu.be/mg4ED_trFE8

Three fixes made the difference:

1. The killer watchdog (_ASYNC_OUTPUT_TIMEOUT = 30.0, undocumented, not env-configurable) got patched to 3600 via a bind-mounted file in the systemd quadlet. No image rebuild.

2. Kernel archaeology on sm_120a: "Blackwell" is two architectures. Every fast flashinfer path (cute-dsl, CUTLASS FMHA) needs tcgen05 MMA, which only datacenter Blackwell (sm_100/103/110) has. fa2 runs, then int32-overflows on long sequences. cuDNN attention won by elimination — 2.7x faster than the safe fallback.

3. The output resolution is model-locked at 1344x768, so the speed lever is the INPUT: the reference video went 1440x1080 → 720x540, cutting the conditioning tokens that ride along at every denoise step.

Bonus lesson: my automation then lied to me twice — a finisher script that declared failure after downloading the video perfectly, and a "crash" alert that was my own systemctl stop. Verify artifacts before believing error logs. Especially your own.

All local, two RTX PRO 6000s, no cloud. Full autopsy with the kernel matrix and every trap documented:

https://stondo.github.io/posts/minimax-h3-15s-ref2va-watchdog-sm120-kernels/

---

## Post 6: DeepSeek-V4.1-Flash — the word salad fix (post with the attached terminal card image: linkedin-dsv41-card.png)

My new 552B model benchmarked clean for two days. Then it started answering English questions in fluent, grammatical, completely deranged Chinese. Only in one CLI. Only after a while.

The report from the person actually using it: "garbage after a while, opencode is bad, qwen is better." Two sentences every debugging session should fear.

Four test batteries against the raw API: spotless. 24-turn conversations, tool calls, streaming, a 152K-token prompt — nothing. The engine was innocent of everything I knew how to test.

So I stopped testing the engine and started reading the bytes. A twenty-line logging proxy between the CLI and the lane caught the actual payloads, and the answer was staring at me:

opencode sends zero sampling parameters. No temperature. No top_p. Nothing.

The model pack shipped no generation_config.json either — so every such request ran vLLM's built-in default: temperature 1.0, top_p 1.0. An uncapped sampling tail on a 2-bit quant. DeepSeek's own model card, in a paragraph I had skimmed: "Evaluations use temperature=1.0, top_p=0.95." The temperature was never the bug. The missing nucleus cap was.

The fix is two lines of JSON. Every client fixed at once, no per-CLI settings, nothing for the next tool I adopt to forget.

Also in the series: why 196B parameters of "Engram" memory live on NVMe, the AM5 RAM ceiling that keeps the last 15% of performance behind a hardware wall, and the reasoning-effort knob that exists as 0-100 in one engine and silently no-ops as an integer in another.

The deployment story: https://stondo.github.io/posts/deepseek-v41-flash-552b-exl3-two-rtx-pro-6000/
The bug hunt: https://stondo.github.io/posts/dsv41-flash-word-salad-top-p-opencode/
The full recipe: https://stondo.github.io/posts/dsv41-flash-complete-recipe-two-rtx-pro-6000/

---

## Post 7: AI Chessathon — 58th of 465 (post with the attached screenshot aichessathon.jpg)

My pure-Python chess engine finished 58th out of 465 bots in an online tournament where every participant is a program. One CPU core. No GPU, no network, a 50 MB zip. The hardest part wasn't the chess.

The rules shaped everything: your process is frozen while the opponent thinks (so no pondering), you get 90 seconds of init (so numba JIT warmup becomes an engineering discipline), and games start from curated positions, never the standard opening.

Four neural networks died getting me to the final build. Each one had better training loss than classical piece-square tables. Each one lost the arena anyway — the NNUE replacement went 0–30. The lesson that survived: held-out loss does not play chess. Nothing ships without winning measured games.

What worked: a 128-wide NNUE residual ADDED on top of the classical evaluation, trained on 6.3M positions from my own engine's games, labeled by Stockfish. +224 Elo over the classical engine, confirmed again at the exact tournament time control.

I built it with an AI coding agent (Codex) running the experiments under pre-registered promotion gates: the plan and pass criteria are written before any game runs, and a result that straddles zero gets rejected no matter how much it cost. A two-day endgame campaign died exactly that way. The gates are the product; the engine is what they produced.

The most valuable finding came from losing: 34 of my 40 losses were gradual positional bleeds, not blunders. And one mid-tournament forensics pass found my time manager stopping search early while approved clock sat unused — a five-line fix worth more than every rejected search tweak combined.

Final: rating 2207, 39 wins / 24 draws / 39 losses, 39 checkmates delivered. The last recorded round ended with the engine walking a forced mate from mate-in-9 down to mate-in-1.

You can play the tournament build yourself: https://chess.bitsentangled.ch

Full story, with every dead experiment documented: https://stondo.github.io/posts/aichessathon-numba-nnue-engine-58-of-465/

---

## Post 8 placeholder — keep ordering: newest first above this line when adding new posts
