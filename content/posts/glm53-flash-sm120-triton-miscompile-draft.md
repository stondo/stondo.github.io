---
title: "GLM-5.3-Flash on SM120: The Triton Miscompile That Speaks in locklock (DRAFT SEED)"
date: 2026-08-27T23:30:00+00:00
draft: true
description: "Draft seed parked from the Flash-Next post (2026-08-28): the full GLM-5.3-Flash SM120 saga. Publish only when GLM actually works (user decision 2026-08-27)."
---

> Status: parked. Publish when GLM-5.3-Flash actually serves correctly on this
> hardware (user decision 2026-08-27). Watch
> [upstream issue #4](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark/issues/4)
> and [vLLM PR #53906](https://github.com/vllm-project/vllm/pull/53906).
> Fresh facts live in private project notes from the debugging session.
> SANITIZE BEFORE PUBLISH: no hostnames, no internal service names, no
> personal paths (same policy as the 2026-08-28 sanitization pass).

## From one gigabyte short to a Triton miscompile

My first GLM-5.3-Flash attempt died at weight load: 90.5 GiB of NVFP4 weights per GPU plus a 3.4 GiB Marlin repack workspace against 95.01 GiB of capacity. One gigabyte short, at the one phase no context or sequence knob can touch. Parked.

The escape was a different MoE backend. `--moe-backend flashinfer_cutlass` skips the Marlin `convert_to_nvfp4_moe_kernel_format` repack entirely (it needed an `nvrtc.h` symlink and an unversioned `libnvrtc.so` inside the image for its JIT, with FlashInfer and CUTLASS DSL pinned), and with `--block-size 2304`, gpu-mem-util 0.98, eager mode, vision off and a 65,536-token window (91,405-token pool), GLM-5.3-Flash **boots and serves on the same two cards**. Eight minutes cold, most of it FlashInfer JIT that a cache volume now absorbs.

Then it answered: a plausible first token, then phrase loops, `locklocklock`, every request, at every temperature. A booted stack producing garbage is worse than one that OOMs, because nothing in the log is telling you where the lie lives.

The exoneration list first, because that is the boring half of debugging: not the weights (121/121 shards sha256-verified against HF), not the NoPE-MLA attention kernel (probed against an fp32 reference at rel 0.0024), not the MoE path, not KV dtype, not NCCL, not custom allreduce. The smoking gun was a self-consistency probe against vLLM's flash-linear-attention KDA chunk kernel, `chunk_kda_with_fused_gate`, compiled for sm_120f. Its token outputs are **bit-exact**, full chunk or split in halves. The **recurrent state it returns is wrong whenever an initial state is carried in**: norm 0.14 where an unsplit run yields 21.07, off by 146x, exactly what you get if the kernel keeps only the chunk-local contribution and drops the carried state. The decode path consumes that poisoned state and overflows to NaN within one token. Prefill stays coherent because it never leaves the chunk kernel; every decode step then runs on garbage. Which is precisely "plausible first token, then locklock".

The same code path compiles correctly on GB10 (sm_121a), where the community recipes actually run: an arch-specific Triton miscompile in the final-state accumulate, not a model bug, not my config. The minimal repro is [filed upstream](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark/issues/4), and the open [vLLM GLM-5.3-Flash PR #53906](https://github.com/vllm-project/vllm/pull/53906) may replace the whole path when it lands. Until then the model stays parked a second time, for a much more interesting reason than a gigabyte.

## Lesson for the list

**Garbage output from a clean boot means the lie is inside a kernel.** GLM-5.3-Flash's KDA chunk kernel passed every output check as bit-exact while handing decode a recurrent state that was 146x wrong. When a hybrid-attention model loops after a plausible first token, probe the states it carries, not just the tokens it emits.
