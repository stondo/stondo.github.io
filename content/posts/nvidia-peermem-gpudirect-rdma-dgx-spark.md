---
title: "Why nvidia_peermem Fails on DGX Spark: a Deep Dive into GPUDirect RDMA, Secure Boot, and Unified Memory"
date: 2026-02-23T00:00:00+00:00
lastmod: 2026-02-23T00:00:00+00:00
draft: false
description: "A detailed investigation into why nvidia_peermem refuses to load on HP ZGX Nano G1n (DGX Spark) systems, covering Secure Boot module signing, kernel source code analysis, the fundamental hardware limitation that makes GPUDirect RDMA impossible on DGX Spark / GB10 SoC, and the PCIe Gen5 x4 bandwidth bottleneck on the ConnectX-7 NICs."
summary: "I chased a modprobe error across two DGX Spark nodes, peeled back layers of Secure Boot signing and kernel build flags, read the nvidia_peermem source code, and discovered that GPUDirect RDMA is architecturally unsupported on the GB10 Grace Blackwell SoC. Here's the full story."
tags:
  - nvidia
  - dgx-spark
  - gpudirect
  - rdma
  - linux
  - kernel
  - networking
  - gpu
  - infiniband
  - mellanox
categories:
  - engineering
  - infrastructure
keywords:
  - nvidia_peermem
  - GPUDirect RDMA
  - DGX Spark
  - HP ZGX Nano G1n
  - GB10
  - Grace Blackwell
  - Secure Boot
  - modprobe
  - Mellanox ConnectX-7
  - unified memory
  - NCCL
  - multi-node training
  - 200GbE
  - QSFP
  - kernel module signing
  - MOK
  - MLNX_OFED
  - PCIe Gen5
  - bandwidth bottleneck
  - LLM inference
  - tensor parallelism
params:
  author: "stondo"
  toc: true
  math: true
---

## Prelude

Two weeks ago, I decided to gift myself 2 HP ZGX Nano G1n to play with. Rather expensive toys for sure, but I had both some fun and some nightmares. The story about the nightmares is for another post, though. Here I want to talk about what I discovered when trying to create a mini cluster.

## The Setup

I run a two-node GPU cluster built from HP ZGX Nano G1n AI Stations, HP's OEM rebranding of the [NVIDIA DGX Spark](https://www.nvidia.com/en-us/autonomous-machines/dgx-spark/). Each node packs a GB10 Grace Blackwell Superchip (ARM64 / aarch64) and the two are connected back-to-back with a Mellanox ConnectX-7 200GbE QSFP cable.

The goal: fast distributed training and inference across both nodes. The 200 Gbps link should give me blazing GPU-to-GPU transfers using GPUDirect RDMA, letting the Mellanox NIC read directly from GPU memory without bouncing through CPU RAM.

Or so I thought.

## The Error

It started with a simple `modprobe`:

```bash
$ sudo modprobe nvidia_peermem
modprobe: ERROR: could not insert 'nvidia_peermem': Key was rejected by service
```

`nvidia_peermem` is the kernel module that enables GPUDirect RDMA. It registers as a "peer memory client" with the InfiniBand subsystem, giving the Mellanox HCA the ability to DMA directly into NVIDIA GPU memory.

Without it, data has to take the slow path:

```
GPU → CPU RAM → Mellanox NIC → (200GbE) → Mellanox NIC → CPU RAM → GPU
```

Instead of the fast path:

```
GPU → Mellanox NIC → (200GbE) → Mellanox NIC → GPU
```

That CPU bounce buffer costs latency, bandwidth, and CPU cycles. I had to fix this.

## Layer 1: Secure Boot and module signing

The `Key was rejected by service` error pointed straight at Secure Boot. I confirmed:

```bash
$ mokutil --sb-state
SecureBoot enabled
```

With Secure Boot enabled, the kernel only loads modules signed by a trusted key. I checked who signed what:

```bash
$ for mod in nvidia-drm nvidia nvidia-modeset nvidia-peermem nvidia-uvm; do
    signer=$(modinfo -F signer "/lib/modules/$(uname -r)/kernel/nvidia-580-open/${mod}.ko")
    echo "${mod}.ko: ${signer}"
  done
```

```
nvidia-drm.ko:     Canonical Ltd. Kernel Module Signing
nvidia.ko:         Canonical Ltd. Kernel Module Signing
nvidia-modeset.ko: Canonical Ltd. Kernel Module Signing
nvidia-peermem.ko: NVIDIA Support                        ← THE OUTLIER
nvidia-uvm.ko:    Canonical Ltd. Kernel Module Signing
```

Four out of five NVIDIA modules were signed by Canonical (Ubuntu's trusted key, already enrolled in MOK). But `nvidia-peermem.ko` was signed by "NVIDIA Support", a key that isn't in the MOK database.

**This looks like a packaging inconsistency.** When Canonical builds the `linux-modules-nvidia-580-open` package, they re-sign all the NVIDIA kernel modules with their own trusted key. Whether intentional or an oversight, `nvidia-peermem.ko` is left with its original NVIDIA signature.

### The fix for Secure Boot (if it were the only problem)

I explored two approaches:

**Option A: disable Secure Boot**, the simplest fix for a dedicated compute cluster. Just toggle it off in BIOS/UEFI. No more signature verification. This is what most HPC clusters do.

**Option B: sign the module yourself**. Create a Machine Owner Key (MOK), sign the module, enroll the key via `mokutil`, and complete enrollment on reboot at the blue MOK Management screen:

```bash
# Generate a signing key pair
sudo openssl req -x509 -new -nodes -utf8 -sha256 -days 36500 \
    -batch -config x509.genkey -outform DER \
    -out /root/mok-signing.der \
    -keyout /root/mok-signing.priv

# Sign the module
sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 \
    /root/mok-signing.priv \
    /root/mok-signing.der \
    /lib/modules/$(uname -r)/kernel/nvidia-580-open/nvidia-peermem.ko

# Enroll the key in MOK
sudo mokutil --import /root/mok-signing.der
# Reboot → blue screen → Enroll MOK → enter password → reboot
```

But I never got to deploy either fix, because the problem ran deeper.

## Layer 2: testing on the second node

My second node (zgx-3852) already had Secure Boot disabled:

```bash
$ mokutil --sb-state
SecureBoot disabled
Platform is in Setup Mode
```

Great, no signing issue here. The module should just load, right?

```bash
$ sudo modprobe nvidia_peermem
modprobe: ERROR: could not insert 'nvidia_peermem': Invalid argument
```

A *different* error. Not `Key was rejected` (signing), but `Invalid argument` (`EINVAL`). With Secure Boot off, the kernel happily loaded the unsigned module (just tainting itself) but the module's `init` function returned an error.

I used `strace` to confirm the kernel was the one rejecting it:

```bash
$ sudo strace -e trace=finit_module modprobe nvidia_peermem 2>&1
finit_module(3, "", 0) = -1 EINVAL (Invalid argument)
```

The `finit_module` syscall itself returned `EINVAL`. The module loaded into kernel space but its initialization function failed.

And the kernel log was eerily quiet: no error message, no stack trace, nothing beyond:

```
nvidia_peermem: module verification failed: signature and/or required key missing - tainting kernel
```

That taint is just a warning (expected with unsigned modules when Secure Boot is off). But no actual init error was logged. Something in the init function returned `-EINVAL` silently.

## Layer 3: reading the source code

The `nvidia-kernel-source-580-open` package provides the full source. The critical file is `nvidia-peermem.c`:

```bash
$ find /usr/src/nvidia-580.126.09/ -name "*peermem*"
/usr/src/nvidia-580.126.09/nvidia-peermem/nvidia-peermem.c
```

The module's init function (`nv_mem_client_init`) tells the whole story:

```c
static int __init nv_mem_client_init(void)
{
#if defined (NV_MLNX_IB_PEER_MEM_SYMBOLS_PRESENT)
    int rc;

    // ... parameter validation ...
    // ... register as IB peer memory client ...
    // ... all the actual GPUDirect RDMA setup ...

    return rc;
#else
    return -EINVAL;   // ← THIS IS WHAT RUNS
#endif
}
```

The entire functional body of the init function is wrapped in `#if defined(NV_MLNX_IB_PEER_MEM_SYMBOLS_PRESENT)`. If that preprocessor macro isn't defined at build time, the module compiles down to a single `return -EINVAL;`, an empty shell that always fails.

### How that macro gets defined

The NVIDIA build system uses `conftest.sh` to check for the `ib_register_peer_memory_client` and `ib_unregister_peer_memory_client` symbols at compile time:

```bash
# From /usr/src/nvidia-580.126.09/conftest.sh
check_for_ib_peer_memory_symbols() {
    kernel_dir="$1"
    module_symvers="${kernel_dir}/Module.symvers"

    if grep "ib_register_peer_memory_client" "${module_symvers}" > /dev/null 2>&1 &&
       grep "ib_unregister_peer_memory_client" "${module_symvers}" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}
```

It checks for these symbols in:
1. The kernel's own `Module.symvers`
2. MLNX_OFED's `Module.symvers` (at `/usr/src/ofa_kernel/`)
3. DKMS-built MOFED sources

I confirmed: neither the kernel nor any installed package exports these symbols:

```bash
$ grep 'ib_register_peer_memory_client' \
    /usr/src/linux-headers-$(uname -r)/Module.symvers
# (no output, symbol does not exist)

$ ls /usr/src/ofa_kernel/
# ls: cannot access '/usr/src/ofa_kernel/': No such file or directory
```

The `ib_register_peer_memory_client` API is provided by MLNX_OFED (Mellanox's out-of-tree InfiniBand stack). Ubuntu's in-tree `ib_core` module doesn't export it. Without MOFED installed *at compile time*, `nvidia-peermem.ko` is built as a no-op.

I also verified this by checking the compiled binary, and it contains *no* peer memory registration strings whatsoever:

```bash
$ strings /lib/modules/$(uname -r)/kernel/nvidia-580-open/nvidia-peermem.ko \
    | grep 'ib_register_peer_memory'
# (nothing, the code was #ifdef'd out)
```

### Could installing MOFED and rebuilding fix it?

On paper, yes: install MLNX_OFED, then rebuild the NVIDIA driver. The `conftest.sh` would find the symbols, define `NV_MLNX_IB_PEER_MEM_SYMBOLS_PRESENT`, and compile the actual GPUDirect RDMA code.

But there's a catch that makes all of this moot.

## Layer 4: the hardware limitation

While researching the issue, I found the [official NVIDIA FAQ for DGX Spark](https://forums.developer.nvidia.com/t/dgx-spark-gb10-faq/347344):

> **Q: Is GPUDirect RDMA supported on DGX Spark?**
>
> DGX Spark SoC is characterized by a unified memory architecture.
>
> For performance reasons, specifically for CUDA contexts associated to the iGPU, the system memory returned by the pinned device memory allocators (e.g. cudaMalloc) cannot be coherently accessed by the CPU complex nor by I/O peripherals like PCI Express devices.
>
> **Hence the GPUDirect RDMA technology is not supported**, and the mechanisms for direct I/O based on that technology, for example nvidia-peermem (for DOCA-Host), dma-buf or GDRCopy, do not work.

And the [NVIDIA Support KB article #5780](https://nvidia.custhelp.com/app/answers/detail/a_id/5780/) (updated January 2026) says the same thing.

### Why it's architecturally impossible

The GB10 Grace Blackwell Superchip uses a **unified memory architecture**. Unlike discrete GPUs (A100, H100, etc.) that have their own dedicated VRAM connected via PCIe, the GB10's GPU shares system memory with the CPU.

You can see this in `nvidia-smi`:

```
GPU 0000000F:01:00.0
    Product Name:    NVIDIA GB10
    Addressing Mode: ATS              ← Address Translation Services (unified memory)
    FB Memory Usage
        Total:       N/A              ← No dedicated framebuffer
        Used:        N/A
        Free:        N/A
```

GPUDirect RDMA works by giving the Mellanox NIC a *direct DMA path to GPU VRAM*, bypassing the CPU entirely. But when there's no dedicated VRAM, and `cudaMalloc` returns memory that "cannot be coherently accessed by I/O peripherals like PCI Express devices," there's nothing for the NIC to DMA into.

The `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED` CUDA attribute returns `false` on these systems. It's a hardware-level restriction of the SoC design.

### The forum thread that confirms everything

An [NVIDIA Developer Forum post from November 2025](https://forums.developer.nvidia.com/t/gpu-direct-rdma-not-working-on-dgx-spark-systems-nvidia-peermem-module-fails-to-load/349837) describes a user with an identical setup, two DGX Spark systems, 200 Gbps ConnectX-7 link, same `nvidia_peermem` failure. NVIDIA's response pointed to the FAQ above. The thread was marked solved and closed.

## What actually works

Despite GPUDirect RDMA being unsupported, the 200 GbE Mellanox link is far from useless. NCCL (NVIDIA's Collective Communication Library) automatically detects that GPUDirect RDMA is unavailable and falls back to the CPU buffer path:

```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [RO]; OOB enp1s0f0np0:10.0.0.1<0>
NCCL INFO NET/IB : GPU Direct RDMA Disabled for HCA 0 'mlx5_0'
```

This fallback path still uses RDMA over the InfiniBand verbs API, not TCP. The data flow is:

```
GPU → (unified memory) → cudaMemcpy → host buffer → ib_reg_mr → RDMA Send
→ [200 GbE link]
→ RDMA Recv → host buffer → cudaMemcpy → GPU → (unified memory)
```

Users report approximately **25 GB/s** throughput on this path, which is roughly 100 Gbps effective, about half the raw link rate. Not as fast as true GPUDirect RDMA, but still substantial for distributed training.

### NVIDIA's recommended application pattern

For IB verbs applications, NVIDIA recommends:

1. **Allocate communication buffers** with `cudaHostAlloc` (pinned host memory, accessible to both GPU and NIC)
2. **Register them** with `ib_reg_mr` (InfiniBand memory registration)
3. **Query capabilities first** via `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED` and `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORT` to choose the right code path

## Layer 5: the PCIe Gen5 x4 bottleneck

Even without the GPUDirect RDMA limitation, there's another bandwidth constraint worth understanding: the ConnectX-7 NICs are connected to the GB10 SoC via PCIe Gen5 **x4**, not the x16 lanes these cards normally get in server-class hardware.

### Diagnosing the PCIe link

```bash
$ sudo lspci -vvv -s 0000:01:00.0 | grep -E "LnkCap:|LnkSta:"
        LnkCap: Port #0, Speed 32GT/s, Width x4, ASPM not supported
        LnkSta: Speed 32GT/s, Width x4
```

The ConnectX-7 (MT2910) supports up to PCIe 5.0 x16, that's 504 Gbps per direction. But the GB10 SoC only allocates **x4 lanes** per PCIe root port. That gives:

$$\text{PCIe Gen5 x4} = 32 \text{ GT/s} \times 4 \text{ lanes} \times \frac{128}{130} \approx 126 \text{ Gbps per direction} \approx 15.75 \text{ GB/s}$$

Meanwhile, each ConnectX-7 port negotiates at 200 Gbps on the wire:

```bash
$ ethtool enp1s0f1np1 | grep Speed
        Speed: 200000Mb/s

$ cat /sys/class/infiniband/rocep1s0f1/ports/1/rate
200 Gb/sec (4X HDR)
```

So even though the Ethernet link is 200 Gbps, the PCIe can only feed it at ~126 Gbps, a **37% bottleneck** before the data even hits the wire.

### Two ConnectX-7 cards, two active ports

Each DGX Spark node actually has **two** ConnectX-7 cards on separate PCIe domains:

```bash
$ lspci | grep -i mellanox
0000:01:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
0000:01:00.1 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
0002:01:00.0 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
0002:01:00.1 Ethernet controller: Mellanox Technologies MT2910 Family [ConnectX-7]
```

Each card is dual-port, and one port per card is active at 200G:

```bash
$ for dev in rocep1s0f0 rocep1s0f1 roceP2p1s0f0 roceP2p1s0f1; do
    printf "%-16s " "$dev"
    paste -d' ' /sys/class/infiniband/$dev/ports/1/{rate,state}
  done
rocep1s0f0       40 Gb/sec (4X QDR)  1: DOWN
rocep1s0f1       200 Gb/sec (4X HDR) 4: ACTIVE
roceP2p1s0f0     40 Gb/sec (4X QDR)  1: DOWN
roceP2p1s0f1     200 Gb/sec (4X HDR) 4: ACTIVE
```

With both cards active, the aggregate bandwidth is:

| | Per Card | Both Cards |
|---|---|---|
| **Wire (Ethernet)** | 200 Gbps | 400 Gbps |
| **PCIe (Gen5 x4)** | ~126 Gbps | ~252 Gbps |
| **Effective limit** | ~126 Gbps (~15.75 GB/s) | ~252 Gbps (~31.5 GB/s) |

The PCIe x4 width is a design choice by NVIDIA for the GB10 SoC. The compact form factor simply doesn't have the pin count or power budget for x16 lanes to each peripheral.

### Does this matter for LLM inference?

For most LLM serving workloads across two nodes, the PCIe bottleneck is **not the limiting factor**. Here's why:

**Token generation (autoregressive decoding)** is memory-bandwidth bound, not network-bound. Each generated token requires an allreduce across nodes, but the data volume is tiny: for a 70B model with `hidden_dim=8192`, that's just `8192 × 2 bytes = 16 KB` per transformer layer. At ~15 GB/s per link, 16 KB transfers complete in about 1 microsecond. The bottleneck is reading KV cache from memory, not the interconnect.

**Prompt prefill** is more communication-intensive. Processing a 4096-token prompt with tensor parallelism sends `batch × seq_len × hidden_dim × 2 bytes` per layer, roughly 64 MB per layer for a 70B model. At ~15 GB/s, that's ~4 ms per layer, which adds up across 80 layers but is still dwarfed by the compute time on the Blackwell GPU.

**The real bottleneck is the CPU bounce buffer**, not PCIe width. Since GPUDirect RDMA is unavailable (Layer 4), NCCL must copy data through host memory:

```
GPU ↔ unified memory → cudaMemcpy → host buffer → RDMA → wire
```

This extra copy step and address translation overhead reduces effective throughput to roughly **12–15 GB/s** in practice, well below even the PCIe Gen5 x4 ceiling. So the PCIe width becomes irrelevant because the software path saturates before the hardware link does.

**Bottom line:** for two-node LLM inference serving, the ~12–15 GB/s effective throughput from the NCCL fallback path is sufficient. The interconnect adds a few milliseconds of latency per layer during prefill, but token generation, the latency-critical path, is dominated by memory bandwidth. You'd need much larger clusters (4+ nodes) or extremely latency-sensitive workloads for the PCIe bottleneck to matter more than the missing GPUDirect RDMA.

## Summary of the three failure layers

| Layer | Node | Error | Root Cause |
|-------|------|-------|------------|
| **1. Secure Boot** | zgx-3285 (SB enabled) | `Key was rejected by service` | Packaging inconsistency: `nvidia-peermem.ko` ships signed by "NVIDIA Support" while the other 4 NVIDIA modules in the same package are re-signed by Canonical. |
| **2. Build Flags** | zgx-3852 (SB disabled) | `Invalid argument` (EINVAL) | Module compiled without `NV_MLNX_IB_PEER_MEM_SYMBOLS_PRESENT` because MLNX_OFED wasn't installed at build time. Init function is a no-op `return -EINVAL`. |
| **3. Architecture** | Both nodes | N/A (fundamental) | GB10 Grace Blackwell SoC uses unified memory. `cudaMalloc` memory cannot be coherently accessed by PCIe devices. GPUDirect RDMA is architecturally unsupported. |

Even if you fix layers 1 and 2 (disable Secure Boot + install MOFED + rebuild the module), layer 3 is an immovable hardware constraint. The `nvidia_peermem` module would load but the underlying `nvidia_p2p_get_pages` kernel API would fail because the GB10 doesn't support peer-to-peer DMA to its memory.

## Lessons learned

1. **Always check `modinfo -F signer`** when `modprobe` returns `Key was rejected by service`. Compare against `mokutil --list-enrolled` to find the mismatch.

2. **`EINVAL` from `modprobe` with no kernel log** usually means the module's `__init` function returned an error silently. Read the source, especially `#ifdef` guards around the entire init body.

3. **Pre-built `.ko` files are compile-time snapshots.** A module compiled without MOFED symbols will *never* work, regardless of what you install later. You must rebuild the module after installing the dependencies.

4. **Unified memory SoCs change the rules.** GPUDirect RDMA, GDRCopy, and dma-buf all assume dedicated GPU VRAM accessible via PCIe BAR mappings. The Grace Blackwell GB10's unified memory architecture breaks this assumption at the hardware level.

5. **NCCL's fallback path is not catastrophic.** The CPU bounce buffer path over InfiniBand still delivers ~25 GB/s on DGX Spark, enough for productive multi-node training, just not at the theoretical 50 GB/s of true GPUDirect RDMA.

6. **Read the official FAQ before debugging.** NVIDIA documented this limitation months ago. I could have saved hours of kernel investigation if I'd checked the [DGX Spark FAQ](https://forums.developer.nvidia.com/t/dgx-spark-gb10-faq/347344) first.

---

*Investigation performed on two HP ZGX Nano G1n AI Stations (DGX Spark), running Ubuntu 24.04 (DGX OS), kernel 6.17.0-1008-nvidia (aarch64), NVIDIA driver 580.126.09, CUDA 13.0, ConnectX-7 200GbE QSFP link. February 2026.*
