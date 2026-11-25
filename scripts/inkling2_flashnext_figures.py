"""Render the measured comparison and a terminal-style LinkedIn card.

Run with: uv run --no-project --python 3.13 --with matplotlib --with pillow \
  python scripts/inkling2_flashnext_figures.py [OUTPUT_DIR]
Data: static/benchmarks/inkling2-flashnext-basalt-20261125/measurements.json
Without OUTPUT_DIR the two PNG files go to static/ beside the earlier benchmark figures.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'static/benchmarks/inkling2-flashnext-basalt-20261125/measurements.json').read_text())
FLASH = DATA['models']['flash-next-iq4xs']
INK = DATA['models']['inkling-2-mini']
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'static'
BG, FG, MUTED, CYAN, GREEN = '#0c1117', '#e1e7ed', '#8b959f', '#5ed8ef', '#35c34b'

fs = FLASH['speed_on_basalt_2026_11_25']
is_ = INK['speed_on_basalt_median_of_visits']
ink_rec = INK['recommended_sampling_budget_16384']
ram_flash = FLASH['context_256k']['minimum_available_ram_gib']
ram_ink = ink_rec['context_256k_fp16_kv']['minimum_available_ram_gib']   # the fp16 cache, the setting the post keeps
ram_back = round(ram_ink - ram_flash)
on = INK['sampling_runs']['core_thinking_on']['scores']
he = INK['sampling_runs']['humaneval_subset']['scores']
off_tokens = INK['sampling_runs']['core_thinking_off']['completion_tokens_per_run']
ratio = [t / FLASH['core_thinking_off']['completion_tokens'] for t in off_tokens]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': FG, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': FG, 'axes.edgecolor': '#343c45'})
fig, axes = plt.subplots(1, 3, figsize=(15, 5.1), facecolor=BG)
for ax, values, title, xlabel, limit, fmt, step in [
    (axes[0], [fs['short_prompt_decode_tps_median'], is_['short_prompt_decode_tps']],
     'Generation, prompt under 100 tokens', 'tokens / second', 400, '{:.0f}', 50),
    (axes[1], [fs['prefill_16k_tps_median'], is_['prefill_16k_tps']],
     'Prompt read, one long text', 'tokens / second', 20000, '{:,.0f}', 5000),
    (axes[2], [ram_flash, ram_ink], 'RAM free during the 256K check', 'GiB available', 84, '{:.1f}', 20),
]:
    ax.set_facecolor(BG)
    ax.barh([1, 0], values, height=.42, color=[CYAN, GREEN])
    ax.set_yticks([1, 0], ['Flash-Next', 'Inkling 2 Mini'])
    ax.set_xlim(0, limit); ax.set_ylim(-.65, 1.65)
    ax.set_xticks(range(0, limit + 1, step), ['{:,}'.format(v) for v in range(0, limit + 1, step)])
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=13, color=FG, pad=18)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=12)
    for y, value in zip([1, 0], values):
        ax.text(value + limit * .025, y, fmt.format(value), va='center', fontsize=13, color=FG)
fig.suptitle('Inkling 2 Mini against Qwen3.8-Flash-Next on one RTX 5090', x=.055, ha='left', fontsize=21, fontweight='bold')
fig.text(.055, .845, '96 GB RAM · RTX 5090 held to 450 W · speed measured on Basalt 0.5.7 on the same day', color=MUTED, fontsize=11)
fig.text(.055, .05,
         'Generation: median of three 512-token runs per round; Inkling is the median of four rounds. Short prompts, not a full context.\n'
         f'The same prompt text is {fs["prefill_16k_tokens_median"]:,} tokens to Flash-Next and {is_["prefill_16k_tokens"]:,.0f} to Inkling. '
         'RAM is system-wide free memory, desktop included.\n'
         "Flash-Next's RAM figure is from its run of November 5 on Strata, with the same weights.", color=MUTED, fontsize=10)
fig.subplots_adjust(left=.10, right=.975, top=.70, bottom=.27, wspace=.42)
fig.savefig(OUT / 'inkling2-flashnext-benchmarks.png', dpi=170, facecolor=BG)
plt.close(fig)

font_path = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans Mono'))
font = lambda size: ImageFont.truetype(font_path, size)
im = Image.new('RGB', (1200, 1500), BG); d = ImageDraw.Draw(im)
d.rounded_rectangle((60, 90, 1140, 1330), radius=18, fill=BG, outline='#343c45', width=2)
d.rounded_rectangle((60, 90, 1140, 154), radius=18, fill='#171d24')
d.rectangle((60, 123, 1140, 154), fill='#171d24')
for x, c in [(96, '#ff5f56'), (136, '#ffbd2e'), (176, '#27c93f')]:
    d.ellipse((x - 10, 112, x + 10, 132), fill=c)


def text(x, y, value, size=26, color=FG):
    f = font(size)
    assert d.textbbox((x, y), value, font=f)[2] <= 1120, value
    d.text((x, y), value, font=f, fill=color)


text(220, 105, '5090 / basalt 0.5.7 / 450 W', 24, MUTED)
text(96, 193, '# Inkling 2 Mini (26B MoE, 5.4B active) vs Flash-Next', 23, MUTED)
text(96, 250, 'Inkling 2 Mini nearly matches Flash-Next', 34)
text(96, 292, f'on my tests, and frees {ram_back} GiB of RAM', 34)
text(96, 370, '# Inkling at temperature 1.0, its publisher settings: 3 runs', 26, GREEN)
text(96, 406, '# scores below are the range over those runs', 26, GREEN)
text(600, 478, 'Flash-Next', 28, CYAN); text(860, 478, 'Inkling 2 Mini', 28, CYAN)
rows = [
    (540, 'decode, short prompt', f"{fs['short_prompt_decode_tps_median']:.0f} t/s", f"{is_['short_prompt_decode_tps']:.0f} t/s"),
    (600, 'prompt read, one long text', f"{fs['prefill_16k_tps_median']:,.0f} t/s", f"{is_['prefill_16k_tps']:,.0f} t/s"),
    (660, 'RAM free, 256K check', f"{ram_flash:.1f} GiB", f"{ram_ink:.1f} GiB"),
    (720, 'core cases, thinking on', '16/16', f"{min(on)}-{max(on)}/16"),
    (780, 'HumanEval subset', '31/32', f"{min(he)}-{max(he)}/32"),
    (840, 'GSM8K, first 200 problems', f"{FLASH['gsm8k_first_200_passed']}/200", f"{INK['gsm8k_first_200']['passed']}/200"),
]
for y, label, left, right in rows:
    text(96, y, label, 25); text(600, y, left, 26, CYAN); text(860, y, right, 25, GREEN)
d.line((96, 900, 1104, 900), fill='#343c45', width=2)
text(96, 936, '# recall near 250K tokens: all 3 markers, both models', 26, CYAN)
text(96, 992, f"First token: {FLASH['context_256k']['ttft_seconds']:.1f} s vs {ink_rec['context_256k_fp16_kv']['ttft_seconds']:.1f} s", 27)
text(96, 1066, 'Inkling cannot switch thinking off:', 27)
text(96, 1102, f"{min(ratio):.0f} to {max(ratio):.0f}x the tokens where Flash-Next answers at once", 27)
text(96, 1170, '# small local suite. One seeded sample per case and run.', 21, MUTED)
text(96, 1206, '# It is slower where Flash-Next would not think.', 21, MUTED)
text(96, 1242, '# Keep: Flash-Next as default, Inkling as the second model.', 21, MUTED)
d.line((60, 1375, 1140, 1375), fill='#343c45', width=2)
text(60, 1410, 'stondo.github.io  /  Bits-Entangled', 26)
text(60, 1453, 'one GPU · Inkling sampled three times · ranges, not a ranking', 22, MUTED)
im.save(OUT / 'inkling2-flashnext-card.png', optimize=True)
print('Saved the chart and LinkedIn terminal card to', OUT)
