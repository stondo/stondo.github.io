"""Render the measured comparison and a terminal-style LinkedIn card.

Run with: uv run --no-project --python 3.13 --with matplotlib --with pillow \
  python scripts/flash_next_quant_figures.py
Data: static/benchmarks/qwen38-strata-5090-20261105/measurements.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT/'static/benchmarks/qwen38-strata-5090-20261105/measurements.json').read_text())
a, b = DATA['models']['iq4-v31'], DATA['models']['nvfp4']
BG, FG, MUTED, CYAN, GREEN = '#0c1117', '#e1e7ed', '#8b959f', '#5ed8ef', '#35c34b'

plt.rcParams.update({'font.family':'DejaVu Sans', 'text.color':FG, 'axes.labelcolor':MUTED,
                     'xtick.color':MUTED, 'ytick.color':FG, 'axes.edgecolor':'#343c45'})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), facecolor=BG)
for ax, values, title, xlabel, limit in [
    (axes[0], [a['short_prompt_decode_tps'], b['short_prompt_decode_tps']],
     'Short-prompt generation', 'tokens / second', 210),
    (axes[1], [a['minimum_available_ram_gib'], b['minimum_available_ram_gib']],
     'Minimum available system RAM', 'GiB', 18)]:
    ax.set_facecolor(BG)
    ax.barh([1, 0], values, height=.42, color=[CYAN, '#566471'])
    ax.set_yticks([1,0], ['IQ4 v3.1', 'NVFP4'])
    ax.set_xlim(0,limit); ax.set_ylim(-.65,1.65)
    ax.set_xlabel(xlabel);ax.set_title(title, fontsize=13, color=FG, pad=18)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    for y, value in zip([1,0],values):
        ax.text(value+limit*.025,y,f'{value:.1f}',va='center',fontsize=12,color=FG)
fig.suptitle('Flash-Next on one RTX 5090', x=.07, ha='left', fontsize=21, fontweight='bold')
fig.text(.07,.83,'96 GB RAM · Strata engine 0.1.57 · 450 W limit · native 262,144 context', color=MUTED,fontsize=11)
fig.text(.07,.06,'Small local suite. Decode: median of 3 short-prompt runs, not full-context speed.\nRAM: sampled system-wide MemAvailable. Core checks and HumanEval subset tied.',color=MUTED,fontsize=9)
fig.subplots_adjust(left=.14,right=.94,top=.70,bottom=.25,wspace=.50)
fig.savefig(ROOT/'static/qwen38-strata-5090-benchmarks.png',dpi=170,facecolor=BG)
plt.close(fig)

font_path = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans Mono'))
font = lambda size: ImageFont.truetype(font_path,size)
im=Image.new('RGB',(1200,1500),BG);d=ImageDraw.Draw(im)
d.rounded_rectangle((60,90,1140,1330),radius=18,fill=BG,outline='#343c45',width=2)
d.rounded_rectangle((60,90,1140,154),radius=18,fill='#171d24')
d.rectangle((60,123,1140,154),fill='#171d24')
for x,c in [(96,'#ff5f56'),(136,'#ffbd2e'),(176,'#27c93f')]:d.ellipse((x-10,112,x+10,132),fill=c)

def text(x,y,value,size=26,color=FG):
    f=font(size)
    assert d.textbbox((x,y),value,font=f)[2] <= 1120, value
    d.text((x,y),value,font=f,fill=color)

text(220,105,'5090 / strata 0.1.57 / 450 W',24,MUTED)
text(96,193,'# Qwen3.8-Flash-Next | 32 GB VRAM | 96 GB RAM',23,MUTED)
text(96,253,'Flash-Next: the quant I kept',39)
text(96,335,'# native context: 262,144 tokens',29,GREEN)
text(540,410,'IQ4 v3.1',28,CYAN);text(865,410,'NVFP4',28,MUTED)
rows=[
    (490,'decode (short)',f"{a['short_prompt_decode_tps']:.1f} t/s",f"{b['short_prompt_decode_tps']:.1f} t/s"),
    (570,'RAM available (min)',f"{a['minimum_available_ram_gib']:.1f} GiB",f"{b['minimum_available_ram_gib']:.1f} GiB"),
    (650,'HumanEval subset','31/32','31/32'),
    (730,'direct / thinking','16/16 each','16/16 each'),
    (810,'focused reruns','6/6','5/6')]
for y,label,left,right in rows:
    text(96,y,label,25);text(540,y,left,27,CYAN);text(865,y,right,25,MUTED)
d.line((96,884,1104,884),fill='#343c45',width=2)
text(96,927,'# recall probe: 256,553 prompt tokens',28,GREEN)
text(96,984,'3/3 markers recalled: BOTH',29,CYAN)
text(96,1030,'first token: 40.84 s / 41.37 s',25,MUTED)
text(96,1107,'selected: Unsloth UD-IQ4_XS-v31',29,GREEN)
text(96,1185,'# small local tests; no general quality ranking',21,MUTED)
text(96,1221,'# 32/164 HumanEval; one seed for focused reruns',21,MUTED)
text(96,1257,'# native 256K; 512K not certified',21,MUTED)
d.line((60,1375,1140,1375),fill='#343c45',width=2)
text(60,1410,'stondo.github.io  /  Bits-Entangled',26)
text(60,1453,'one GPU · a versioned experiment · a usable desktop',22,MUTED)
im.save(ROOT/'static/qwen38-strata-5090-card.png',optimize=True)
print('Saved the chart and LinkedIn terminal card.')
