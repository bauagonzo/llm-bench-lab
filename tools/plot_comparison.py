#!/usr/bin/env python3
"""Generate comparison charts: PRO 6000 vs 5070 Ti."""

import json
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/..'
OUT_DIR = f'{RESULTS_DIR}/results/psyche-suse/5070ti'

MODELS = ['gemma-3-1b', 'llama-3.2-1b', 'phi-4-mini-3.8b', 'ministral-8b', 'gemma-3-12b', 'mistral-nemo-12b']
LABELS = ['Gemma 3\n1B', 'Llama 3.2\n1B', 'Phi-4 Mini\n3.8B', 'Ministral\n8B', 'Gemma 3\n12B', 'Mistral Nemo\n12B']
TI_MAP = {'gemma-3-1b':'gemma3-1b', 'llama-3.2-1b':'llama32-1b', 'phi-4-mini-3.8b':'phi4-mini-3.8b',
          'ministral-8b':'ministral-8b', 'gemma-3-12b':'gemma3-12b', 'mistral-nemo-12b':'mistral-nemo-12b'}

def extract_localscore(path):
    d = json.load(open(path))
    if isinstance(d, list):
        pp = max((x.get('avg_ts',0) for x in d if x.get('n_prompt',0) >= 512 and x.get('n_gen',0) <= 128), default=0)
        tg = max((x.get('avg_ts',0) for x in d if x.get('n_gen',0) >= 128), default=0)
    else:
        r = d['results']
        pp = max((x['prompt_tps'] for x in r if x['n_prompt'] >= 1024), default=0)
        tg = max((x['gen_tps'] for x in r if x['n_gen'] >= 1024), default=0)
    return pp, tg

# Gather data
pro_cuda = {}
for m in MODELS:
    best_pp, best_tg = 0, 0
    for run in ['run-1','run-2','run-3']:
        f = f'{RESULTS_DIR}/results/psyche-suse/2026-02-11/scaling-test/{run}/{m}-cuda13.json'
        try:
            pp, tg = extract_localscore(f)
            best_pp = max(best_pp, pp)
            best_tg = max(best_tg, tg)
        except: pass
    pro_cuda[m] = (best_pp, best_tg)

ti_vulkan, ti_cuda = {}, {}
for m in MODELS:
    t = TI_MAP[m]
    ti_vulkan[m] = extract_localscore(f'{RESULTS_DIR}/results/psyche-suse/5070ti/run-1/{t}-vulkan.json')
    ti_cuda[m] = extract_localscore(f'{RESULTS_DIR}/results/psyche-suse/5070ti/run-1/{t}-cuda13.json')

os.makedirs(OUT_DIR, exist_ok=True)

# Colors
C_PRO = '#1f77b4'
C_TI_VK = '#ff7f0e'
C_TI_CU = '#2ca02c'

x = np.arange(len(MODELS))
width = 0.25

# --- Chart 1: Prompt Processing ---
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, [pro_cuda[m][0] for m in MODELS], width, label='PRO 6000 (CUDA)', color=C_PRO)
bars2 = ax.bar(x, [ti_vulkan[m][0] for m in MODELS], width, label='5070 Ti (Vulkan)', color=C_TI_VK)
bars3 = ax.bar(x + width, [ti_cuda[m][0] for m in MODELS], width, label='5070 Ti (CUDA)', color=C_TI_CU)

ax.set_ylabel('Tokens/second', fontsize=12)
ax.set_title('Prompt Processing: RTX PRO 6000 (96GB) vs RTX 5070 Ti (16GB)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add ratio labels on PRO bars
for i, m in enumerate(MODELS):
    best_ti = max(ti_vulkan[m][0], ti_cuda[m][0])
    ratio = pro_cuda[m][0] / best_ti if best_ti else 0
    ax.text(i - width, pro_cuda[m][0] + 500, f'{ratio:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_PRO)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/comparison-pp.png', dpi=150)
print(f'Saved: {OUT_DIR}/comparison-pp.png')

# --- Chart 2: Token Generation ---
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, [pro_cuda[m][1] for m in MODELS], width, label='PRO 6000 (CUDA)', color=C_PRO)
bars2 = ax.bar(x, [ti_vulkan[m][1] for m in MODELS], width, label='5070 Ti (Vulkan)', color=C_TI_VK)
bars3 = ax.bar(x + width, [ti_cuda[m][1] for m in MODELS], width, label='5070 Ti (CUDA)', color=C_TI_CU)

ax.set_ylabel('Tokens/second', fontsize=12)
ax.set_title('Token Generation: RTX PRO 6000 (96GB) vs RTX 5070 Ti (16GB)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for i, m in enumerate(MODELS):
    best_ti = max(ti_vulkan[m][1], ti_cuda[m][1])
    ratio = pro_cuda[m][1] / best_ti if best_ti else 0
    ax.text(i - width, pro_cuda[m][1] + 5, f'{ratio:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_PRO)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/comparison-tg.png', dpi=150)
print(f'Saved: {OUT_DIR}/comparison-tg.png')

# --- Chart 3: Perf per dollar (estimated) ---
# PRO 6000 ~$5,000, 5070 Ti ~$750
PRO_PRICE = 10000
TI_PRICE = 950

fig, ax = plt.subplots(figsize=(12, 6))
pro_ppd = [pro_cuda[m][1] / PRO_PRICE * 1000 for m in MODELS]
ti_best_ppd = [max(ti_vulkan[m][1], ti_cuda[m][1]) / TI_PRICE * 1000 for m in MODELS]

bars1 = ax.bar(x - 0.15, pro_ppd, 0.3, label=f'PRO 6000 (${PRO_PRICE:,})', color=C_PRO)
bars2 = ax.bar(x + 0.15, ti_best_ppd, 0.3, label=f'5070 Ti (${TI_PRICE:,})', color=C_TI_VK)

ax.set_ylabel('TG tokens/s per $1,000', fontsize=12)
ax.set_title('Token Generation per Dollar: PRO 6000 vs 5070 Ti', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for i in range(len(MODELS)):
    ratio = ti_best_ppd[i] / pro_ppd[i] if pro_ppd[i] else 0
    ax.text(i + 0.15, ti_best_ppd[i] + 2, f'{ratio:.0f}x', ha='center', va='bottom', fontsize=9, fontweight='bold', color=C_TI_VK)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/comparison-perf-per-dollar.png', dpi=150)
print(f'Saved: {OUT_DIR}/comparison-perf-per-dollar.png')

# --- Chart 4: 5070 Ti Vulkan vs CUDA ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

vk_pp = [ti_vulkan[m][0] for m in MODELS]
cu_pp = [ti_cuda[m][0] for m in MODELS]
vk_tg = [ti_vulkan[m][1] for m in MODELS]
cu_tg = [ti_cuda[m][1] for m in MODELS]

ax1.bar(x - 0.15, vk_pp, 0.3, label='Vulkan', color=C_TI_VK)
ax1.bar(x + 0.15, cu_pp, 0.3, label='CUDA', color=C_TI_CU)
ax1.set_ylabel('Tokens/second')
ax1.set_title('5070 Ti: Prompt Processing')
ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, fontsize=9)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2.bar(x - 0.15, vk_tg, 0.3, label='Vulkan', color=C_TI_VK)
ax2.bar(x + 0.15, cu_tg, 0.3, label='CUDA', color=C_TI_CU)
ax2.set_ylabel('Tokens/second')
ax2.set_title('5070 Ti: Token Generation')
ax2.set_xticks(x)
ax2.set_xticklabels(LABELS, fontsize=9)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('RTX 5070 Ti: Vulkan vs CUDA 13.1', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/5070ti-vulkan-vs-cuda.png', dpi=150)
print(f'Saved: {OUT_DIR}/5070ti-vulkan-vs-cuda.png')

print('\nDone. 4 charts generated.')
