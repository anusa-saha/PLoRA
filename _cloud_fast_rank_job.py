import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

RUN_DIR = Path('/tmp/jt_run.rankbudgets.fast')
RUN_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(RUN_DIR)

CACHE_ROOT = RUN_DIR / 'hf_cache'
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(CACHE_ROOT)
os.environ['HF_HUB_CACHE'] = str(CACHE_ROOT / 'hub')
os.environ['HF_DATASETS_CACHE'] = str(CACHE_ROOT / 'datasets')
os.environ['DATASETS_CACHE'] = str(CACHE_ROOT / 'datasets')
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_ROOT / 'transformers')
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CACHE_ROOT / 'hub')
os.environ['XDG_CACHE_HOME'] = str(CACHE_ROOT / 'xdg')
os.environ['TMPDIR'] = str(RUN_DIR / 'tmp')
Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LANGUAGES = {
    'hin_Deva': 'Hindi',
    'fra_Latn': 'French',
    'cmn_Hans': 'Chinese',
    'urd_Arab': 'Urdu',
    'eng_Latn': 'English',
    'awa_Deva': 'Awadhi',
    'ben_Beng': 'Bengali',
    'mar_Deva': 'Marathi',
    'nld_Latn': 'Dutch',
    'pol_Latn': 'Polish',
    'snd_Arab': 'Sindhi',
    'azb_Arab': 'South Azerbaijani',
}

FLORES_TO_OPUS = {
    'eng_Latn':'en','fra_Latn':'fr','cmn_Hans':'zh','urd_Arab':'ur','hin_Deva':'hi',
    'ben_Beng':'bn','mar_Deva':'mr','nld_Latn':'nl','pol_Latn':'pl','snd_Arab':'sd',
    'awa_Deva':'awa','azb_Arab':'azb'
}

MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'
MAX_LENGTH = 512
N_SAMPLES = 60
N_SUBSAMPLES = 1
SUBSAMPLE_SIZE = 40
KMIN_FRAC = 0.10
KMAX_FRAC = 0.90
BATCH_SIZE = 1
SEED = 42

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


def run(cmd):
    print('>>>', ' '.join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, env=os.environ.copy())
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def compute_power_law_alpha(activations: torch.Tensor, kmin_frac=KMIN_FRAC, kmax_frac=KMAX_FRAC) -> float:
    svd_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    centered = activations.to(svd_device, dtype=torch.float32)
    centered = centered - centered.mean(dim=0, keepdim=True)
    try:
        S = torch.linalg.svdvals(centered).cpu().numpy()
    except Exception:
        return 0.0

    S = S[S > 1e-10]
    r = len(S)
    if r < 4:
        return 0.0

    kmin = max(1, int(kmin_frac * r))
    kmax = min(r, int(kmax_frac * r))
    if kmax <= kmin:
        kmax = kmin + 1

    tail_ranks = np.arange(kmin, kmax + 1)
    tail_sigmas = S[kmin - 1 : kmax]
    log_k = np.log(tail_ranks)
    log_s = np.log(tail_sigmas + 1e-12)
    slope, _ = np.polyfit(log_k, log_s, 1)
    beta = -slope
    if beta < 1e-6:
        return 0.0
    return float(1.0 / beta)


print('Installing runtime deps...')
run(['/opt/miniconda/envs/SFT/bin/python','-m','pip','install','-q','transformers','datasets','accelerate','pandas','numpy','matplotlib','seaborn','scipy','pot','jupyter','nbconvert'])

print('Loading multilingual datasets...')
opus_cfg = set(get_dataset_config_names('Helsinki-NLP/opus-100'))
language_data = {}
dataset_used = {}

for lc, lname in LANGUAGES.items():
    try:
        iso = FLORES_TO_OPUS[lc]
        if iso == 'en':
            config, field = 'en-fr', 'en'
        elif f'{iso}-en' in opus_cfg:
            config, field = f'{iso}-en', iso
        elif f'en-{iso}' in opus_cfg:
            config, field = f'en-{iso}', iso
        else:
            print(f'SKIP {lname}: missing OPUS config for {lc}/{iso}', flush=True)
            continue

        ds = load_dataset('Helsinki-NLP/opus-100', config, split='train')
        texts = []
        for row in ds:
            t = row.get('translation', {}).get(field, '')
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
            if len(texts) >= N_SAMPLES:
                break

        if len(texts) < 30:
            print(f'SKIP {lname}: only {len(texts)} samples', flush=True)
            continue

        language_data[lc] = texts
        dataset_used[lc] = f'Helsinki-NLP/opus-100::{config}'
        print(f'Loaded {lname}: {len(texts)} from {config}', flush=True)
    except Exception as e:
        print(f'SKIP {lname}: {e}', flush=True)

if len(language_data) < 3:
    raise SystemExit(f'Not enough languages loaded: {len(language_data)}')

print('Loading model...')

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map='auto',
    low_cpu_mem_usage=True,
)
model.eval()

n_layers = model.config.num_hidden_layers
device = next(model.parameters()).device
print(f'Model loaded: {MODEL_ID} | layers={n_layers} | device={device}', flush=True)

language_activations = {}

for lc, texts in language_data.items():
    print(f'Extracting activations: {lc} ({len(texts)} samples)', flush=True)
    all_acts = {i: [] for i in range(n_layers)}

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        mask = attention_mask.unsqueeze(-1).to(out.hidden_states[1].dtype)
        denom = mask.sum(dim=1).clamp(min=1)

        for ell in range(n_layers):
            hs = out.hidden_states[ell + 1]
            mean_act = (hs * mask).sum(dim=1) / denom
            all_acts[ell].append(mean_act.detach().cpu())

        if i % 20 == 0:
            print(f'  {lc}: {i}/{len(texts)}', flush=True)

        del out, input_ids, attention_mask, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    language_activations[lc] = {k: torch.cat(v, dim=0) for k, v in all_acts.items()}

print('Computing alpha statistics...')
alpha_scores = {lc: [] for lc in language_activations}
for lc in language_activations:
    for ell in range(n_layers):
        alpha = compute_power_law_alpha(language_activations[lc][ell])
        alpha_scores[lc].append(alpha)

alpha_mean, alpha_std = {}, {}
for lc in alpha_scores:
    alpha_mean[lc], alpha_std[lc] = [], []
    for ell in range(n_layers):
        layer_acts = language_activations[lc][ell]
        n = layer_acts.shape[0]
        ss = min(SUBSAMPLE_SIZE, n)
        vals = []
        for _ in range(N_SUBSAMPLES):
            idx = np.random.choice(n, size=ss, replace=False)
            vals.append(compute_power_law_alpha(layer_acts[idx]))
        alpha_mean[lc].append(float(np.mean(vals)))
        alpha_std[lc].append(float(np.std(vals)))

lang_codes = sorted(alpha_mean.keys())
A = np.array([alpha_mean[lc] for lc in lang_codes])
mu = A.mean(axis=0)
sigma = A.std(axis=0) + 1e-8

d_tilde = {}
for lc in lang_codes:
    alphas = np.array(alpha_mean[lc])
    d_tilde[lc] = (np.abs(alphas - mu) / sigma).tolist()

results = {'layer': list(range(n_layers))}
for lc in lang_codes:
    name = LANGUAGES[lc]
    results[f'{name}_alpha_mean'] = alpha_mean[lc]
    results[f'{name}_alpha_std'] = alpha_std[lc]
    results[f'{name}_d_tilde'] = d_tilde[lc]
results['mu'] = mu.tolist()
results['sigma'] = sigma.tolist()

df = pd.DataFrame(results)
csv_path = RUN_DIR / 'spinal_plora_qwen3_4b_results_12lang.csv'
df.to_csv(csv_path, index=False)
print(f'Saved {csv_path}', flush=True)

Path('/content').mkdir(parents=True, exist_ok=True)
shutil.copy2(csv_path, '/content/spinal_plora_qwen3_4b_results_12lang.csv')

print('Running Step 4 notebook...')
run([
    '/opt/miniconda/envs/SFT/bin/python', '-m', 'jupyter', 'nbconvert',
    '--to', 'notebook', '--execute', 'plora_step4_rank_budgeting_SN.ipynb',
    '--output', 'plora_step4_rank_budgeting_SN.executed.ipynb',
    '--ExecutePreprocessor.timeout=-1'
])

out_json = RUN_DIR / 'plora_step4_rank_budgets.json'
out_csv = RUN_DIR / 'plora_step4_rank_budgets.csv'
out_sum = RUN_DIR / 'plora_step4_param_summary.csv'

result = {
    'ok': out_json.exists(),
    'rank_budget_json': str(out_json),
    'rank_budget_csv': str(out_csv),
    'summary_csv': str(out_sum),
    'languages_used': lang_codes,
    'datasets_used': dataset_used,
    'n_samples_per_lang': N_SAMPLES,
}

with open(RUN_DIR / 'result.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2), flush=True)
