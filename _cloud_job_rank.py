import json, os, shutil, subprocess, sys
from pathlib import Path
import nbformat

run_dir = Path('/tmp/jt_run.rankbudgets.bbdb891e')
os.chdir(run_dir)
py = '/opt/miniconda/envs/SFT/bin/python'
cache_root = run_dir / 'hf_cache'
cache_root.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env['HF_HOME'] = str(cache_root)
env['HF_HUB_CACHE'] = str(cache_root / 'hub')
env['HF_DATASETS_CACHE'] = str(cache_root / 'datasets')
env['TRANSFORMERS_CACHE'] = str(cache_root / 'transformers')
env['HUGGINGFACE_HUB_CACHE'] = str(cache_root / 'hub')
env['XDG_CACHE_HOME'] = str(cache_root / 'xdg')
env['TMPDIR'] = str(run_dir / 'tmp')
Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)

def run(cmd):
    print('>>>', ' '.join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def patch_step12_notebook(path: Path):
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.get('cell_type') != 'code':
            continue
        src = cell.get('source', '')

        if 'from kaggle_secrets import UserSecretsClient' in src:
            cell['source'] = src.replace(
                '# HuggingFace login\nfrom huggingface_hub import login\nfrom kaggle_secrets import UserSecretsClient\nuser_secrets = UserSecretsClient()\nhf_token = user_secrets.get_secret("HF_TOKEN")\nlogin(token=hf_token)\n',
                '# Optional HuggingFace login (Cloudexe-safe)\nimport os\nfrom huggingface_hub import login\nhf_token = os.getenv("HF_TOKEN", "")\nif hf_token:\n    login(token=hf_token)\n'
            )

        if 'N_SAMPLES      = 1000' in src:
            cell['source'] = src.replace('N_SAMPLES      = 1000', 'N_SAMPLES      = 600')

        if "ds = load_dataset('openlanguagedata/flores_plus', lang_code, split='devtest')" in src:
            cell['source'] = '''
from datasets import load_dataset, get_dataset_config_names

# OPUS-100 mapping for FLORES-style codes
FLORES_TO_OPUS = {
    'eng_Latn':'en','fra_Latn':'fr','cmn_Hans':'zh','urd_Arab':'ur','hin_Deva':'hi',
    'ben_Beng':'bn','mar_Deva':'mr','nld_Latn':'nl','pol_Latn':'pl','snd_Arab':'sd',
    'awa_Deva':'awa','azb_Arab':'azb'
}
opus_cfg = set(get_dataset_config_names('Helsinki-NLP/opus-100'))

language_data = {}
for lang_code, lang_name in LANGUAGES.items():
    print(f'Loading {lang_name}...')
    try:
        iso = FLORES_TO_OPUS.get(lang_code)
        if not iso:
            raise ValueError(f'No ISO mapping for {lang_code}')
        if iso == 'en':
            config = 'en-fr'
            field = 'en'
        else:
            c1 = f'{iso}-en'
            c2 = f'en-{iso}'
            if c1 in opus_cfg:
                config, field = c1, iso
            elif c2 in opus_cfg:
                config, field = c2, iso
            else:
                raise ValueError(f'No OPUS-100 config for {lang_code} ({iso})')

        ds = load_dataset('Helsinki-NLP/opus-100', config, split='train')
        texts = []
        for row in ds:
            t = row.get('translation', {}).get(field, '')
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
            if len(texts) >= N_SAMPLES:
                break

        if len(texts) < 100:
            raise ValueError(f'Insufficient data for {lang_code}: {len(texts)}')

        language_data[lang_code] = texts
        print(f'  {len(language_data[lang_code])} sentences via OPUS-100::{config}')
    except Exception as e:
        print(f'  Error loading {lang_name}: {e}')
'''

    patched = path.with_name('plora-s1-and-s2.cloudexe.ipynb')
    nbformat.write(nb, patched)
    return patched

# deps for notebook execution
run([py, '-m', 'pip', 'install', '-q', 'jupyter', 'nbconvert', 'ipykernel', 'nbformat', 'matplotlib', 'seaborn', 'scipy', 'pot', 'transformers', 'datasets', 'accelerate'])

patched_step12 = patch_step12_notebook(run_dir / 'plora-s1-and-s2.ipynb')

# Execute patched Step 1/2 notebook
run([
    py, '-m', 'jupyter', 'nbconvert',
    '--to', 'notebook', '--execute', str(patched_step12.name),
    '--output', 'plora-s1-and-s2.executed.ipynb',
    '--ExecutePreprocessor.timeout=-1'
])

csv = run_dir / 'spinal_plora_qwen3_4b_results_12lang.csv'
if not csv.exists():
    raise SystemExit('Expected CSV missing after Step 1/2 execution')

Path('/content').mkdir(parents=True, exist_ok=True)
shutil.copy2(csv, '/content/spinal_plora_qwen3_4b_results_12lang.csv')

# Execute Step 4 notebook (with built-in Step 3 fallback)
run([
    py, '-m', 'jupyter', 'nbconvert',
    '--to', 'notebook', '--execute', 'plora_step4_rank_budgeting_SN.ipynb',
    '--output', 'plora_step4_rank_budgeting_SN.executed.ipynb',
    '--ExecutePreprocessor.timeout=-1'
])

out_json = run_dir / 'plora_step4_rank_budgets.json'
out_csv  = run_dir / 'plora_step4_rank_budgets.csv'
out_sum  = run_dir / 'plora_step4_param_summary.csv'

result = {
    'ok': out_json.exists(),
    'rank_budget_json': str(out_json),
    'rank_budget_csv': str(out_csv),
    'summary_csv': str(out_sum),
}
with open(run_dir / 'result.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2), flush=True)
