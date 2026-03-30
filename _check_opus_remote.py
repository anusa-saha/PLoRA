from datasets import get_dataset_config_names, load_dataset
langs={
"hin_Deva":"hi","fra_Latn":"fr","cmn_Hans":"zh","urd_Arab":"ur","eng_Latn":"en","awa_Deva":"awa","ben_Beng":"bn","mar_Deva":"mr","nld_Latn":"nl","pol_Latn":"pl","snd_Arab":"sd","azb_Arab":"azb"}
cfg=set(get_dataset_config_names("Helsinki-NLP/opus-100"))
print('cfg',len(cfg))
for lc,iso in langs.items():
    if iso=='en':
        conf='en-fr';field='en'
    elif f'{iso}-en' in cfg:
        conf=f'{iso}-en';field=iso
    elif f'en-{iso}' in cfg:
        conf=f'en-{iso}';field=iso
    else:
        print(lc,'missing')
        continue
    ds=load_dataset('Helsinki-NLP/opus-100', conf, split='train[:3]')
    print(lc,conf,field,ds[0]['translation'][field][:30])
