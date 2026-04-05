These are the non-exact benchmark substitutions in [plora_task_dataset_manifest.hub.json](C:\Work\PLoRA\plora_task_dataset_manifest.hub.json).

- French summarization uses `wiki_lingua` (`french`) instead of `MLSUM`. The legacy `reciTAL/mlsum` dataset is script-only in the current `datasets` environment and no hub-native French mirror with standard splits was found.
- Chinese summarization uses `wiki_lingua` (`chinese`) instead of `XL-Sum`. The available `xlsum` hubs are exposed through legacy dataset scripts or tar archives rather than a hub-native split layout that the current loader can consume directly.
- Chinese QA uses `xquad.zh` with a sliced validation split because a clean hub-native MLQA Chinese train split was not found in this environment.
- Dutch QA uses `Nelis5174473/Dutch-QA-Pairs-Rijksoverheid`, which is question-answer style without an extractive context field.
- Marathi QA uses `l3cube-pune/indic-squad` (`Marathi`) instead of `MahaSQuAD`. No accessible hub-native `MahaSQuAD` source was found.
- Marathi sentiment uses `mteb/IndicSentiment` (`mr`) instead of `MahaSent`. No accessible hub-native `MahaSent` source was found.
- Urdu sentiment uses `sepidmnorozy/Urdu_sentiment` instead of `SentiUrdu-1M`. No accessible hub-native `SentiUrdu-1M` source was found.

If you want exact benchmark matching for any of the substitutions above, send the preferred source URLs or local files and the manifest can be swapped without changing the training or evaluation code.
