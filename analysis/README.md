# Analysis Code

This folder contains the scripts for downstream text-based analysis


## Features

We compute a series of textual features in our analysis, including:

- Semantic similarity features as proposed in [1, 2]
- Perplexity features as proposed in [3]
- Speech graph features as proposed in [4, 5, 6, 7]



[1] Xu, W., Wang, W., Portanova, J., Chander, A., Campbell, A., Pakhomov, S., ... & Cohen, T. (2022). Fully automated detection of formal thought disorder with Time-series Augmented Representations for Detection of Incoherent Speech (TARDIS). Journal of biomedical informatics, 126, 103998.

[2] Xu, W., Pakhomov, S., Heagerty, P., Horvitz, E., Bradley, E. R., Woolley, J., ... & Cohen, T. (2025). Perplexity and proximity: Large language model perplexity complements semantic distance metrics for the detection of incoherent speech. Journal of Biomedical Informatics, 104899.

[3] Li, C., Xu, W., Pakhomov, S., Bradley, E., Ben-Zeev, D., & Cohen, T. (2025, May). Bigger But Not Better: Small Neural Language Models Outperform LLMs in Detection of Thought Disorder. In Proceedings of the 10th Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2025) (pp. 90-105).

[4] Carrillo, F., Mota, N., Copelli, M., Ribeiro, S., Sigman, M., Cecchi, G., & Fernandez Slezak, D. (2013, December). Automated speech analysis for psychosis evaluation. In International Workshop on Machine Learning and Interpretation in Neuroimaging (pp. 31-39). Cham: Springer International Publishing.

[5] Mota, N. B., Vasconcelos, N. A., Lemos, N., Pieretti, A. C., Kinouchi, O., Cecchi, G. A., ... & Ribeiro, S. (2012). Speech graphs provide a quantitative measure of thought disorder in psychosis. PloS one, 7(4), e34928.

[6] Mota, N. B., Copelli, M., & Ribeiro, S. (2017). Thought disorder measured as random speech structure classifies negative symptoms and schizophrenia diagnosis 6 months in advance. npj Schizophrenia, 3(1), 18.

[7] Mota, N. B., Sigman, M., Cecchi, G., Copelli, M., & Ribeiro, S. (2018). The maturation of speech structure in psychosis is resistant to formal education. npj Schizophrenia, 4(1), 25.

## Environments

Our analysis scripts are in python 3.12 with CUDA 12.6. For more details, please see the `pyproject.toml` file.

## Structure

```
├── data
│   ├── ccc
│   │   ├── {corpus}_pid_picture_baseline_{feature}_ccc.parquet
│   ├── {corpus}_{model}_utter.parquet
│   ├── error.csv
│   ├── merged
│   │   ├── {corpus}_ccc_meta.parquet
│   │   ├── {corpus}_{model}.parquet
│   │   ├── {corpus}_pid_picture_baseline_meta.parquet
│   │   ├── {corpus}_pid_picture_baseline.parquet
│   ├── meta
│   │   ├── {corpus}.csv
│   ├── {corpus}_{model}_utter.parquet
├── compare_features.py
├── config.ini
├── eval_data.py
├── feature_extractor.py
├── get_features.py
```

- `eval_data.py`: get the statistics of ASR output and merge with metadata
- `feature_extractor.py`: feature extraction pipeline
- `get_features.py`: the driver code for the feature extraction pipeline
- `compare_features.py`: generate tables for feature comparison
- `data`: the folder containing the merged data files