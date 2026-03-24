import configparser
from pathlib import Path
import polars as pl
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

def merge_with_ccc(meta_df: pl.DataFrame, corpus: str):
    asr_ccc_file = f"./data/ccc/{corpus}_pid_pictures_baseline_prediction_ccc.parquet"
    text_ccc_file = f"./data/ccc/{corpus}_pid_pictures_baseline_transcription_ccc.parquet"
    text_df = pl.read_parquet(text_ccc_file)
    asr_df = pl.read_parquet(asr_ccc_file)
    ccc_df = text_df.join(asr_df, on='pid', suffix='_pred')

    final_df = meta_df.join(ccc_df, on='pid')
    final_df.write_parquet(f"./data/merged/{corpus}_ccc_meta.parquet")

def compute_rho(
    df: pl.DataFrame,
    target_cols: list,
    scale_col: str,
    group: str
) -> pl.DataFrame:
    rows = []

    for c in target_cols:
        sub = df.select([c, scale_col]).drop_nulls()
        if sub.height < 2:
            continue

        rho, p = spearmanr(
            sub[c].to_numpy(),
            sub[scale_col].to_numpy()
        )

        rows.append(((c, group, round(rho, 2), p, p < 0.05)))

    return pl.DataFrame(
        rows,
        schema={
            'feature': pl.Utf8,
            'group': pl.Utf8,
            'rho': pl.Float64,
            'p_val': pl.Float64,
            'significant': pl.Boolean,
        },
        orient='row'
    )


def get_sig_features(merged_file: str, scale_col: str) -> pl.DataFrame:
    df = pl.read_parquet(merged_file)
    cols = set(df.columns)
    

    meta_cols = {
        'pid','wer','cer','panss-total','PatientCat',
        'TLITOTAL','TLIDISORG','label',
        'transcription','prediction'
    }

    if "phenotype" in df.columns:
        meta_cols.add("phenotype")

    ppl_text = {'avg_ppl','sliding_window','sliding_window_batch','contextmodel','topicmodel'}
    graph_text = {
        'number_of_nodes','number_of_edges','PE','number_scc',
        'LSC','density','degree_average','degree_std','L1'
    }

    ppl_asr = {f"{c}_pred" for c in ppl_text}
    graph_asr = {f"{c}_pred" for c in graph_text}

    exclude = meta_cols | ppl_text | ppl_asr | graph_text | graph_asr

    ccc_text = [c for c in cols if c not in exclude and not c.endswith("_pred")]
    ccc_asr  = [c for c in cols if c not in exclude and c.endswith("_pred")]

    text_features = set(ccc_text) | ppl_text | graph_text
    asr_features  = set(ccc_asr)  | ppl_asr  | graph_asr

    dfs = [
        compute_rho(df, ccc_text,   scale_col, 'ccc_text'),
        compute_rho(df, ccc_asr,    scale_col, 'ccc_asr'),
        compute_rho(df, list(ppl_text),  scale_col, 'ppl_text'),
        compute_rho(df, list(ppl_asr),   scale_col, 'ppl_asr'),
        compute_rho(df, list(graph_text), scale_col, 'graph_text'),
        compute_rho(df, list(graph_asr),  scale_col, 'graph_asr'),
    ]

    result = pl.concat(dfs).drop("p_val")
    sig = result.filter(pl.col("significant"))

    print(
        f"# significant features: {sig.height} "
        f"out of ({len(text_features)} text + {len(asr_features)} ASR)"
    )
    return sig

def format_for_latex(result: pl.DataFrame) -> pl.DataFrame:
    df = (
        result
        .with_columns([
            pl.when(pl.col("group").str.contains("asr"))
              .then(pl.lit("ASR"))
              .otherwise(pl.lit("Verbatim"))
              .alias("transcript"),
            pl.col("feature").str.replace("_pred$", "")
              .alias("feature")
        ])
        .select("feature", "transcript", "rho")
        .pivot(
            index="feature",
            on="transcript",
            values="rho",
            aggregate_function="first"
        )
        .with_columns([
            pl.col("Verbatim").fill_null("--").cast(pl.Utf8),
            pl.col("ASR").fill_null("--").cast(pl.Utf8),
        ])
        .select("feature", "Verbatim", "ASR")
        .sort("feature")
    )

    return df

def to_latex_rows(df: pl.DataFrame):
    for row in df.iter_rows():
        feature, verbatim, asr = row
        print(f"{feature} & {verbatim} & {asr} \\\\")


def find_diff(corpus: str, sig_features: list[str]):
    df = pl.read_parquet(f"./data/merged/{corpus}_ccc_meta.parquet")

    features = [
        c for c in sig_features
        if c in df.columns and f"{c}_pred" in df.columns
    ]

    x = df.select(features).to_numpy()
    y = df.select([f"{c}_pred" for c in features]).to_numpy()

    p_values = [
        ttest_rel(x[:, i], y[:, i], nan_policy="omit")[1]
        for i in range(len(features))
    ]

    _, p_fdr, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    mean_orig = np.nanmean(x, axis=0)
    mean_pred = np.nanmean(y, axis=0)
    mean_diff = mean_orig - mean_pred

    result = pl.DataFrame({
        "feature": features,
        "mean_orig": mean_orig,
        "mean_pred": mean_pred,
        "mean_diff": mean_diff,
        "p_value_fdr": p_fdr,
        "significant": p_fdr < 0.05
    })

    subset = result.filter(pl.col("significant"))
    print(" ------------ diff --------------")
    print(f"\nNumber of features that are both significant: {subset.height}")
    with pl.Config(tbl_rows=result.height):
        subset = result.filter(pl.col('significant') == True)
        subset = subset.select(['feature','mean_orig', 'mean_pred', 'mean_diff'])
    
        latex = subset.to_pandas().to_latex(
            index=False,
            escape=True,
            column_format="c" * len(result.columns),
            float_format="%.4g",
            longtable=False,
            caption="Paired t-test results with FDR correction",
            label="tab:ttest_results",
            bold_rows=False,
            multicolumn=False,
            multicolumn_format="c",
            header=True,
            na_rep=""
        )

        latex = latex.replace("\\toprule", "\\toprule\n")
        print(latex)

if __name__ == "__main__":
    config_parser = configparser.ConfigParser()
    config_parser.read("config.ini")
    scales = ['TLIDISORG', 'TLITOTAL']
    for corpus in ['discourse', 'topsy']:
        print(f"======= {corpus} ========")
        
        merged_df = pl.read_parquet(f"./data/merged/{corpus}_pid_pictures_baseline.parquet")
        merge_with_ccc(meta_df=merged_df, corpus=corpus)
        sig_df = get_sig_features(
            merged_file=f"./data/merged/{corpus}_ccc_meta.parquet",
            scale_col=scales[0]
        )
        sig_latex_df = format_for_latex(sig_df)
        to_latex_rows(sig_latex_df)
        sig_features = sig_df["feature"].to_list()
        find_diff(corpus=corpus, sig_features=sig_features)