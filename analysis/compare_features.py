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
    text_ccc_file = f"./data/ccc/{corpus}_pid_picture_baseline_prediction_full_ccc.parquet"
    asr_ccc_file = f"./data/ccc/{corpus}_pid_picture_baseline_prediction_full_ccc.parquet"
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


def get_sig_features(merged_file: str, scale_col: str):
    df = pl.read_parquet(merged_file)
    cols = df.columns

    meta_cols = [ 
        'pid','wer', 'cer', 'panss-total',
        'TLITOTAL', 'TLIDISORG','label']

    ppl_text_cols = ['avg_ppl', 'sliding_window',
                    'sliding_window_batch', 'contextmodel', 'topicmodel']
    ppl_asr_cols = [f"{item}_pred" for item in ppl_text_cols]

    graph_text_cols = ['number_of_nodes', 'number_of_edges',
                    'PE', 'number_scc', 'LSC', 'density',
                    'degree_average', 'degree_std', 'L1']
    graph_asr_cols = [f"{item}_pred" for item in graph_text_cols]

    exclude_cols = (
        ppl_text_cols
        + ppl_asr_cols
        + graph_text_cols
        + graph_asr_cols
        + meta_cols
    )

    ccc_text_cols = [
        col for col in cols
        if col not in exclude_cols and not col.endswith("_pred")
    ]
    ccc_asr_cols = [
        col for col in cols
        if col not in exclude_cols and col.endswith("_pred")
    ]

    dfs = [
        compute_rho(df, ccc_text_cols,  scale_col, 'ccc_text'),
        compute_rho(df, ccc_asr_cols,   scale_col, 'ccc_asr'),
        compute_rho(df, ppl_text_cols,  scale_col, 'ppl_text'),
        compute_rho(df, ppl_asr_cols,   scale_col, 'ppl_asr'),
        compute_rho(df, graph_text_cols, scale_col, 'graph_text'),
        compute_rho(df, graph_asr_cols,  scale_col, 'graph_asr'),
    ]

    result = pl.concat(dfs).drop(pl.col('p_val'))
    print(result.filter(pl.col('significant') == True).to_pandas())

    return result.filter(pl.col('significant') == True)


def format_for_latex(result: pl.DataFrame) -> pl.DataFrame:
    df = (
        result
        .filter(pl.col("significant"))
        .with_columns(
            pl.when(pl.col("feature").str.ends_with("_pred"))
              .then(pl.lit("ASR"))
              .otherwise(pl.lit("Verbatim"))
              .alias("transcript"),
            pl.col("feature").str.replace("_pred$", "").alias("feature")
        )
        .select(["feature", "transcript", "rho"])
        .pivot(
            values="rho",
            index="feature",
            on="transcript",
            aggregate_function="first"
        )
    )

    for col in ["Verbatim", "ASR"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit("--").alias(col))

    df = (
        df
        .with_columns([
            pl.col("Verbatim").fill_null("--").cast(pl.Utf8),
            pl.col("ASR").fill_null("--").cast(pl.Utf8),
        ])
        .select(["feature", "Verbatim", "ASR"])
        .sort("feature")
    )

    print(df.to_pandas())
    return df

def to_latex_rows(df: pl.DataFrame):
    for row in df.iter_rows():
        feature, verbatim, asr = row
        print(f"{feature} & {verbatim} & {asr} \\\\")


def find_diff(corpus: str):
    total_df = pl.read_parquet(f"./data/merged/{corpus}_ccc_meta.parquet")
    
    exclude = [
        'pid', 'wer', 'cer', 'panss-total',
        'TLITOTAL', 'TLIDISORG','label']
    features = [
        c for c in total_df.columns
        if c not in exclude
        and not c.endswith("_pred")
        and f"{c}_pred" in total_df.columns
    ]
    diff_exprs = [
        (pl.col(c) - pl.col(f"{c}_pred")).alias(f"{c}_diff")
        for c in features
        if c not in exclude
    ]
    df_diff = total_df.select(["pid", *diff_exprs])

    p_values = []
    for c in features:
        x = total_df[c].to_numpy()
        y = total_df[f"{c}_pred"].to_numpy()
        _, p = ttest_rel(x, y, nan_policy="omit")
        p_values.append(p)
    reject, pvals_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    result = pl.DataFrame({
        "feature": features,
        "p_value": p_values,
        "p_value_fdr": pvals_fdr,
        "significant": pvals_fdr < 0.05,
    })

    orig_means = []
    pred_means = []
    diff_means = []

    for c in features:
        orig = total_df[c].to_numpy()
        pred = total_df[f"{c}_pred"].to_numpy()
        diff = orig - pred

        orig_means.append(np.nanmean(orig))
        pred_means.append(np.nanmean(pred))
        diff_means.append(np.nanmean(diff))

    result = result.with_columns([
        pl.Series("mean_orig", orig_means),
        pl.Series("mean_pred", pred_means),
        pl.Series("mean_diff", diff_means),
    ])
    with pl.Config(tbl_rows=result.height):
        subset = result.filter(pl.col('significant') == True)
        subset = subset.select(['feature','mean_orig', 'mean_pred', 'mean_diff'])
        print(result.filter(pl.col('significant') == True))
        print(result.filter(pl.col('significant') == False).height)
    
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
        meta_file = f"./data/merged/{corpus}_pid_picture_baseline_meta.parquet"
        meta_df = pl.read_parquet(meta_file)
        merge_with_ccc(meta_df=meta_df, corpus='discourse')
        merged_file = f"./data/merged/{corpus}_ccc_meta.parquet"
        sig_df = get_sig_features(
            merged_file=merged_file,
            scale_col=scales[1]
        )
        sig_latex_df = format_for_latex(sig_df)
        to_latex_rows(sig_latex_df)
        find_diff(corpus=corpus)