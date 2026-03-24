import configparser
from pathlib import Path
import polars as pl
from jiwer import wer, cer
from scipy.stats import spearmanr
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

def compute_wer_cer(
        df: pl.DataFrame,
        text_col: str,
        pred_col: str,
        normalizer) -> tuple[float, float]:
    """
    Helper function to compute WER/CER

    Args:
        df: _description_
        text_col: _description_
        pred_col: _description_
        normalizer: _description_

    Returns:
        _description_
    """
    refs = df[text_col].to_list()
    hyps = df[pred_col].to_list()

    refs = [normalizer(r.lower().strip()) for r in refs]
    hyps = [normalizer(h.lower().strip()) for h in hyps]

    avg_wer = wer(reference=refs, hypothesis=hyps)
    avg_cer = cer(reference=refs, hypothesis=hyps)

    return avg_wer*100, avg_cer*100

def compute_group_metrics(df, text_col, pred_col, normalizer, group_cols):
    return (
        df
        .group_by(group_cols)
        .map_groups(
            lambda g: pl.DataFrame({
                **{c: g[c][0] for c in group_cols},
                "wer": compute_wer_cer(g, text_col, pred_col, normalizer)[0],
                "cer": compute_wer_cer(g, text_col, pred_col, normalizer)[1],
            })
        )
    )


def infer_subset_suffix(rel: Path, modality_dir: str) -> tuple[str | None, str | None]:
    """
    Get subset/suffix via full path

    Args:
        rel: _description_
        modality_dir: _description_

    Returns:
        _description_
    """
    parts = list(rel.parts)

    # normalize: modality/file.ext
    if len(parts) == 2 and parts[0] == modality_dir:
        parts = [parts[1]]

    subset = None
    suffix = None

    if len(parts) >= 3:
        if parts[-2] == modality_dir:
            subset = parts[-3]
        elif parts[-3] == modality_dir:
            subset = parts[-2]
        else:
            subset = parts[-3]
            suffix = parts[-2]
    elif len(parts) == 2:
        subset = parts[-2]

    return subset, suffix


def get_asr_utter(split: str, corpus: str, model_name: str):
    asr_root = Path(config_parser['outputs']['asr']) / split / corpus
    asr_files = asr_root.rglob(f"{model_name}_output.parquet")

    asr_dfs: list[pl.DataFrame] = []
    for f in asr_files:
        subset, suffix = infer_subset_suffix(f, modality_dir='asr')
        df = pl.read_parquet(f)
        df = df.with_columns(
            pl.lit(subset).alias("subset"),
            pl.lit(suffix).alias("suffix"),
            pl.col("audio_path")
            .str.split("/")
            .list.last()
            .str.replace(r"\.[^.]+$", "")
            .str.split("_")
            .list.get(0)
            .alias("task"),
            pl.col("audio_path")
            .str.split("/")
            .list.last()
            .str.replace(r"\.[^.]+$", "")
            .str.split("_")
            .list.get(1)
            .alias("pid"),
        )
        asr_dfs.append(df)
    asr_df = pl.concat(asr_dfs)
    # add start and end
    asr_df = asr_df.with_columns(
        pl.col("audio_path")
        .str.split("/")
        .list.last()
        .str.replace(r"\.[^.]+$", "")
        .str.split("_")
        .alias("parts")
    ).with_columns(
        pl.col("parts").list.get(-2).cast(pl.Int64).alias("start"),
        pl.col("parts").list.get(-1).cast(pl.Int64).alias("end"),
    ).drop("parts")
    
    asr_df.write_parquet((f"./data/{corpus}_{split}_utter_asr.parquet"))

    # merge to pid level
    asr_df = asr_df.sort(["pid", "task", "subset", "suffix", "start", "end"])
    merged = (
        asr_df
        .group_by(
            ["pid", "task", "subset", "suffix"],
            maintain_order=True
        )
        .agg([
            pl.col("transcription").str.join(" ").alias("transcription"),
            pl.col("prediction").str.join(" ").alias("prediction"),
            pl.exclude(
                "transcription",
                "prediction",
                "start",
                "end",
                "audio_path"
            ).first(),
        ])
    )
    merged.write_parquet(f"./data/merged/{corpus}_{split}_pid.parquet")

def get_utter_df(split: str, corpus: str):
    """
    Utterance-level statistics
    """
    clips_root = Path(config_parser['outputs']['clips']) / split / corpus
    clips_meta = list(clips_root.rglob("*.parquet"))

    meta_dfs: list[pl.DataFrame] = []
    for f in clips_meta:
        subset, suffix = infer_subset_suffix(f, modality_dir='clips')
        meta = pl.read_parquet(f)
        meta = meta.with_columns(
            pl.lit(subset).alias('subset'),
            pl.lit(suffix).alias('suffix'),
            pl.col("clip_path")
                .map_elements(lambda p: Path(p).stem.split("_")[0])
                .alias("task")
        )
        meta_dfs.append(meta)
    meta_df = pl.concat(meta_dfs, how='vertical')

    meta_df = meta_df.with_columns(
        normalize_pid(pl.col("pid")).alias("pid")
    )

    print(f"Number of recordings in {corpus} in {split}: {meta_df['source_audio'].n_unique()}")
    print(f"Number of clips in {corpus} in {split}: {meta_df['clip_path'].n_unique()}")
    print(f"Number of unique pid in {corpus} in {split}: {meta_df["pid"].n_unique()}")
    print(f"Number of utterances in {corpus} in {split}: {meta_df.height}")
    print("--------------")

    meta_df.write_parquet(f"./data/{corpus}_{split}_utter.parquet")
        
def normalize_pid(col: pl.Expr) -> pl.Expr:
    return (
        pl.when(col.str.contains("_"))
        .then(col.str.split("_").list.first())
        .when(col.str.contains("-"))
        .then(col.str.split("-").list.first())
        .otherwise(col)
    )


def get_meta(corpus: str, scale_col: list | str) -> pl.DataFrame:
    if isinstance(scale_col, list):
        scale_sel = scale_col
    else:
        scale_sel = [scale_col]

    if corpus == "discourse":
        meta_df = (
            pl.read_csv('./data/meta/DISCOURSE_baseline.csv')
            .select(
                ['ID', 'PatientCat']
                + scale_sel
                + [pl.selectors.starts_with('PANSS10')]
            )
            .drop(["PANSS10G2", "PANSS10G6"])
            .with_columns(
                pl.sum_horizontal(pl.selectors.starts_with("PANSS10"))
                .alias("panss-total")
            )
            .drop(pl.selectors.starts_with("PANSS10"))
            .rename({'ID': 'pid'})
        )
        meta_df = meta_df.drop_nulls()
        meta_df = meta_df.with_columns(
            pl.when(pl.col("PatientCat") == 1)
            .then(pl.lit("HC"))
            .otherwise(pl.lit("PT"))
            .alias("label")
        )
    else:
        meta_df = (
            pl.read_csv('./data/meta/TOPSY.csv', ignore_errors=True)
            .select(['participant_id', 'session_id', 'phenotype', 'panss-total']
                    + scale_sel)
            .filter(pl.col("session_id") == 'ses-1')
            .rename({'participant_id': 'pid'})
            .with_columns(
                pl.col("pid").str.split("-").list.last()
            )
            .rename({'TLI_Disorg': 'TLIDISORG', 'TLI_Total': 'TLITOTAL'})
            .drop('session_id')
        )
        meta_df = meta_df.drop_nulls()
        meta_df = meta_df.with_columns(
            pl.when(pl.col("phenotype") != "HC")
            .then(pl.lit("PT"))
            .otherwise(pl.lit("HC"))
            .alias("label")
        )
    return meta_df

def main():
    normalizer = BasicTextNormalizer()
    for corpus in CORPUS:
        print(f"====== {corpus} ======")
        for split in ['task', 'full']:
            get_asr_utter(split=split, corpus=corpus, model_name='whisper-large-v3')
        if corpus == "discourse":
            scale_col = ['TLIDISORG', 'TLITOTAL']
            df = pl.read_parquet(f"./data/merged/{corpus}_task_pid.parquet")
            df = df.filter(
                (pl.col('task') == "Pictures") &
                (pl.col('suffix') == "Baseline")
            )
        else:
            scale_col = ['TLI_Disorg', 'TLI_Total']
            df = pl.read_parquet(f"./data/merged/{corpus}_full_pid.parquet")
            df = df.filter(
                pl.col('suffix') == "TOPSY-0"
            )
        # ----- picture description task verbatim utterances, merged on participant level ------
        meta = get_meta(corpus=corpus, scale_col=scale_col)
        merged = df.join(meta, on='pid', how='inner')
        # WER/CER
        errors = compute_group_metrics(
            df=merged,
            text_col='transcription',
            pred_col='prediction',
            normalizer=normalizer,
            group_cols=['pid']
        )
        merged = merged.join(errors, on='pid', how='inner')
        merged.write_parquet(f"./data/merged/{corpus}_pid_pictures_baseline.parquet")

        rho, p_value = spearmanr(merged['wer'].to_numpy(), merged['TLIDISORG'].to_numpy())
        print(f"TLI disorgnization rho on {corpus}: {round(rho, 2)}; p-value: {round(p_value, 4)}")
        print("-----------")
        rho, p_value = spearmanr(merged['wer'].to_numpy(), merged['TLITOTAL'].to_numpy())
        print(f"TLI total rho on {corpus}: {round(rho, 2)}; p-value: {round(p_value, 4)}")
        print("-----------")
        rho, p_value = spearmanr(merged['wer'].to_numpy(), merged['panss-total'].to_numpy())
        print(f"PANSS total rho on {corpus}: {round(rho, 2)}; p-value: {round(p_value, 4)}")
        print("-----------")

        # --- task level utterance ----
        for split in ['task', 'full']:
            get_utter_df(split=split, corpus=corpus)

def print_task_asr():
    normalizer = BasicTextNormalizer()
    rows = []
    for corpus in CORPUS:
        for model_name in MODELS:
            asr_dir = Path(config_parser['outputs']['asr']) / "task" / corpus 
            asr_files = list(asr_dir.rglob(f"{model_name}_output.parquet"))
            
            dfs: list[pl.DataFrame] = []
            for f in asr_files:
                df = pl.read_parquet(f)
                dfs.append(df)
            asr_df = pl.concat(dfs, how='vertical')
            asr_df = asr_df.with_columns(
                pl.col("audio_path")
                    .str.split("/")
                    .list.last()
                    .str.replace(r"\.[^.]+$", "")
                    .str.split("_")
                    .list.get(0)
                    .alias("task")
            )

            for task, g in asr_df.group_by('task'):
                wer_, cer_ = compute_wer_cer(
                    g,
                    text_col='transcription',
                    pred_col='prediction',
                    normalizer=normalizer
                )
                rows.append(
                {
                    "corpus": corpus,
                    "model": model_name,
                    "task": task[0],
                    "wer": wer_,
                    "cer": cer_,
                }
            )
    error_df = pl.DataFrame(rows)
    error_df = error_df.with_columns(
        [
        pl.col("wer").round(2),
        pl.col("cer").round(2),
        ]
    )
    error_df.write_csv("./data/error.csv", float_precision=2)

if __name__ == "__main__":
    config_parser = configparser.ConfigParser()
    config_parser.read("config.ini")

    MODELS = ['wav2vec2-large-960h', 'hubert-large-ls960-ft', 'whisper-large-v3']
    CORPUS =  ['topsy', 'discourse']
    
    main()
    # print_task_asr()
    # df = pl.read_parquet("./data/merged/discourse_pid_pictures_baseline.parquet")
    # print(df.head())
    # print(df.schema)

    # df = pl.read_parquet("./data/merged/topsy_pid_pictures_baseline.parquet")
    # print(df.head())
    # print(df.schema)