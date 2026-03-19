import configparser
from pathlib import Path
import polars as pl
from jiwer import wer, cer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from scipy.stats import spearmanr

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

def get_utter_df():
    """
    Utterance-level statistics
    """
    for corpus in CORPUS:
        audio_root = Path(config_parser['outputs']['audio']) / corpus
        audio_lists = list(audio_root.rglob("*.wav"))
        clips_root = Path(config_parser['outputs']['clips']) / corpus
        clips_lists = list(clips_root.rglob("*.wav"))
        text_root = Path(config_parser['outputs']['text']) / corpus
        utterance_lists = list(text_root.rglob("*_utterance.parquet"))
        dfs = []
        for f in utterance_lists:
            task = f.stem.split("_")[0]
            df = pl.read_parquet(f).with_columns(
                pl.lit(task).alias("task")
            )
            dfs.append(df)

        df = pl.concat(dfs, how="vertical")
        df = pl.concat(dfs, how="vertical")
        df = df.with_columns(
            pl.when(pl.col("pid").str.contains("_"))
            .then(pl.col("pid").str.split("_").list.first())
            .when(pl.col("pid").str.contains("-"))
            .then(pl.col("pid").str.split("-").list.first())
            .otherwise(pl.col("pid"))
            .alias("pid")
        )

        print(f"Number of recordings in {corpus}: {len(audio_lists)}")
        print(f"Number of clips in {corpus}: {len(clips_lists)}")
        print(f"Number of unique pid in {corpus}: {df["pid"].n_unique()}")
        print(f"Number of utterances in {corpus}: {df.height}")
        print("--------------")

def get_asr():
    """
    Compute WER/CER per utterance
    """
    for model_name in MODELS:
        for corpus in CORPUS:
            asr_root = Path(config_parser['outputs']['asr']) / corpus
            asr_lists = list(asr_root.rglob(f"{model_name}*.parquet"))
            
            dfs = []

            for f in asr_lists:
                df = pl.read_parquet(f)

                subset, suffix = infer_subset_suffix(f, 'asr')
                meta_path = Path(config_parser['outputs']['clips'])

                if not str(subset).startswith(corpus):
                    meta_path = meta_path / corpus
                
                meta_path = meta_path / subset
                
                if suffix:
                    meta_path = meta_path / suffix

                meta_path = meta_path / "metadata.parquet"

                meta = pl.read_parquet(
                    meta_path,
                    columns=["clip_path", "source_audio"],
                )

                # join source audio
                df = df.join(
                    meta,
                    left_on="audio_path",
                    right_on="clip_path",
                    how="left",
                )

                dfs.append(df)

            df = pl.concat(dfs, how="vertical")
            # add pid and task
            df = df.with_columns(
                pl.col("audio_path")
                .map_elements(lambda x: Path(x).stem)
                .str.split("_")
                .alias("parts")
            ).with_columns(
                [
                    pl.col("parts").list.get(0).alias("task"),
                    pl.col("parts").list.get(1).alias("pid"),
                ]
            ).drop("parts")
            # split pid if needed
            df = df.with_columns(
                pl.when(pl.col("pid").str.contains("_"))
                .then(pl.col("pid").str.split("_").list.first())
                .when(pl.col("pid").str.contains("-"))
                .then(pl.col("pid").str.split("-").list.first())
                .otherwise(pl.col("pid"))
                .alias("pid")
            )
            df = df.with_columns(
                pl.col("audio_path")
                .map_elements(
                    lambda x: infer_subset_suffix(Path(x), "clips"),
                    return_dtype=pl.Struct(
                        [
                            pl.Field("subset", pl.Utf8),
                            pl.Field("suffix", pl.Utf8),
                        ]
                    ),
                )
                .alias("out")
            ).with_columns(
                [
                    pl.col("out").struct.field("subset").alias("subset"),
                    pl.col("out").struct.field("suffix").alias("suffix"),
                ]
            ).drop("out")
            # with pl.Config(fmt_str_lengths=1000):
            #     print(df.schema)
            df.write_parquet(f"./data/{corpus}_{model_name}_utter.parquet")

def compute_errors():
    """
    Compute WER/CER as per task
    """
    normalizer = BasicTextNormalizer()
    rows = []
    for corpus in CORPUS:
        for model_name in MODELS:
            df = pl.read_parquet(f"./data/{corpus}_{model_name}_utter.parquet")
            for task, g in df.group_by('task'):
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

def reformat_data():
    """
    Group utterances with pid-subset-suffix-task level transcripts
    Each task has their own transcript per pid in each study session
    """
    for corpus in CORPUS:
        for model_name in MODELS:
            df = pl.read_parquet(f"./data/{corpus}_{model_name}_utter.parquet")
            model = model_name.split("-")[0]
            
            out = (
                df
                .filter(
                    pl.any_horizontal(
                        pl.col("prediction").is_not_null() & (pl.col("prediction") != ""),
                        pl.col("transcription").is_not_null() & (pl.col("transcription") != ""),
                    )
                )
                .group_by(["pid", "subset", "suffix", "task"])
                .agg([
                    (
                        pl.col("prediction")
                        .str.join(" ")
                        .str.to_lowercase()
                    ).alias("prediction_concat"),

                    (
                        pl.col("transcription")
                        .str.join(" ")
                        .str.to_lowercase()
                    ).alias("transcription_concat"),

                    pl.col("source_audio").first().alias("source_audio"),
                ])
            )
            out.write_parquet(f"./data/merged/{corpus}_{model}.parquet")

def compute_group_metrics(df, text_col, pred_col, normalizer, group_cols):
    """
    Compute WER/CER by groups

    Args:
        df: _description_
        text_col: _description_
        pred_col: _description_
        normalizer: _description_
        group_cols: _description_

    Returns:
        _description_
    """
    rows = []

    for keys, g in df.group_by(group_cols, maintain_order=True):
        # keys is a tuple if multiple columns, scalar if one
        if not isinstance(keys, tuple):
            keys = (keys,)
        wer_v, cer_v = compute_wer_cer(
            g, text_col=text_col, pred_col=pred_col, normalizer=normalizer
        )
        row = dict(zip(group_cols, keys))
        row.update({"wer": wer_v, "cer": cer_v})
        rows.append(row)

    return pl.DataFrame(rows)

def merge_with_meta(corpus: str, model: str, scale_col: list | str):
    """
    Merge with metadata

    Args:
        corpus: _description_
        model: _description_
        scale_col: _description_

    Returns:
        _description_
    """
    normalizer = BasicTextNormalizer()
    df = pl.read_parquet(f"./data/merged/{corpus}_{model}.parquet")
    if isinstance(scale_col, list):
        scale_sel = scale_col
    else:
        scale_sel = [scale_col]

    if corpus == "discourse":
        # PANSS total = all PANSS10G columns - G2 - G6
        df = df.filter(pl.col('suffix') == 'Baseline')
        df = df.filter(pl.col('task') == 'Pictures')
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
        df = df.filter(pl.col('suffix') == 'TOPSY-0')
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
    
    # merge on pid level
    out = (
        df
        .group_by(["pid"])
        .agg([
            (
                pl.col("prediction_concat")
                .str.join(" ")
                .str.to_lowercase()
            ).alias("prediction_full"),

            (
                pl.col("transcription_concat")
                .str.join(" ")
                .str.to_lowercase()
            ).alias("transcription_full"),

            pl.col("source_audio").first().alias("source_audio"),
        ])
    )
    out.write_parquet(f"./data/merged/{corpus}_pid_picture_baseline.parquet")
    
    merged = out.join(meta_df, on='pid', how='left')    
    
    metrics_df = compute_group_metrics(
        merged,
        text_col="transcription_full",
        pred_col="prediction_full",
        normalizer=normalizer,
        group_cols=['pid']
    )

    final_cols = ['pid', 'label', 'panss-total', 'TLIDISORG', 'TLITOTAL']
    final_df = metrics_df.join(
        meta_df.select(final_cols),
        on="pid",
        how="left",
    )
    final_df = final_df.drop_nulls()
    print(f"Number of unique pid: {final_df['pid'].n_unique()}")
    print(f"Number of rows: {final_df.height}")

    rho, p_value = spearmanr(final_df['wer'].to_numpy(), final_df['TLIDISORG'].to_numpy())
    print(f"TLI disorgnization rho on {corpus}: {round(rho, 2)}; p-value: {round(p_value, 4)}")
    print("-----------")
    rho, p_value = spearmanr(final_df['wer'].to_numpy(), final_df['TLITOTAL'].to_numpy())
    print(f"TLI total rho on {corpus}: {round(rho, 2)}; p-value: {round(p_value, 4)}")
    print("-----------")
    rho, p_value = spearmanr(final_df['wer'].to_numpy(), final_df['panss-total'].to_numpy())
    print(f"PANSS total rho on {corpus}: {round(rho, 2)}; p-value: {round(p_value, 4)}")
    print("-----------")
    
    return final_df


if __name__ == "__main__":
    config_parser = configparser.ConfigParser()
    config_parser.read("config.ini")

    MODELS = ['wav2vec2-large-960h', 'hubert-large-ls960', 'whisper-large-v3']
    CORPUS =  ['topsy', 'discourse']

    get_utter_df()
    get_asr()
    compute_errors()
    reformat_data()
    for corpus in CORPUS:
        df = merge_with_meta(
            corpus='discourse',
            model='whisper',
            scale_col=['TLIDISORG', 'TLITOTAL'])
        df.write_parquet(f"./data/merged/{corpus}_pid_picture_baseline_meta.parquet")
        print(df.head())
        print("=====================")
