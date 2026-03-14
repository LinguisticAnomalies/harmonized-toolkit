from pathlib import Path
from tqdm import tqdm
from typing import (
    Literal
)
import re
import polars as pl


class ChaProcessor:
    def __init__(
            self,
            txt_patterns: dict[str, any],
            files: list[str],
            out_dir: Path,
            audio_files: list[Path] | None=None):
        """

        Args:
            txt_patterns: a dict with regex patterns for preprocessing
            files: a list of files
            out_dir: output directory
            audio_files: the corresponding audio files
        """
        self.txt_patterns = txt_patterns
        self.files = files
        self.out_dir = Path(out_dir)
        self.audio_files = audio_files or []

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.audio_map = {
            p.stem: p for p in self.audio_files
        }
    
    def clean_text(self, text: str, speaker: str='PAR'):
        """
        basic pre-processing for .cha transcripts
        :param text: the transcript for pre-processing
        :type text: str
        """
        pattern = re.search(r"(\d+_\d+)", text)
        start, end = 0, 0
        if pattern:
            start, end = map(int, pattern.group(1).split('_'))
            text = re.sub(pattern.group(1), '', text)
        text = re.sub(speaker, '', text)
        for pattern, replacement in self.txt_patterns.items():
            try:
                pattern = re.compile(pattern)
                text = re.sub(pattern, replacement, text)
            except re.error:
                pass
        return start, end, text.lower().strip()
    
    def clean_cha(
            self,
            out_file: str,
            format: Literal["parquet", "jsonl", "csv"] = "parquet",
            speaker: str='PAR',
            content_mark: str | None = None):
        """
        1. Clean .cha files given an optional content mark.
        2. Save the preprocessed output to the designated location as per utterance or per participant

        Args:
            out_file: the basename of the output file
            format: the format of the output file. Defaults to 'parquet'.
                    Supporting format: parquet, jsonl, csv
            speaker: the speaker's mark in .cha. Defaults to 'PAR'.
            content_mark: specific task mark in .cha files. Defaults to None.
        """
        out_path = self.out_dir / Path(out_file)
        out_path = out_path.with_suffix(f".{format}")

        all_records = []

        for cha_file in tqdm(self.files, desc='Porcessing .cha files', total=len(self.files)):
            file_name = Path(cha_file).stem
            audio_path = self.audio_map.get(file_name)
            
            with open(cha_file, encoding='utf-8') as f:
                text = f.read()
            if content_mark:
                match = re.search(content_mark, text, re.DOTALL)
                text = match.group() if match else ""
                if not match:
                    print(f"No content_mark match in {cha_file}")
            
            text = re.sub(r"\n\s+", "\n", text)

            speaker_pat = re.compile(rf"\*{speaker}:")
            
            for line in text.splitlines():
                if not speaker_pat.match(line):
                    continue

                start, end, new_sent = self.clean_text(line, rf"\*{speaker}:\s+")
                if not new_sent:
                    continue

                all_records.append({
                    "start": start,
                    "end": end,
                    "text": new_sent,
                    "pid": file_name,
                    "audio_path": str(audio_path) if audio_path else None,
                })

        # utterance-level df
        df_utt = pl.DataFrame(all_records)
        # participant-level df
        df_pid = (
            df_utt
            .group_by("pid")
            .agg(
                pl.col("text").str.join(" ").alias("text"),
                pl.col("audio_path").drop_nulls().first().alias("audio_path"),
            )
        )

        # save to local file
        out_base = self.out_dir / out_file
        if format == 'parquet':
            df_utt.write_parquet(out_base.with_name(f"{out_file}_utterance.parquet"))
            df_pid.write_parquet(out_base.with_name(f"{out_file}_participant.parquet"))
        elif format == 'csv':
            df_utt.write_csv(out_base.with_name(f"{out_file}_utterance.csv"))
            df_pid.write_csv(out_base.with_name(f"{out_file}_participant.csv"))
        elif format =='jsonl':
            df_utt.write_ndjson(out_base.with_name(f"{out_file}_utterance.jsonl"))
            df_pid.write_ndjson(out_base.with_name(f"{out_file}_participant.jsonl"))