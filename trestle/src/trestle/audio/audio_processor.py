from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Literal
from dataclasses import dataclass
from tqdm import tqdm
import torchaudio
from torch.utils.data import Dataset
import torchaudio
import polars as pl
from trestle.io import BatchWrapperBase
from trestle.io.audio_utils import clip_audio_batch

@dataclass
class AudioClipBatch:
    corpus: str
    subset: str | None
    suffix: str | None
    text_files: list[Path]

class AudioClipper(BatchWrapperBase):
    """
    Clip audio recordings into utterance-level clips based on the timestamp provided by the text files
    """
    def __init__(
            self,
            corpus: str,
            text_root: Path,
            out_root: Path,
            dry_run: bool=False,
            mode: Literal["full", "task"] = "task",
            num_worksers: int=2,
            format: str='parquet'):
        super().__init__(
            corpus=corpus,
            root=Path(text_root) / corpus,
            out_root=out_root,
            modality_dir='clips'
        )
        self.mode=mode
        self.num_workers = num_worksers
        self.dry_run = dry_run
        self.format = format
    
    def _iter_files(self):
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        
        files = list(self.root.rglob(f"*_utterance.{self.format}"))

        if self.mode == "full":
            files = [item for item in files if item.name.startswith("full")]
        else:
            files = [item for item in files if not item.name.startswith("full")]

        return files
    
    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        audio_path = Path(row["clip_path"])

        waveform, sr = torchaudio.load(audio_path)

        return {
            "waveform": waveform.squeeze(0).numpy(),
            "sampling_rate": sr,
            "clip_path": str(audio_path),
            "transcription": row.get("text"),
        }

    def _make_batch(self, subset, suffix, files):
        return AudioClipBatch(
            corpus=self.corpus,
            subset=subset,
            suffix=suffix,
            text_files=files,
        )

    def run(self):
        for batch in self.iter_batches():
            out_dir = self.resolve_out_dir(batch)
            out_dir.mkdir(parents=True, exist_ok=True)

            meta_path = out_dir / f"metadata.{self.format}"
            audio_jobs: dict[Path, list[dict[str, Any]]] = {}

            for text_file in batch.text_files:
                task = text_file.stem.split("_")[0]
                df = pl.read_parquet(text_file).sort("start")

                for n, row in enumerate(df.iter_rows(named=True)):
                    src = Path(row["audio_path"])
                    if not src.exists():
                        continue

                    clip_path = out_dir / f"{task}_{row['pid']}_{row['start']}_{row['end']}.wav"

                    audio_jobs.setdefault(src, []).append({
                        "pid": row["pid"],
                        "text": row["text"],
                        "start": row["start"],
                        "end": row["end"],
                        "clip_path": clip_path,
                    })

            if self.dry_run:
                num_clips = sum(len(clips) for clips in audio_jobs.values())
                num_recordings = len(audio_jobs)
                print(
                    f"[DRY RUN] corpus={self.corpus}\n"
                    f"mode={self.mode}\n"
                    f"recordings={num_recordings}\n"
                    f"clips={num_clips}\n"
                    f"output_dir={out_dir}\n"
                )
                continue

            records: list[dict[str, Any]] = []
            for job in tqdm(
                audio_jobs.items(),
                total=len(audio_jobs),
                desc=f"Clipping audio ({self.corpus})",
            ):
                batch_records = clip_audio_batch(job)
                records.extend(batch_records)

            if records:
                meta_df = pl.DataFrame(records)

                if self.format == "parquet":
                    meta_df.write_parquet(meta_path)
                elif self.format == "jsonl":
                    meta_df.write_ndjson(meta_path)
                elif self.format == "csv":
                    meta_df.write_csv(meta_path)
		


class AudioClipDataset(Dataset):
    def __init__(
        self,
        meta_path: Path,
        root: Path | None = None,
        subset: str | None = None,
        limit: int | None = None
    ):
        """
        meta_path: path to metadata.parquet
        root: base dir for relative clip_path (defaults to meta_path parent)
        return_text: return reference text if available
        """
        self.meta_path = Path(meta_path)
        self.root = root or self.meta_path.parent
        df = pl.read_parquet(self.meta_path)

        if subset is None:
            subset = self.meta_path.parent.name

        self.subset = subset
        if "subset" in df.columns:
            df = df.filter(pl.col("subset") == subset)

        if limit is not None:
            df = df.head(limit)

        self.df = df

    def __len__(self):
        return self.df.height
    
    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        audio_path = self.root / row["clip_path"]
        try:
            waveform, sr = torchaudio.load(audio_path)
        except RuntimeError:
            print(f"Decode failed: {audio_path}")
            return None

        sample = {
            "waveform": waveform.squeeze(0).numpy(),
            "sampling_rate": sr,
            "clip_path": str(audio_path),
            "transcription": row.get("text"),
        }

        return sample