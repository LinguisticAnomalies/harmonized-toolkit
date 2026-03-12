from pathlib import Path
from typing import Any
from dataclasses import dataclass
from tqdm import tqdm
import torchaudio
from torch.utils.data import Dataset
from pydub import AudioSegment
import polars as pl
from trestle.io.batch_wrapper import BatchWrapperBase

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
            format: str='parquet'):
        super().__init__(
            corpus=corpus,
            root=Path(text_root) / corpus,
            out_root=out_root,
            modality_dir='clips'
        )

        self.dry_run = dry_run
        self.format = format
    
    def _iter_files(self):
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        return self.root.rglob(f"*_utterance.{self.format}")

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
            meta_path = out_dir / f'metadata.{self.format}'

            records: list[dict[str, Any]]=[]

            for text_file in tqdm(batch.text_files, desc=f"Clipping audio in {self.corpus}"):
                task = text_file.stem.split("_")[0]

                input_df = pl.read_parquet(text_file)
                for (pid,), pid_df in input_df.group_by("pid"):
                    for n, row in enumerate(pid_df.iter_rows(named=True)):
                        src = Path(row["audio_path"])

                        clip_name = f"{task}_{pid}_{n}.wav"
                        clip_path = (out_dir / clip_name).resolve()

                        record = {
                                    "clip_path": str(clip_path),
                                    "pid": pid,
                                    "text": row["text"],
                                    "source_audio": str(src.resolve()),
                                }

                        if self.dry_run:
                            records.append(record)
                            continue

                        audio = AudioSegment.from_file(src)
                        audio_len = len(audio)
                        
                        start = row['start']
                        end = row['end']
                        # handle empty clips
                        if end <= start:
                            continue
                        if start < 0 or end > audio_len:
                            continue

                        sliced_audio = audio[start:end]

                        if len(sliced_audio) == 0:
                            continue
                        
                        sliced_audio.export(clip_path, format='wav')

                        records.append(record)
            
            if records:
                meta_df = pl.DataFrame(records)

                if self.dry_run:
                    print("=== DRY RUN: metadata preview ===")
                    with pl.Config(fmt_str_lengths=1000):
                        print(meta_df)
                else:
                    if self.format == 'parquet':
                        meta_df.write_parquet(meta_path)
                    elif self.format == "jsonl":
                        meta_df.write_ndjson(meta_path)
                    elif self.format == 'csv':
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