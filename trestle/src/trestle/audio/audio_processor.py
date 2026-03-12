import os
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from tqdm import tqdm
import torchaudio
import torch
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
            root=text_root,
            out_root=out_root,
            modality_dir='clips'
        )

        self.dry_run = dry_run
        self.format = format
    
    def _iter_files(self):
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
            meta_path = out_dir / f'metadata_.{self.format}'

            records: list[dict[str, Any]]=[]

            for text_file in tqdm(batch.text_files, desc=f"Clipping audio in {self.corpus}"):
                task = text_file.stem.split("_")[0]
                
                input_df = pl.read_parquet(text_file).head(n=100)
                for (pid,), pid_df in input_df.group_by("pid"):
                    for n, row in enumerate(pid_df.iter_rows(named=True)):
                        src = Path(row["audio_path"])

                        clip_name = f"{task}_{pid}_{n}.wav"
                        clip_path = out_dir / clip_name

                        if self.dry_run:
                            # print(f"[DRY] {src} -> {clip_path}")
                            records.append(
                                {
                                    "clip_path": str(clip_path),
                                    "pid": pid,
                                    "task": task,
                                    "text": row["text"],
                                    "source_audio": str(src.resolve()),
                                }
                            )
                            continue

                        audio = AudioSegment.from_file(src)
                        start = row['start']
                        end = row['end']
                        sliced_audio = audio[start:end]
                        sliced_audio.export(clip_path, format='wav')

                        records.append(
                            {
                                "clip_path": str(clip_path.relative_to(self.out_root)),
                                "pid": pid,
                                "task": task,
                                "text": row["text"],
                                "source_audio": str(src.resolve()),
                            }
                        )
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


class AudioUtils:
    @staticmethod
    def init_env_for_mp() -> None:
        """
        - Silence huggingface/tokenizers fork warning.
        - Use 'spawn' to avoid forking after HF objects are created.
        """
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            import multiprocessing as mp
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)
        except Exception:
            pass

    @staticmethod
    def ensure_ffmpeg_backend() -> None:
        try:
            torchaudio.set_audio_backend("ffmpeg")
        except Exception as e:
            raise RuntimeError(
                "torchaudio must use the 'ffmpeg' backend (install torchaudio==2.8 with FFmpeg)."
            ) from e

    @staticmethod
    def fast_duration(path: str) -> float:
        """Probe duration without full decode."""
        info = torchaudio.info(path)
        if info.sample_rate <= 0:
            return 0.0
        return info.num_frames / info.sample_rate


class AudioDataset(Dataset):
    def __init__(self, items: list[tuple[str, float]], target_sr: int = 16_000):
        self.items = items
        self.target_sr = int(target_sr)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, _ = self.items[idx]
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            sr = self.target_sr
        wav_np = wav.squeeze(0).to(torch.float32).cpu().numpy()
        return {"path": path, "waveform": wav_np, "sr": sr}