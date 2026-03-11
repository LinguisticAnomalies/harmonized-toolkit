import warnings
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
import torchaudio
from trestle.io.batch_wrapper import BatchWrapperBase
warnings.filterwarnings('ignore')

@dataclass
class AudioBatch:
    corpus: str
    subset: str
    suffix: str | None
    audio_files: list[Path]

class AudioWrapper(BatchWrapperBase):
    def __init__(
        self,
        corpus: str,
        audio_root: Path,
        out_root: Path,
        source_format: str = "mp3",
        target_format: str = "wav",
        target_sr: int = 16_000,
        dry_run: bool = False,
    ):
        super().__init__(
            corpus=corpus,
            root=audio_root,
            out_root=out_root,
            modality_dir="audio",
        )
        self.source_format = source_format
        self.target_format = target_format
        self.target_sr = target_sr
        self.dry_run = dry_run

    def _iter_files(self):
        return self.root.rglob(f"*.{self.source_format}")

    def _make_batch(self, subset, suffix, audio_files):
        return AudioBatch(
            corpus=self.corpus,
            subset=subset,
            suffix=suffix,
            audio_files=audio_files,
        )

    def run(self):
        for batch in self.iter_batches():
            out_dir = self.resolve_out_dir(batch)

            for src in tqdm(batch.audio_files, desc="Processing audio"):
                out_path = out_dir / src.with_suffix(
                    f".{self.target_format}"
                ).name

                if self.dry_run:
                    print(f"[DRY] {src} -> {out_path}")
                    continue

                waveform, sr = torchaudio.load(src)

                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                if sr != self.target_sr:
                    waveform = torchaudio.functional.resample(
                        waveform, sr, self.target_sr
                    )

                torchaudio.save(
                    out_path,
                    waveform,
                    self.target_sr,
                    format=self.target_format,
                )
                