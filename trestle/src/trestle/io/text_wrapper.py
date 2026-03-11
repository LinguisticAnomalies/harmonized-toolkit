from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Callable, Iterator
from trestle.io.batch_wrapper import BatchWrapperBase

@dataclass
class TaskBoundary:
    """
    Definition of an utterance boundary for a task
    """
    name: str
    content_mark: Callable[[str], str] 

@dataclass
class TextBatch:
    corpus: str
    subset: str | None
    suffix: str | None
    pairs: list[tuple[Path, Path]]  # (cha, wav)

class ChaTextWrapper(BatchWrapperBase):
    """
    Wrapper for processing a SINGLE TalkBank corpus
    """
    def __init__(self, corpus, text_root, out_root, audio_root, task_boundaries):
        super().__init__(
            corpus=corpus,
            root=text_root,
            out_root=out_root,
            modality_dir="text",
        )
        self.audio_root = Path(audio_root)
        self.task_boundaries = task_boundaries
    
    def _iter_files(self):
        return self.root.rglob("*.cha")

    def _make_batch(self, subset, suffix, cha_files):
        pairs = []

        for cha in cha_files:
            audio_dir = self.audio_root / self.corpus
            if subset:
                audio_dir /= subset
            if suffix:
                audio_dir /= suffix

            audio = audio_dir / f"{cha.stem}.wav"
            if audio.exists():
                pairs.append((cha, audio))

        if not pairs:
            return None

        return TextBatch(
            corpus=self.corpus,
            subset=subset,
            suffix=suffix,
            pairs=pairs,
        )

    def run(self, cha_processor_cls, txt_patterns, format="parquet"):
        for batch in self.iter_batches():
            out_dir = self.resolve_out_dir(batch)

            cha_files = [c for c, _ in batch.pairs]
            audio_files = [a for _, a in batch.pairs]

            processor = cha_processor_cls(
                txt_patterns=txt_patterns,
                files=cha_files,
                audio_files=audio_files,
                out_dir=out_dir,
            )

            for task in self.task_boundaries:
                name_parts = []
                if batch.suffix:
                    name_parts.append(batch.suffix)
                name_parts.append(task.name)

                out_file = "_".join(name_parts)

                content_mark = (
                    task.content_mark()
                    if callable(task.content_mark)
                    else task.content_mark
                )

                processor.clean_cha(
                    out_file=out_file,
                    content_mark=content_mark,
                    format=format,
                )