from pathlib import Path
import json
from dataclasses import dataclass
from typing import Callable
from trestle.io.batch_wrapper import BatchWrapperBase

@dataclass
class TaskBoundary:
    """
    Definition of an utterance boundary for a task
    """
    name: str
    content_mark: Callable[[], str] | str | None

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
    def __init__(
            self,
            corpus: str,
            text_root: Path,
            out_root: Path,
            audio_root: Path,
            meta_root: Path,
            task_boundaries: list[TaskBoundary] | None=None,
            strict_audio: bool = False,
            dry_run: bool=False):
        super().__init__(
            corpus=corpus,
            root=text_root,
            out_root=out_root,
            modality_dir="text",

        )
        self.audio_root = Path(audio_root) if audio_root else None
        self.meta_root = Path(meta_root) if meta_root else None
        self.task_boundaries = task_boundaries or [
            TaskBoundary(name="full", content_mark=None)
        ]

        self.dry_run = dry_run
        self.strict_audio = strict_audio

        # if no task boundaries
        if not self.task_boundaries:
            self.task_boundaries = [
                TaskBoundary(name="full", content_mark=None)
            ]
    
    def _iter_files(self):
        return self.root.rglob("*.cha")

    def _make_batch(self, subset, suffix, cha_files):
        pairs = []

        for cha in cha_files:
            # text-only mode
            if self.audio_root is None:
                pairs.append((cha, None))
                continue
            audio_dir = self.audio_root / self.corpus
            if subset:
                audio_dir /= subset
            if suffix:
                audio_dir /= suffix

            audio = audio_dir / f"{cha.stem}.wav"
            if audio.exists():
                pairs.append((cha, audio))
            else:
                if self.strict_audio:
                    print(f"[DROP] Missing audio for {cha} (expected {audio})")
                else:
                    print(f"[TEXT-ONLY] Missing audio for {cha}")
                    pairs.append((cha, None))

        if not pairs:
            return None

        return TextBatch(
            corpus=self.corpus,
            subset=subset,
            suffix=suffix,
            pairs=pairs,
        )
    
    def _write_patterns(self, txt_patterns: dict):
        path = self.meta_root / f"{self.corpus}_text_patterns.json"
        rules = [
            {"pattern": k, "replace": v}
            for k, v in txt_patterns.items()
        ]

        with open(path, "w") as f:
            json.dump(rules, f, ensure_ascii=False)


    def run(self, cha_processor_cls, txt_patterns, format="parquet"):
        for batch in self.iter_batches():
            out_dir = self.resolve_out_dir(batch)
            self._write_patterns(txt_patterns)

            cha_files = [c for c, _ in batch.pairs]
            audio_files = [a for _, a in batch.pairs]

            processor = cha_processor_cls(
                txt_patterns=txt_patterns,
                files=cha_files,
                audio_files=audio_files,
                out_dir=out_dir,
            )

            for task in self.task_boundaries:
                name_parts = [task.name]
                if batch.suffix:
                    name_parts.append(batch.suffix)

                out_file = "_".join(name_parts)

                if self.dry_run:
                    print(out_dir / f"{out_file}_utterance.{format}")
                    print(out_dir / f"{out_file}_participant.{format}")
                    continue

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