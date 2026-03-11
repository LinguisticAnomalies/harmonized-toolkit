from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

class BatchWrapperBase(ABC):
    def __init__(
            self,
            corpus: str,
            root: Path,
            out_root: Path,
            modality_dir: str):
        self.corpus = corpus
        self.root = Path(root)
        self.out_root = Path(out_root)
        self.modality_dir = modality_dir
    
    def _infer_subset_suffix(
        self, rel: Path
    ) -> tuple[str | None, str | None]:
        parts = list(rel.parts)

        # normalize: modality/file.ext
        if len(parts) == 2 and parts[0] == self.modality_dir:
            parts = [parts[1]]

        subset = None
        suffix = None

        if len(parts) >= 3:
            if parts[-2] == self.modality_dir:
                subset = parts[-3]
            elif parts[-3] == self.modality_dir:
                subset = parts[-2]
            else:
                subset = parts[-3]
                suffix = parts[-2]
        elif len(parts) == 2:
            subset = parts[-2]

        return subset, suffix
    
    @abstractmethod
    def _iter_files(self) -> Iterator[Path]:
        pass

    @abstractmethod
    def _make_batch(self, subset, suffix, files):
        pass

    def iter_batches(self):
        buckets: dict[tuple[str | None, str | None], list[Path]] = {}

        for f in self._iter_files():
            rel = f.relative_to(self.root)
            key = self._infer_subset_suffix(rel)
            buckets.setdefault(key, []).append(f)

        for (subset, suffix), files in buckets.items():
            batch = self._make_batch(subset, suffix, files)
            if batch:
                yield batch

    def resolve_out_dir(self, batch) -> Path:
        out_dir = self.out_root / self.corpus
        if batch.subset:
            out_dir /= batch.subset
        if batch.suffix:
            out_dir /= batch.suffix
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir