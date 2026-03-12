from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoModelForCTC,
    AutoFeatureExtractor
)
from trestle.io.batch_wrapper import BatchWrapperBase

@dataclass
class ClipBatch:
    corpus: str
    subset: str | None
    suffix: str | None
    meta_files: list[Path]


class CTCPipeline(BatchWrapperBase):
    def __init__(
            self,
            model_name: str,
            corpus: str,
            root: Path,
            out_root: Path,
            device: str | torch.device = "cuda",
            batch_size: int = 8,
            format: str = 'wav',
            out_format: str = 'parquet',
            dry_run: bool = False,
            use_flash_attn2: bool = True):
        super().__init__(
            corpus=corpus,
            root=Path(root) / corpus,
            out_root=Path(out_root),
            modality_dir="clips",
        )

        self.batch_size = batch_size
        self.device = torch.device(device)
        self.format = format
        self.out_format = out_format
        self.dry_run = dry_run
        self.device = self.device
        self.model_name = model_name
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        attn_impl = "flash_attention_2" if use_flash_attn2 else None

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(
            model_name,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map="cuda" if self.device.type == "cuda" else None,
            attn_implementation=attn_impl)
        self.model.eval()

    def _iter_files(self):
        return self.root.rglob(f"metadata.parquet")

    def _make_batch(self, subset, suffix, files):
        return ClipBatch(
            corpus=self.corpus,
            subset=subset,
            suffix=suffix,
            meta_files=files,
        )

    def _collate(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None, None

        waveforms = [b["waveform"] for b in batch]
        sr = batch[0]["sampling_rate"]

        inputs = self.processor(
            waveforms,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        return {
            k: v.to(self.device, dtype=self.dtype)
            for k, v in inputs.items()}, batch

    @torch.no_grad()
    def run(
            self,
            dataset_cls, limit: int | None = None):
        """
        dataset_cls: callable(meta_path) -> AudioClipData
        """
        model_base = self.model_name.split("/")[-1]
        for batch in self.iter_batches():
            out_dir = self.resolve_out_dir(batch)
            out_path = out_dir / f"{model_base}_output.{self.out_format}"

            records = []

            for meta_path in batch.meta_files:
                dataset = dataset_cls(meta_path, limit=limit)

                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=self._collate,
                )

                for inputs, samples in tqdm(
                    loader, desc=f"CTC inference on {self.corpus}",
                    leave=False
                ):
                    if inputs is None:
                        continue

                    logits = self.model(**inputs).logits
                    pred_ids = torch.argmax(logits, dim=-1)
                    preds = self.processor.batch_decode(pred_ids)

                    for sample, pred in zip(samples, preds):
                        records.append(
                            {
                                "audio_path": sample["clip_path"],
                                "prediction": pred,
                                "transcription": sample.get("transcription").upper()
                                if sample.get("transcription")
                                else None,
                            }
                        )

            df = pl.DataFrame(records)

            if self.dry_run:
                print(f"[DRY] {out_path}")
                print(df)
            else:
                if self.out_format == "parquet":
                    df.write_parquet(out_path)
                elif self.out_format == 'csv':
                    df.write_csv(out_path)
                elif self.out_format == 'jsonl':
                    df.write_csv(out_path)

class Seq2SeqPipeline:
    def __init__(self):
        pass