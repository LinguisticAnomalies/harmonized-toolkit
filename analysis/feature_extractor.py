import os
import time
from tqdm import tqdm
from typing import Any
import requests
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
import polars as pl

class Utility:
    @staticmethod
    def derive_target_schema(
        df: pl.DataFrame,
        uid_col: str = "output_file",
    ) -> list[tuple[str, pl.DataType]]:
        if uid_col not in df.columns:
            raise ValueError(
                f"[ERROR] Existing parquet must contain '{uid_col}' column."
            )

        return list(zip(df.columns, df.dtypes))

    @staticmethod
    def align_to_schema(
        df: pl.DataFrame,
        schema: list[tuple[str, pl.DataType]],
        uid_col: str = "output_file",
    ) -> pl.DataFrame:
        if df.is_empty():
            return pl.DataFrame(
                {name: pl.Series(name, [], dtype) for name, dtype in schema}
            )

        schema_dict = dict(schema)

        if uid_col not in df.columns:
            raise ValueError(
                f"Input data frame must contain '{uid_col}' column."
            )

        df = df.with_columns(pl.col(uid_col).cast(schema_dict[uid_col]))

        df = df.with_columns([
            pl.col(c).cast(schema_dict[c]) if c in df.columns
            else pl.lit(None).cast(schema_dict[c]).alias(c)
            for c in schema_dict
        ])

        return df.select(schema_dict.keys())


class FeatureExtractorClient:
    """
    Microserver-based feature extraction
    """
    def __init__(
        self,
        url: str,
        log_prefix: str,
        content_type: str = "application/json",
        batch_size: int = 64,
        workers: int = 2,

        max_retries: int = 1,
        backoff_base_sec: float = 0.5,
        connection_close: bool = True,
        uid_col: str = "output_file",
        text_col: str = "prediction",
        server_uid_col: str = "filename",
        server_text_col: str = "text",
    ):
        self.url = url.rstrip("/") + "/"
        self.content_type = content_type
        self.batch_size = batch_size
        self.workers = workers
        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec
        self.connection_close = connection_close
        self.log_prefix = log_prefix
        self.uid_col = uid_col
        self.text_col = text_col
        self.server_uid_col = server_uid_col
        self.server_text_col = server_text_col

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def _call_api(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    self.url + "infer",
                    json={"items": items},
                    timeout=(5, 60*60*6),
                )
                resp.raise_for_status()
                return resp.json()["rows"]
            except Exception as e:
                if attempt + 1 == self.max_retries:
                    print(f"[{self.log_prefix}] ERROR API call failed: {e}")
                    return []
                time.sleep(self.backoff_base_sec * (attempt + 1))

    def _process_batch(self, batch: list[dict[str, Any]]) -> pl.DataFrame:
        items = [
            {
                self.server_uid_col: r[self.uid_col],
                self.server_text_col: r[self.text_col],
            }
            for r in batch
            if r.get(self.uid_col) and r.get(self.text_col) and str(r[self.text_col]).strip()
        ]

        if not items:
            return pl.DataFrame()

        rows = self._call_api(items)
        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)

        if self.server_uid_col in df.columns:
            df = df.rename({self.server_uid_col: self.uid_col})

        return df
    
    def _flush_feature_results(
        self,
        dfs: list[pl.DataFrame],
        output_parquet: str,
    ):
        if not dfs:
            return

        new_df = pl.concat(dfs, how="vertical")

        if os.path.exists(output_parquet):
            existing_df = pl.read_parquet(output_parquet)

            existing_schema = Utility.derive_target_schema(
                existing_df, self.uid_col
            )
            new_schema = Utility.derive_target_schema(
                new_df, self.uid_col
            )

            target_schema = list(dict.fromkeys(
                existing_schema + new_schema
            ))

            existing_aligned = Utility.align_to_schema(
                existing_df, target_schema, self.uid_col
            )
            new_aligned = Utility.align_to_schema(
                new_df, target_schema, self.uid_col
            ).unique(subset=[self.uid_col], keep="last")

            combined = pl.concat(
                [existing_aligned, new_aligned], how="vertical"
            )
        else:
            target_schema = Utility.derive_target_schema(
                new_df, self.uid_col
            )
            combined = Utility.align_to_schema(
                new_df, target_schema, self.uid_col
            ).unique(subset=[self.uid_col], keep="last")

        combined.write_parquet(output_parquet)

    def append_missing_features(
        self,
        input_parquet: str,
        output_parquet: str,
        rerun: bool = False,
    ):
        input_df = (
            pl.read_parquet(input_parquet)
            .select([self.uid_col, self.text_col])
            .filter(
                pl.col(self.text_col).is_not_null()
                & (pl.col(self.text_col).str.len_bytes() > 0)
            )
        )

        if rerun or not os.path.exists(output_parquet):
            existing_df = pl.DataFrame([])
        else:
            existing_df = pl.read_parquet(output_parquet)

        if existing_df.is_empty():
            missing_df = input_df
        else:
            existing_keys = existing_df.select(self.uid_col).unique()
            missing_df = input_df.join(existing_keys, on=self.uid_col, how="anti")

        if missing_df.is_empty():
            print(f"[{self.log_prefix}] INFO nothing to append.")
            return

        records = list(missing_df.iter_rows(named=True))
        batches = [
            records[i:i + self.batch_size]
            for i in range(0, len(records), self.batch_size)
        ]

        flush_buffer: list[pl.DataFrame] = []
        FLUSH_SIZE = self.batch_size / 2

        for batch in tqdm(
            batches,
            desc=f"Processing [{self.log_prefix}]",
            mininterval=60,
        ):
            df = self._process_batch(batch)
            if df.is_empty():
                continue

            flush_buffer.append(df)

            if len(flush_buffer) >= FLUSH_SIZE:
                self._flush_feature_results(flush_buffer, output_parquet)
                flush_buffer.clear()

        if flush_buffer:
            self._flush_feature_results(flush_buffer, output_parquet)