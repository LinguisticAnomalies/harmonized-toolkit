import os
import json
from tqdm import tqdm
import requests
from time import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from typing import Any
import polars as pl

class Utility:
    @staticmethod
    def derive_target_schema(
        df: pl.DataFrame,
        uid_col: str = "output_file",
    ) -> list[tuple[str, pl.DataType]]:
        if df.is_empty() and not df.columns:
            raise ValueError("[ERROR] Existing parquet has no schema (empty columns).")

        cols = df.columns
        dtypes = df.dtypes

        if uid_col not in cols:
            raise ValueError(
                f"[ERROR] Existing parquet must contain '{uid_col}' column."
            )

        return list(zip(cols, dtypes))

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

        schema_names = [name for name, _ in schema]

        if uid_col not in df.columns:
            raise ValueError(
                f"Input data frame must contain '{uid_col}' column."
            )

        idx_uid = schema_names.index(uid_col)
        uid_dtype = schema[idx_uid][1]
        df = df.with_columns(pl.col(uid_col).cast(uid_dtype))

        select_exprs = []
        for name, dtype in schema:
            if name in df.columns:
                select_exprs.append(pl.col(name).cast(dtype).alias(name))
            else:
                select_exprs.append(pl.lit(None).cast(dtype).alias(name))

        return df.select(select_exprs)


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
        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec
        self.connection_close = connection_close
        self.log_prefix = log_prefix
        self.uid_col = uid_col
        self.text_col = text_col
        self.server_uid_col = server_uid_col
        self.server_text_col = server_text_col

        self._session = requests.Session()
    
    def _call_api(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        API call with new server code

        Args:
            items: _description_

        Returns:
            _description_
        """
        body = {"items": items}

        # 24 hours
        resp = self._session.post(
            self.url + "infer",
            json=body,
            timeout=(5, 24*60*60),
        )
        resp.raise_for_status()

        data = resp.json()
        return data.get("rows", [])

    def _process_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        for r in batch:
            uid = r.get(self.uid_col)
            text = r.get(self.text_col)

            if uid is None or text is None or not str(text).strip():
                continue

            items.append({
                self.server_uid_col: uid,
                self.server_text_col: text,
            })

        if not items:
            return pl.DataFrame()
    
        rows = self._call_api(items)
        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)
        
        if self.server_uid_col in df.columns:
            df = df.rename(
                {self.server_uid_col: self.uid_col})
        
        print("sent:", len(items), "received:", len(rows))

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

            target_schema = Utility.derive_target_schema(
                existing_df, uid_col=self.uid_col
            )
            existing_aligned = Utility.align_to_schema(
                existing_df, target_schema, uid_col=self.uid_col
            )
            new_aligned = Utility.align_to_schema(
                new_df, target_schema, uid_col=self.uid_col
            )

            combined = pl.concat(
                [existing_aligned, new_aligned], how="vertical"
            ).unique(subset=[self.uid_col], keep="last")
        else:
            target_schema = Utility.derive_target_schema(
                new_df, uid_col=self.uid_col
            )
            combined = Utility.align_to_schema(
                new_df, target_schema, uid_col=self.uid_col
            )

        combined.write_parquet(output_parquet)
    
    def append_missing_features(
        self,
        input_parquet: str,
        output_parquet: str,
        filter_col: str | None=None,
        rerun: bool = False,
    ):
        input_df = (
            pl.read_parquet(input_parquet)
            #.select([self.uid_col, self.text_col])
            .filter(
                pl.col(self.text_col).is_not_null()
                & (pl.col(self.text_col).str.len_bytes() > 0)
            )
        )
        if filter_col is not None:
            input_df = input_df.filter(pl.col('suffix') == filter_col)

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

        records = missing_df.to_dicts()
        batches = [
            records[i:i + self.batch_size]
            for i in range(0, len(records), self.batch_size)
        ]

        flush_buffer: list[pl.DataFrame] = []
        FLUSH_ROWS = 100
        buffer_rows = 0

        for batch in tqdm(
            batches,
            desc=f"Processing [{self.log_prefix}]",
            mininterval=60,
        ):
            df = self._process_batch(batch)

            if df.is_empty():
                continue

            flush_buffer.append(df)
            buffer_rows += len(batch)

            if buffer_rows >= FLUSH_ROWS:
                self._flush_feature_results(flush_buffer, output_parquet)
                flush_buffer.clear()
                buffer_rows = 0

        # final flush
        if flush_buffer:
            self._flush_feature_results(flush_buffer, output_parquet)
                