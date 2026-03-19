from pathlib import Path
import configparser
from feature_extractor import FeatureExtractorClient

CORPUS =  ['topsy', 'discourse']
FILTER_COLS =  {'discourse': 'Baseline', 'topsy': 'TOPSY-0'}

def get_ccc_subset(feature, corpus):
    input_file = f"./data/merged/{corpus}_pid_picture_baseline.parquet"
    out_file = f"./data/ccc/{corpus}_pid_picture_baseline_{feature}_ccc.parquet"
    ccc_client = FeatureExtractorClient(
        url="http://localhost:5003/",
            content_type="application/json",
            log_prefix=f'ccc: {corpus}',
            text_col=feature,
            uid_col='pid',
            batch_size=32
        )
    ccc_client.append_missing_features(
        input_parquet=input_file,
        output_parquet=out_file,
    )

def get_ccc(feature, corpus):
    out_dir = Path(cfg['outputs']['features'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / Path(f"{corpus}_{feature}_ccc.parquet")
    ccc_client = FeatureExtractorClient(
        url="http://localhost:5003/",
            content_type="application/json",
            log_prefix=f'ccc: {corpus}',
            text_col=feature,
            uid_col='uid',
            batch_size=512
        )
    ccc_client.append_missing_features(
        input_parquet=f"./data/merged/{corpus}_whisper.parquet",
        output_parquet=out_file,
        filter_col=FILTER_COLS[corpus]
    )

if __name__ == "__main__":
    cfg= configparser.ConfigParser()
    cfg.read("config.ini")

    get_ccc_subset(feature='transcription_full', corpus='topsy')
    get_ccc_subset(feature='prediction_full', corpus='topsy')
    get_ccc_subset(feature='transcription_full', corpus='discourse')
    get_ccc_subset(feature='prediction_full', corpus='discourse')
    #get_ccc('prediction_concat')