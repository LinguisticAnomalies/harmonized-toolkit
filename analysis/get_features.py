from pathlib import Path
import configparser
from feature_extractor import FeatureExtractorClient

CORPUS =  ['topsy', 'discourse']
FILTER_COLS =  {'discourse': 'Baseline', 'topsy': 'TOPSY-0'}

def get_ccc_subset(feature, corpus):
    input_file = f"./data/merged/{corpus}_pid_pictures_baseline.parquet"
    out_file = f"./data/ccc/{corpus}_pid_pictures_baseline_{feature}_ccc.parquet"
    ccc_client = FeatureExtractorClient(
        url="http://localhost:5003/",
            content_type="application/json",
            log_prefix=f'ccc_{feature}',
            text_col=feature,
            uid_col='pid',
            batch_size=32,
        )
    ccc_client.append_missing_features(
        input_parquet=input_file,
        output_parquet=out_file,
    )


if __name__ == "__main__":
    cfg= configparser.ConfigParser()
    cfg.read("config.ini")

    get_ccc_subset(feature='transcription', corpus='topsy')
    get_ccc_subset(feature='transcription', corpus='discourse')
    get_ccc_subset(feature='prediction', corpus='topsy')
    get_ccc_subset(feature='prediction', corpus='discourse')
    #get_ccc('prediction_concat')