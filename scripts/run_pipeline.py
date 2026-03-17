import os
import sys
import argparse
import yaml
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_data
from src.data.cleaner import clean_data
from src.features.builder import build_features
from src.mining.clustering import run_clustering
from src.models.supervised import run_supervised
from src.evaluation.metrics import evaluate_model


def run_all(config_path='configs/params.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    raw_path = cfg['raw_data_path']
    processed_path = cfg['processed_data_path']

    print('> Load data')
    df = load_data(raw_path)

    print('> Clean data')
    df_clean = clean_data(df, cfg)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_clean.to_csv(processed_path, index=False)

    print('> Build features')
    df_feat = build_features(df_clean, cfg)

    print('> Mining/clustering')
    run_clustering(df_feat, cfg)

    print('> Supervised modeling')
    model, results = run_supervised(df_feat, cfg)

    print('> Evaluation')
    evaluate_model(model, df_feat, cfg)

    print('Pipeline complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full data mining pipeline')
    parser.add_argument('--config', '-c', default='configs/params.yaml', help='Path to params YAML')
    args = parser.parse_args()

    run_all(args.config)
