import os
import time
import pandas as pd
import json
import numpy as np
import yaml
import logging
import ast


def get_json_data(dataset_location: str) -> dict:
    if os.path.exists(dataset_location):
        with open(dataset_location, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    else:
        raise FileNotFoundError(f"File {dataset_location} not found.")
    return json_data


def create_df(json) -> pd.DataFrame:
    unnecessary_vars = [
        'packages', 'screenshots', 'movies', 'score_rank', 'header_image',
        'reviews', 'website', 'support_url', 'notes', 'support_email',
        'user_score', 'required_age', 'metacritic_score',
        'metacritic_url',  'detailed_description', 'about_the_game',
        'windows', 'mac', 'linux', 'achievements', 'full_audio_languages',
        'dlc_count', 'supported_languages', 'developers', 'publishers'
    ]
    games = [{
        **{k: v for k, v in game_info.items() if k not in unnecessary_vars},
        'tags': list(tags.keys()) if isinstance((tags := game_info.get('tags', {})), dict) else [],
        'tag_frequencies': list(tags.values()) if isinstance(tags, dict) else [],
        'app_id': app_id
    } for app_id, game_info in json.items()]

    # Create a DataFrame from the processed list
    df = pd.DataFrame(games)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:

    # remove games with zero owners or zero reviews and categories
    df = df[~((df['estimated_owners'] == "0 - 0") | (df['positive'] +
                                                     df['negative'] == 0) | (df['categories'].str.len() == 0))]

    # Split estimated_owners into two: min_owners and max_owners
    df = df.copy()
    df[['min_owners', 'max_owners']] = df['estimated_owners'].str.split(
        ' - ', expand=True)

    # Remove the original field
    df = df.drop('estimated_owners', axis=1)

    df['positive_percent'] = df['positive'] / \
        (df['positive'] + df['negative']) * 100

    return df


def stream_data_chronologically(file_path, batch_size, output_dir, date_column, delay=0, verbose=True):
    if verbose:
        logging.info(
            f"Streaming data from {file_path} in batches of {batch_size}...")
    df = pd.read_csv(file_path)

    df[date_column] = pd.to_datetime(df['release_date'], errors='coerce')

    df = df.sort_values(by=date_column)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        batch.to_csv(f"{output_dir}/batch_{i // batch_size}.csv", index=False)
        time.sleep(delay)  # Имитируем задержку в 1 секунду между пакетами
        if verbose:
            print(f"Batch {i // batch_size} saved.", end='\r')


def get_batch(file_path, batch_size, date_column, batch_number=0, output_dir=None, verbose=True):
    if verbose:
        logging.info(
            f"Fetching batch number {batch_number} from {file_path}...")
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.sort_values(by=date_column)

    start_index = batch_number * batch_size
    end_index = start_index + batch_size
    batch = df.iloc[start_index:end_index]

    if batch.empty:
        if verbose:
            logging.info(
                f"No more batches available. Batch number {batch_number} is empty.")
        return None  # No more batches available

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        batch_file_path = f"{output_dir}/batch_{batch_number}.csv"
        batch.to_csv(batch_file_path, index=False)
        if verbose:
            logging.info(f"Batch {batch_number} saved to {batch_file_path}.")

    batch_number += 1
    update_batch_number('config.yaml', batch_number)

    return batch

def update_batch_number(config_path, new_batch_number):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['current_batch_number'] = new_batch_number
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
    

def load_from_config():
    config_location = 'config.yaml'
    with open(config_location, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Dataset quality check
def data_quality_metrics(df):
    metrics = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "outliers": {col: ((df[col] < df[col].quantile(0.01)) | (df[col] > df[col].quantile(0.99))).sum()
                     for col in df.select_dtypes(include=["float64", "int64"]).columns}
    }
    return metrics


def log_data_quality(df):
    metrics = data_quality_metrics(df)
    logging.info("Data Quality Metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("dataset_utils.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting the dataset processing script.")
    config = load_from_config()
    json_data = get_json_data(config['dataset_location'])
    logging.info(f"Loaded JSON data from {config['dataset_location']}.")
    df = create_df(json_data)
    df = clean_df(df)
    logging.info("DataFrame created and cleaned.")
    df.to_csv(config['csv_file'], index=False)
    # stream_data_chronologically(
    #     file_path=config['csv_file'],
    #     batch_size=config['batch_size'],
    #     output_dir=config['batch_storage'],
    #     date_column="release_date"
    # )

    get_batch(
        file_path=config['csv_file'],
        batch_size=config['batch_size'],
        date_column="release_date",
        batch_number=config["current_batch_number"],
        output_dir=config['batch_storage'],
    )

    logging.info("Data streaming completed.")
