import argparse

import pandas as pd
from dataset_utils import create_df, clean_df, get_batch, get_json_data, load_from_config
import os
import logging
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
        logging.FileHandler("run.log"),
        logging.StreamHandler()
    ]
        )
    config = load_from_config()
    parser = argparse.ArgumentParser(description="Run different modes of the ML pipeline.")
    parser.add_argument("-mode", type=str, required=True, choices=["inference", "update", "summary"],
                        help="Mode of operation: 'inference', 'update', or 'summary'.")
    parser.add_argument("-file", type=str, required=False,
                        help="Path to the input file (required for 'inference' mode).")

    args = parser.parse_args()

    if args.mode == "inference":
        if not args.file:
            raise ValueError("The '-file' argument is required for 'inference' mode.")
        pass

    elif args.mode == "update":
        
        # if games.csv doesn't exist we create it from games.json
        if not os.path.exists("games.csv"):
            logging.info("games.csv not found. Creating it from games.json.")
            json_data = get_json_data(config['dataset_location'])
            df = create_df(json_data)
            df = clean_df(df)
            df.to_csv("games.csv", index=False)
            logging.info("games.csv created successfully.")

        # Get next batch
        logging.info("Fetching the next batch of data.")
        batch = get_batch(
            file_path=config['csv_file'],
            batch_size=config['batch_size'],
            date_column="release_date",
            batch_number=config["current_batch_number"],
            output_dir=config['batch_storage'],
            verbose=False
        )
        logging.info("Batch fetched and stored successfully.")

        ### Model creation and fitting...

        ###
    elif args.mode == "summary":
        pass