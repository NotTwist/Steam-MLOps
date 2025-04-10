import argparse
import logging
import os
from auto_eda import auto_eda
from train import train
from infer import run_inference
from dataset_utils import create_df, clean_df, get_batch, get_json_data, load_from_config, monitor_and_handle_data_drift
from summary import generate_summary_report

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("run.log"),
            logging.StreamHandler()
        ]
    )

    config = load_from_config()
    parser = argparse.ArgumentParser(
        description="Run different modes of the ML pipeline.")
    parser.add_argument("-mode", type=str, required=True, choices=["inference", "update", "summary"],
                        help="Mode of operation: 'inference', 'update', or 'summary'.")
    parser.add_argument("-file", type=str, required=False,
                        help="Path to the input file (required for 'inference' mode).")

    args = parser.parse_args()

    if args.mode == "inference":
        if not args.file:
            raise ValueError(
                "The '-file' argument is required for 'inference' mode.")
        input_file = args.file
        output_folder = config.get("infer_folder", ".")
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, os.path.splitext(
            os.path.basename(input_file))[0] + "_with_predictions.csv")

        logging.info(f"Running inference on {input_file}")
        try:
            result_path = run_inference(input_file, output_file)
            logging.info(
                f"Inference completed. Results saved to {result_path}")
        except Exception as e:
            logging.error(f"Inference failed: {e}")

    elif args.mode == "update":
        # Existing update logic
        if not os.path.exists("games.csv"):
            logging.info("games.csv not found. Creating it from games.json.")
            json_data = get_json_data(config["dataset_location"])
            df = create_df(json_data)
            df = clean_df(df)
            df.to_csv("games.csv", index=False)
            logging.info("games.csv created successfully.")

        logging.info("Fetching the next batch of data.")
        batch = get_batch(
            file_path=config["csv_file"],
            batch_size=config["batch_size"],
            date_column="release_date",
            batch_number=config["current_batch_number"],
            output_dir=config["batch_storage"],
            verbose=True
        )
        if batch is None:
            logging.error("Update Result: FAILURE")
            exit(0)

        monitor_and_handle_data_drift(
            config["batch_storage"], config["current_batch_number"], batch, config["report_storage"])
        auto_eda(batch)
        train()
        logging.info("Update Result: SUCCESS")

    elif args.mode == "summary":
        logging.info("Generating summary report...")
        try:
            report_path = generate_summary_report(config)
            logging.info(f"Summary report saved to {report_path}")
        except Exception as e:
            logging.error(f"Failed to generate summary report: {e}")
