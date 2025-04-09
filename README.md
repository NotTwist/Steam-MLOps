# Steam Data analysis

## Step 1: Download the Dataset

Begin by downloading the `games.json` file from the [Steam Games Dataset on Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset). This dataset provides comprehensive information about Steam games, which will serve as the foundation for analysis and modeling.

## Step 2: Prepare the Dataset

1. Locate the downloaded `games.json.zip` file.
2. Extract the contents of the `.zip` archive to retrieve the `games.json` file.
3. Place the `games.json` file in the root directory of this repository.

These steps ensure the dataset is properly organized and ready for further processing.

## Step 3: Generate the Dataset CSV File

Run the following command to process the dataset and generate the required files:

```bash
python dataset_utils.py
```

This script will create:
- `games.csv`: A structured CSV file containing the dataset.
- `raw_batches/`: A folder containing chronological batches of the `games.csv` data.

Ensure this step is completed before proceeding with the analysis.