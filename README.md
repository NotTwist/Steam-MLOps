# Steam Data Analysis

## Step 1: Download the Dataset

Begin by downloading the `games.json` file from the [Steam Games Dataset on Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset). This dataset provides comprehensive information about Steam games, which will serve as the foundation for analysis and modeling.

## Step 2: Prepare the Dataset

1. Locate the downloaded `games.json.zip` file.
2. Extract the contents of the `.zip` archive to retrieve the `games.json` file.
3. Place the `games.json` file in the storage/ directory of this repository.

These steps ensure the dataset is properly organized and ready for further processing.

## Step 3: Install Dependencies

Install the required Python dependencies using the following command:

```bash
pip install -r requirements.txt
```
This will ensure all necessary libraries are installed in your environment. Python version is 3.9
## Commands

### 1. Inference
Apply the trained model to external data and generate predictions.

Command:
```
python run.py -mode "inference" -file "./path_to_input.csv"
```
Arguments:

-mode: Set to "inference" to run inference.
-file: Path to the input CSV file.
Output:

A new CSV file with predictions added as a predict column, saved in the folder specified by infer_folder in config.yaml.
### 2. Update
Fetch the next batch of data, preprocess it, and retrain the model.

Command:
```
python run.py -mode "update"
```
Arguments:

-mode: Set to "update" to process the next batch and retrain the model.
Output:

Updated model and metrics saved in the results folder.
Data quality and EDA reports saved in the eda_storage folder.
### 3. Summary
Generate a monitoring report summarizing data quality, model metrics, and hyperparameters.

Command:
```
python run.py -mode "summary"
```
Arguments:

-mode: Set to "summary" to generate the monitoring report.
Output:

A monitoring_report.txt file saved in the report_storage folder.

### Dashboard
Access dashboard by
```
python dashboard/app.py
```