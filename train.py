from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dataset_utils import load_from_config
from datetime import datetime
from tqdm import tqdm
import os
import ast
import pickle
from scipy.stats import zscore

RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% for test set
# 25% of training set for validation (15% of total data)
VALIDATION_SIZE = 0.25
N_JOBS = -1  # Use all available cores

# LOAD CONFIG
cfg = load_from_config()
target = 'price'
categorical = ['categories', 'genres', 'tags']
ignore = ['app_id', 'name', 'release_date',
          'short_description', 'tag_frequencies']


# 1. DATA PREPROCESSING

unique_sets = {cat: set() for cat in categorical}

# Load previous batches
for batch_path in tqdm(os.listdir(cfg['batch_storage']), desc='Scanning unique categories...'):
    batch = pd.read_csv(os.path.join(cfg['batch_storage'], batch_path))
    for cat in categorical:
        arr = batch[cat].apply(ast.literal_eval)
        for item in arr:
            unique_sets[cat].update(item)

# Initialize encoders
encoders = {cat: MultiLabelBinarizer(classes=sorted(
    list(unique_sets[cat]))) for cat in categorical}
for cat in categorical:
    encoders[cat].fit([])  # Dummy fit to set classes_

# Encode previous batches
X_train = []
for batch_path in tqdm(os.listdir(cfg['batch_storage']), desc='Encoding batches...'):
    batch = pd.read_csv(os.path.join(cfg['batch_storage'], batch_path))
    encoded_dfs = []
    for cat in categorical:
        arr = batch[cat].apply(ast.literal_eval)
        enc = encoders[cat].transform(arr)
        encoded_dfs.append(pd.DataFrame(enc, columns=encoders[cat].classes_))
    batch = pd.concat([batch.drop(columns=categorical)] + encoded_dfs, axis=1)
    X_train.append(batch)
X_train = pd.concat(X_train, axis=0)


# 2. DATA PREPARATION

# Prepare features and target
features = [i for i in X_train.columns if i not in ignore and i != target]
X = X_train[features]
y = X_train[target]

# Оставляем только числовые столбцы для обработки
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns


correlation_matrix = X_train[numerical_columns].corr()
# Remove features with low correlation to the target
threshold = 0.1  # Define a threshold for correlation
low_correlation_features = correlation_matrix[target][abs(
    correlation_matrix[target]) < threshold].index
X = X.drop(columns=low_correlation_features)

print(
    f"Removed features with correlation below {threshold}: {list(low_correlation_features)}")
# Split into train and test
X_train_split, X_test, y_train_split, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# --- Интерпретация ошибки (MAE) ---
# Важно: MAE показывает среднюю абсолютную ошибку между предсказанием и реальной ценой.
# Например, MAE = 4 означает, что модель в среднем ошибается на 4 единицы валюты.

# Насколько это "нормально", зависит от диапазона цен:
# - Если большинство игр стоит от 0 до 20 — это много (~20%)
# - Если до 60 — умеренно (~6%)
# - Если распределение скошенное и много дорогих/дешевых — стоит смотреть MAPE или логарифмы

# Можно сравнивать с бейзлайном — модель, которая предсказывает среднюю цену:
baseline_pred = [y_train_split.mean()] * len(y_test)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
print(f"Baseline MAE (mean-only predictor): {baseline_mae:.4f}")

# 3. MODEL INITIALIZATION

print("\nSetting up models and hyperparameters...")

# Define models and their parameter grids
models = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True, False],
            'positive': [True, False]  # Force positive coefficients
        }
    },
    'KNN': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1: Manhattan, 2: Euclidean
        }
    },
    'DecisionTree': {
        'model': DecisionTreeRegressor(random_state=RANDOM_STATE),
        'params': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=RANDOM_STATE),

        'params': {
            'n_estimators': [100],
            'max_depth': [None, 10],
            # 'min_samples_split': [2, 5],
            # 'min_samples_leaf': [1, 2]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
        }
    },
    #     'XGBoost': {
    #         'model': XGBRegressor(random_state=RANDOM_STATE),
    #         'params': {
    #             'n_estimators': [100, 200],
    #             'learning_rate': [0.01, 0.1, 0.2],
    #             'max_depth': [3, 5, 7],
    #             'subsample': [0.8, 1.0],
    #             'colsample_bytree': [0.8, 1.0]
    #         }
    #     }
    # }
}

# 4. TRAINING

best_models = {}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(timestamp)
results_dir = os.path.join(cfg['output_dir'], f'model_results_{timestamp}')
os.makedirs(results_dir, exist_ok=True)

for name, config in models.items():
    print(f"\nTraining and tuning {name}...")

    # Grid search with 5-fold cross-validation
    grid = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,
        n_jobs=N_JOBS,
        scoring='neg_mean_squared_error',
        verbose=1
    )

    grid.fit(X_train_split, y_train_split)

    # Get best model
    best_model = grid.best_estimator_
    best_models[name] = best_model
    if name == 'LinearRegression':
        # Display feature importance for Linear Regression
        feature_importance = pd.DataFrame({
            'Feature': X_train_split.columns,
            'Coefficient': best_model.coef_
        }).sort_values(by='Coefficient', ascending=False)
        feature_importance.to_csv(os.path.join(
            results_dir, f'{name}_feature_importance.csv'), index=False)
        print("\nFeature Importance for Linear Regression:")
        print(feature_importance)
    # Evaluate
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"Best Parameters: {grid.best_params_}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

    # Save model
    model_path = os.path.join(results_dir, f'{name}_best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {model_path}")

    # Save evaluation metrics
    metrics = {
        'best_params': grid.best_params_,
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'feature_importances': getattr(best_model, 'feature_importances_', None),
        'coef': getattr(best_model, 'coef_', None)
    }

    metrics_path = os.path.join(results_dir, f'{name}_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)

print("\nAll models trained and saved successfully!")
