import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import yaml
import os
from pathlib import Path
from datetime import datetime


def load_metrics_data():
    """Load all metrics from YAML files and combine into a DataFrame"""
    data_metrics = []
    reports_path = Path("storage/results/reports")

    # Process data quality reports
    for report_file in reports_path.glob("data_quality_report_batch_*.yaml"):
        with open(report_file) as f:
            data = yaml.safe_load(f)
            batch_num = int(report_file.stem.split('_')[-1])
            data_metrics.append({
                'batch': batch_num,
                'missing_values': sum(data['missing_values'].values()),
                'duplicates': data['duplicates'],
                'outliers': sum(data['outliers'].values()),
                'zero_values': sum(data['zero_values'].values()),
                'report_type': 'data_quality'
            })
    model_metrics = []

    # Process model metrics
    for model_file in Path("storage/results").rglob("metrics.yaml"):
        with open(model_file) as f:
            model_data = yaml.safe_load(f)
            for model_name, metrics_data in model_data.items():
                model_metrics.append({
                    'batch': datetime.strptime('_'.join(model_file.parent.name.split('_')[-2:]), "%Y%m%d_%H%M%S"),                    'model': model_name,
                    'mae': metrics_data['mae'],
                    'mse': metrics_data['mse'],
                    'r2': metrics_data['r2'],
                    'report_type': 'model_performance'
                })

    return pd.DataFrame(data_metrics).sort_values(by='batch'), pd.DataFrame(model_metrics).sort_values(by='batch')


def load_correlation_matrix():
    """Loads correlation matrix from EDA report into DataFrame"""
    correlation_path = Path("storage/results/eda/correlation_matrix.csv")
    corr_matrix = pd.read_csv(correlation_path, index_col=0)
    return corr_matrix
def create_dashboard():
    app = dash.Dash(__name__)

    try:
        data_df, model_df = load_metrics_data()
        corr_matrix = load_correlation_matrix()
        
        if data_df.empty or model_df.empty:
            raise ValueError("No metrics/model data found")
        app.layout = html.Div([
            html.H1("Steam Games MLOps Dashboard"),

            # Data Quality Trends
            html.H2("Data Quality Metrics Over Batches"),
            dcc.Graph(
                figure=px.line(
                    data_df[data_df['report_type'] == 'data_quality'],
                    x='batch',
                    y=['missing_values', 'duplicates',
                        'outliers', 'zero_values'],
                    title='Data Quality Trends'
                )
            ),

            # Model Performance
            html.H2("Model Performance"),
            dcc.Graph(
                figure=px.line(
                    model_df,
                    x='batch',
                    y='mae',
                    color='model',
                    title='MAE by Model Across Batches'
                )
            ),

            html.H2("Feature Correlation"),
            # html.Div(id='correlation-matrix'),

            dcc.Graph(figure=px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                title='Feature Correlation Matrix'
            )),
            dcc.Interval(id='refresh', interval=60*1000)
        ])

    except Exception as e:
        app.layout = html.Div([
            html.H1("Dashboard Error"),
            html.P(str(e)),
            html.P("Please ensure you have run the pipeline to generate data first.")
        ])

    return app


if __name__ == '__main__':
    app = create_dashboard()
    app.run(debug=True)
