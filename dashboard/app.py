import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import yaml
import os
from pathlib import Path

def load_metrics_data():
    """Load all metrics from YAML files and combine into a DataFrame"""
    metrics = []
    reports_path = Path("storage/results/reports")
    
    # Process data quality reports
    for report_file in reports_path.glob("data_quality_report_batch_*.yaml"):
        with open(report_file) as f:
            data = yaml.safe_load(f)
            batch_num = int(report_file.stem.split('_')[-1])
            metrics.append({
                'batch': batch_num,
                'missing_values': sum(data['missing_values'].values()),
                'duplicates': data['duplicates'],
                'outliers': sum(data['outliers'].values()),
                'zero_values': sum(data['zero_values'].values()),
                'report_type': 'data_quality'
            })
    
    # Process model metrics
    for model_file in Path("storage/results").rglob("metrics.yaml"):
        with open(model_file) as f:
            model_data = yaml.safe_load(f)
            for model_name, metrics_data in model_data.items():
                metrics.append({
                    'batch': int(model_file.parent.name.split('_')[-1]),
                    'model': model_name,
                    'mae': metrics_data['mae'],
                    'mse': metrics_data['mse'],
                    'r2': metrics_data['r2'],
                    'report_type': 'model_performance'
                })
    
    return pd.DataFrame(metrics)

def create_dashboard():
    app = dash.Dash(__name__)
    
    try:
        df = load_metrics_data()
        if df.empty:
            raise ValueError("No metrics data found")
            
        app.layout = html.Div([
            html.H1("Steam Games MLOps Dashboard"),
            
            # Data Quality Trends
            html.H2("Data Quality Metrics Over Batches"),
            dcc.Graph(
                figure=px.line(
                    df[df['report_type'] == 'data_quality'],
                    x='batch',
                    y=['missing_values', 'duplicates', 'outliers', 'zero_values'],
                    title='Data Quality Trends'
                )
            ),
            
            # Model Performance
            html.H2("Model Performance"),
            dcc.Graph(
                figure=px.line(
                    df[df['report_type'] == 'model_performance'],
                    x='batch',
                    y='mae',
                    color='model',
                    title='MAE by Model Across Batches'
                )
            ),
            
            # Correlation Matrix (if exists)
            html.H2("Feature Correlation"),
            html.Div(id='correlation-matrix'),
            
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