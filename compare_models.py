"""
Generate a comparison table of model coefficients across different datasets.

Usage:
    python compare_models.py features_1000_1500.csv features_1500_2000.csv features_2000_2500.csv

This will train the simple model on each dataset and output a markdown table
comparing coefficients.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import argparse
import sys


# The simple model features
FEATURES = [
    'material_balance', 
    'elo_diff', 
    'move_number', 
    'white_time_ratio', 
    'black_time_ratio', 
    'time_ratio_diff'
]


def train_simple_model(csv_path: str) -> dict:
    """Train the simple model and return coefficients and metrics."""
    df = pd.read_csv(csv_path)
    
    # Drop rows with missing values
    df_clean = df.dropna(subset=FEATURES + ['white_wins'])
    
    X = df_clean[FEATURES].values
    y = df_clean['white_wins'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get metrics
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    
    # Build results
    coefficients = dict(zip(FEATURES, model.coef_[0]))
    
    return {
        'n_samples': len(df_clean),
        'auc': auc,
        'coefficients': coefficients
    }


def extract_label_from_filename(filename: str) -> str:
    """Try to extract a meaningful label from the filename."""
    # Remove path and extension
    name = filename.replace('\\', '/').split('/')[-1]
    name = name.replace('.csv', '').replace('features_', '').replace('features', '')
    
    # Clean up common patterns
    name = name.replace('_', ' ').strip()
    
    if not name:
        return filename
    
    return name


def generate_markdown_table(results: dict[str, dict]) -> str:
    """Generate a markdown table from results."""
    
    # Header
    lines = []
    header = "| Dataset | N | AUC | " + " | ".join(FEATURES) + " |"
    separator = "|" + "|".join(["---"] * (3 + len(FEATURES))) + "|"
    
    lines.append(header)
    lines.append(separator)
    
    # Rows
    for label, data in results.items():
        coefs = data['coefficients']
        coef_strs = [f"{coefs[f]:.3f}" for f in FEATURES]
        
        row = f"| {label} | {data['n_samples']:,} | {data['auc']:.3f} | " + " | ".join(coef_strs) + " |"
        lines.append(row)
    
    return "\n".join(lines)


def generate_csv(results: dict[str, dict], output_path: str):
    """Generate a CSV file from results."""
    rows = []
    
    for label, data in results.items():
        row = {
            'dataset': label,
            'n_samples': data['n_samples'],
            'auc': data['auc'],
            **data['coefficients']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nCSV saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare model coefficients across datasets')
    parser.add_argument('csv_files', nargs='+', help='CSV files with extracted features')
    parser.add_argument('--labels', nargs='+', help='Labels for each dataset (optional)')
    parser.add_argument('--output_csv', help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Validate labels if provided
    if args.labels and len(args.labels) != len(args.csv_files):
        print("Error: Number of labels must match number of CSV files")
        sys.exit(1)
    
    # Train models and collect results
    results = {}
    
    for i, csv_path in enumerate(args.csv_files):
        if args.labels:
            label = args.labels[i]
        else:
            label = extract_label_from_filename(csv_path)
        
        print(f"Training on {csv_path} ({label})...")
        
        try:
            data = train_simple_model(csv_path)
            results[label] = data
            print(f"  N={data['n_samples']:,}, AUC={data['auc']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Generate and print markdown table
    print("\n" + "=" * 60)
    print("MARKDOWN TABLE (copy this into your Quarto blog):")
    print("=" * 60 + "\n")
    
    table = generate_markdown_table(results)
    print(table)
    
    # Save CSV if requested
    if args.output_csv:
        generate_csv(results, args.output_csv)


if __name__ == '__main__':
    main()