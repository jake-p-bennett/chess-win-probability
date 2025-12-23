"""
Calibration Analysis for Chess Win Probability Model

Analyzes how well-calibrated a model's probability predictions are.
When the model predicts 70% win probability, does white actually win ~70% of the time?

Usage:
    python calibration_analysis.py \
        --model models/1500-1999/blitz_3.pkl \
        --data features/1500-1999/features_3_0.csv \
        --output images/calibration_1500_1999_blitz3.png

    # Compare calibration across rating bands
    python calibration_analysis.py \
        --model models/1000-1499/blitz_3.pkl models/1500-1999/blitz_3.pkl models/2000-2499/blitz_3.pkl \
        --data features/1000-1499/features_3_0.csv features/1500-1999/features_3_0.csv features/2000-2499/features_3_0.csv \
        --labels "1000-1499" "1500-1999" "2000-2499" \
        --output images/calibration_comparison.png
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve


def load_model(model_path: str) -> tuple:
    """Load a saved model and return (model, scaler, feature_names)."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['features']


def get_test_predictions(model, scaler, feature_names, data_path: str) -> tuple:
    """
    Load data, recreate the same train/test split used in training,
    and return predictions on the test set.
    
    Returns: (y_true, y_prob) for the test set
    """
    df = pd.read_csv(data_path)
    
    # Drop rows with missing values (same as training)
    df_clean = df.dropna(subset=feature_names + ['white_wins'])
    
    X = df_clean[feature_names].values
    y = df_clean['white_wins'].values
    
    # Recreate the same split used in training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale and predict
    X_test_scaled = scaler.transform(X_test)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    return y_test, y_prob


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Brier score (mean squared error of probability predictions).
    Lower is better. Range: 0 (perfect) to 1 (worst).
    
    For reference:
    - Always predicting 0.5: Brier = 0.25
    - Perfect predictions: Brier = 0.0
    """
    return np.mean((y_prob - y_true) ** 2)


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                           n_bins: int = 10, ax=None, label: str = None,
                           color: str = None) -> tuple:
    """
    Plot calibration curve and return (fraction_of_positives, mean_predicted_value).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    
    # Plot
    plot_kwargs = {'marker': 'o', 'linewidth': 2, 'markersize': 8}
    if label:
        plot_kwargs['label'] = label
    if color:
        plot_kwargs['color'] = color
    
    ax.plot(mean_predicted_value, fraction_of_positives, **plot_kwargs)
    
    return fraction_of_positives, mean_predicted_value


def plot_calibration_histogram(y_prob: np.ndarray, n_bins: int = 10, 
                                ax=None, color: str = None, alpha: float = 0.7):
    """Plot histogram of predicted probabilities."""
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.hist(y_prob, bins=n_bins, range=(0, 1), alpha=alpha, 
            color=color, edgecolor='white')
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Predictions', fontsize=12)


def analyze_calibration(model_paths: list, data_paths: list, 
                        labels: list = None, output_path: str = None,
                        n_bins: int = 10):
    """
    Run calibration analysis for one or more models.
    """
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(model_paths))]
    
    # Colors for multiple models
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']
    
    # Create figure
    if len(model_paths) == 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_cal, ax_hist = axes
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_cal, ax_hist = axes
    
    print("=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)
    
    all_results = []
    
    for i, (model_path, data_path, label) in enumerate(zip(model_paths, data_paths, labels)):
        print(f"\n{label}")
        print("-" * 40)
        
        # Load model and get predictions
        model, scaler, feature_names = load_model(model_path)
        y_true, y_prob = get_test_predictions(model, scaler, feature_names, data_path)
        
        # Calculate metrics
        brier = calculate_brier_score(y_true, y_prob)
        
        # Baseline Brier (always predict base rate)
        base_rate = y_true.mean()
        brier_baseline = calculate_brier_score(y_true, np.full_like(y_prob, base_rate))
        
        # Brier skill score (improvement over baseline)
        brier_skill = 1 - (brier / brier_baseline)
        
        print(f"Test samples: {len(y_true)}")
        print(f"Actual win rate: {y_true.mean():.1%}")
        print(f"Mean predicted probability: {y_prob.mean():.1%}")
        print(f"Brier score: {brier:.4f}")
        print(f"Brier baseline (always predict {base_rate:.1%}): {brier_baseline:.4f}")
        print(f"Brier skill score: {brier_skill:.4f}")
        
        # Plot calibration curve
        color = colors[i % len(colors)]
        fraction_pos, mean_pred = plot_calibration_curve(
            y_true, y_prob, n_bins=n_bins, ax=ax_cal, 
            label=f"{label} (Brier={brier:.3f})", color=color
        )
        
        # Plot histogram (only for single model, or use transparency for multiple)
        if len(model_paths) == 1:
            plot_calibration_histogram(y_prob, n_bins=n_bins, ax=ax_hist, color=color)
        else:
            ax_hist.hist(y_prob, bins=n_bins, range=(0, 1), alpha=0.5, 
                        color=color, edgecolor='white', label=label)
        
        # Store results
        all_results.append({
            'label': label,
            'brier_score': brier,
            'brier_baseline': brier_baseline,
            'brier_skill': brier_skill,
            'mean_predicted': y_prob.mean(),
            'actual_win_rate': y_true.mean(),
            'n_samples': len(y_true),
        })
        
        # Print bin-by-bin breakdown
        print(f"\nCalibration by bin:")
        print(f"{'Predicted':>12} {'Actual':>12} {'Count':>8}")
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        for j in range(n_bins):
            mask = (y_prob >= bin_edges[j]) & (y_prob < bin_edges[j+1])
            if j == n_bins - 1:  # Include right edge for last bin
                mask = (y_prob >= bin_edges[j]) & (y_prob <= bin_edges[j+1])
            
            if mask.sum() > 0:
                bin_pred = y_prob[mask].mean()
                bin_actual = y_true[mask].mean()
                bin_count = mask.sum()
                print(f"{bin_pred:>12.1%} {bin_actual:>12.1%} {bin_count:>8}")
    
    # Finalize calibration plot
    ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    ax_cal.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax_cal.set_ylabel('Fraction of Positives (Actual Win Rate)', fontsize=12)
    ax_cal.set_title('Calibration Curve', fontsize=14)
    ax_cal.legend(loc='lower right')
    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.grid(True, alpha=0.3)
    ax_cal.set_aspect('equal')
    
    # Finalize histogram
    if len(model_paths) > 1:
        ax_hist.legend()
    ax_hist.set_xlabel('Predicted Probability', fontsize=12)
    ax_hist.set_ylabel('Count', fontsize=12)
    ax_hist.set_title('Distribution of Predictions', fontsize=14)
    ax_hist.set_xlim(0, 1)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze calibration of chess win probability models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model
    python calibration_analysis.py \\
        --model models/1500-1999/blitz_3.pkl \\
        --data features/1500-1999/features_3_0.csv \\
        --output images/calibration_1500_1999.png

    # Compare multiple models
    python calibration_analysis.py \\
        --model models/1000-1499/blitz_3.pkl models/1500-1999/blitz_3.pkl \\
        --data features/1000-1499/features_3_0.csv features/1500-1999/features_3_0.csv \\
        --labels "1000-1499" "1500-1999" \\
        --output images/calibration_comparison.png
        """
    )
    parser.add_argument('--model', nargs='+', required=True, 
                        help='Path(s) to saved model .pkl file(s)')
    parser.add_argument('--data', nargs='+', required=True,
                        help='Path(s) to features CSV file(s)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each model (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for calibration plot (optional, shows plot if not provided)')
    parser.add_argument('--bins', type=int, default=10,
                        help='Number of bins for calibration curve (default: 10)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.model) != len(args.data):
        print("Error: Number of models must match number of data files")
        return
    
    if args.labels and len(args.labels) != len(args.model):
        print("Error: Number of labels must match number of models")
        return
    
    # Run analysis
    analyze_calibration(
        model_paths=args.model,
        data_paths=args.data,
        labels=args.labels,
        output_path=args.output,
        n_bins=args.bins
    )


if __name__ == '__main__':
    main()