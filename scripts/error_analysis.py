"""
Error Analysis for Chess Win Probability Model

Finds games where the model was confident but wrong, enabling investigation
of failure modes.

Usage:
    python error_analysis.py \
        --model models/1500-1999/blitz_3.pkl \
        --data features/1500-1999/features_3_0.csv \
        --output results/errors_1500_1999_blitz3.csv \
        --threshold 0.80 \
        --top 20 \
        --output_image results/confidence_vs_error_rate.png
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_model(model_path: str) -> tuple:
    """Load a saved model and return (model, scaler, feature_names)."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['features']


def get_test_set_with_predictions(model, scaler, feature_names, data_path: str) -> pd.DataFrame:
    """
    Load data, recreate the same train/test split used in training,
    and return the test set with predictions.
    """
    df = pd.read_csv(data_path)
    
    # Drop rows with missing values (same as training)
    df_clean = df.dropna(subset=feature_names + ['white_wins']).copy()
    
    X = df_clean[feature_names].values
    y = df_clean['white_wins'].values
    
    # Recreate the same split used in training
    indices = np.arange(len(df_clean))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale and predict
    X_test_scaled = scaler.transform(X_test)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Build result dataframe
    test_df = df_clean.iloc[idx_test].copy()
    test_df['predicted_prob'] = y_prob
    test_df['predicted_winner'] = y_pred
    test_df['actual_winner'] = y_test
    test_df['correct'] = (y_pred == y_test)
    test_df['confidence'] = np.abs(y_prob - 0.5) + 0.5  # How far from 50%
    test_df['error_magnitude'] = np.abs(y_prob - y_test)  # How wrong
    
    return test_df


def find_confident_errors(df: pd.DataFrame, threshold: float = 0.80, top_n: int = 20) -> pd.DataFrame:
    """
    Find cases where model was confident (predicted prob > threshold or < 1-threshold)
    but got it wrong.
    """
    # Wrong predictions
    errors = df[~df['correct']].copy()
    
    # High confidence errors (predicted > threshold for white, or < 1-threshold for black)
    confident_errors = errors[
        (errors['predicted_prob'] >= threshold) | 
        (errors['predicted_prob'] <= (1 - threshold))
    ].copy()
    
    # Sort by how wrong the model was
    confident_errors = confident_errors.sort_values('error_magnitude', ascending=False)
    
    return confident_errors.head(top_n)


def format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    if pd.isna(seconds):
        return "N/A"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def plot_confidence_vs_error_rate(test_df: pd.DataFrame, output_path: str, n_bins: int = 10):
    """Plot error rate vs confidence level."""
    
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_centers = []
    error_rates = []
    sample_counts = []
    
    for i in range(len(bin_edges) - 1):
        mask = (test_df['confidence'] >= bin_edges[i]) & (test_df['confidence'] < bin_edges[i+1])
        if i == len(bin_edges) - 2:  # Include right edge for last bin
            mask = (test_df['confidence'] >= bin_edges[i]) & (test_df['confidence'] <= bin_edges[i+1])
        
        subset = test_df[mask]
        if len(subset) > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            error_rates.append((~subset['correct']).mean() * 100)
            sample_counts.append(len(subset))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot error rate
    color1 = '#C44E52'
    ax1.plot(bin_centers, error_rates, 'o-', color=color1, linewidth=2, markersize=8, label='Error Rate')
    ax1.set_xlabel('Model Confidence', fontsize=12)
    ax1.set_ylabel('Error Rate (%)', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim(0.48, 1.02)
    ax1.set_ylim(0, max(error_rates) * 1.2 if error_rates else 50)
    
    # Add sample count on secondary axis
    ax2 = ax1.twinx()
    color2 = '#4C72B0'
    ax2.bar(bin_centers, sample_counts, width=0.04, alpha=0.3, color=color2, label='Sample Count')
    ax2.set_ylabel('Sample Count', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and legend
    ax1.set_title('Error Rate vs Model Confidence\n(Lower error at higher confidence = well-behaved model)', fontsize=14)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nConfidence vs error rate plot saved to {output_path}")


def analyze_errors(model_path: str, data_path: str, threshold: float = 0.80, 
                   top_n: int = 20, output_path: str = None, output_image: str = None):
    """Run error analysis and print/save results."""
    
    print("Loading model and data...")
    model, scaler, feature_names = load_model(model_path)
    test_df = get_test_set_with_predictions(model, scaler, feature_names, data_path)
    
    # Overall stats
    total = len(test_df)
    correct = test_df['correct'].sum()
    accuracy = correct / total
    
    print(f"\n{'='*70}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Test samples: {total}")
    print(f"Correct predictions: {correct} ({accuracy:.1%})")
    print(f"Errors: {total - correct} ({1-accuracy:.1%})")
    
    # Find confident errors
    confident_errors = find_confident_errors(test_df, threshold=threshold, top_n=top_n)
    
    print(f"\n{'='*70}")
    print(f"CONFIDENT ERRORS (predicted >{threshold:.0%} or <{1-threshold:.0%}, but wrong)")
    print(f"{'='*70}")
    print(f"Found {len(confident_errors)} confident errors\n")
    
    if len(confident_errors) == 0:
        print("No confident errors found! Try lowering the threshold.")
        return
    
    # Display each error
    display_cols = ['predicted_prob', 'actual_winner', 'material_balance', 
                    'elo_diff', 'white_time_ratio', 'black_time_ratio', 'move_number']
    
    for i, (idx, row) in enumerate(confident_errors.iterrows()):
        print(f"\n--- Error #{i+1} ---")
        
        # Show game URL if available
        if 'game_url' in row and pd.notna(row.get('game_url')):
            print(f"Game: {row['game_url']}")
        
        print(f"Predicted: {row['predicted_prob']:.1%} white wins")
        print(f"Actual: {'White won' if row['actual_winner'] == 1 else 'Black won'}")
        print(f"")
        print(f"Features:")
        print(f"  Material balance: {row['material_balance']:.0f} centipawns")
        print(f"  Elo diff (W-B): {row['elo_diff']:.0f}")
        print(f"  Move number: {row['move_number']:.0f}")
        
        if 'white_time_remaining' in row and not pd.isna(row.get('white_time_remaining')):
            print(f"  White time: {format_time(row['white_time_remaining'])} ({row['white_time_ratio']:.1%} remaining)")
            print(f"  Black time: {format_time(row['black_time_remaining'])} ({row['black_time_ratio']:.1%} remaining)")
        else:
            print(f"  White time ratio: {row['white_time_ratio']:.1%}")
            print(f"  Black time ratio: {row['black_time_ratio']:.1%}")
        
        # Why might this be wrong?
        print(f"\nPossible explanation:")
        
        pred_white = row['predicted_prob'] > 0.5
        actual_white = row['actual_winner'] == 1
        
        if pred_white and not actual_white:
            # Model predicted white, black won
            if row['elo_diff'] > 100:
                print(f"  - Model favored white due to +{row['elo_diff']:.0f} Elo advantage")
            if row['material_balance'] > 200:
                print(f"  - Model favored white due to +{row['material_balance']:.0f} material")
            if row['white_time_ratio'] > row['black_time_ratio'] + 0.2:
                print(f"  - Model favored white due to time advantage")
            print(f"  - But black won anyway (tactical oversight? time scramble? blunder?)")
        else:
            # Model predicted black, white won
            if row['elo_diff'] < -100:
                print(f"  - Model favored black due to {row['elo_diff']:.0f} Elo advantage")
            if row['material_balance'] < -200:
                print(f"  - Model favored black due to {row['material_balance']:.0f} material")
            if row['black_time_ratio'] > row['white_time_ratio'] + 0.2:
                print(f"  - Model favored black due to time advantage")
            print(f"  - But white won anyway (tactical oversight? time scramble? blunder?)")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("ERROR PATTERNS")
    print(f"{'='*70}")
    
    errors_df = test_df[~test_df['correct']]
    
    # Material balance in errors
    print(f"\nMaterial balance in errors:")
    print(f"  Mean: {errors_df['material_balance'].mean():.1f}")
    print(f"  Std: {errors_df['material_balance'].std():.1f}")
    
    # Elo diff in errors  
    print(f"\nElo difference in errors:")
    print(f"  Mean: {errors_df['elo_diff'].mean():.1f}")
    print(f"  Std: {errors_df['elo_diff'].std():.1f}")
    
    # Time ratio diff in errors
    print(f"\nTime ratio diff in errors:")
    print(f"  Mean: {errors_df['time_ratio_diff'].mean():.3f}")
    print(f"  Std: {errors_df['time_ratio_diff'].std():.3f}")
    
    # Errors by confidence level
    print(f"\nErrors by confidence level:")
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(bins)-1):
        mask = (test_df['confidence'] >= bins[i]) & (test_df['confidence'] < bins[i+1])
        subset = test_df[mask]
        if len(subset) > 0:
            error_rate = (~subset['correct']).mean()
            print(f"  {bins[i]:.0%}-{bins[i+1]:.0%} confidence: {error_rate:.1%} error rate ({len(subset)} samples)")
    
    # Save to CSV if requested
    if output_path:
        # Save all errors, not just top N
        all_errors = test_df[~test_df['correct']].sort_values('error_magnitude', ascending=False)
        all_errors.to_csv(output_path, index=False)
        print(f"\nAll {len(all_errors)} errors saved to {output_path}")
    
    # Save plot if requested
    if output_image:
        plot_confidence_vs_error_rate(test_df, output_image)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze errors in chess win probability predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python error_analysis.py \\
        --model models/1500-1999/blitz_3.pkl \\
        --data features/1500-1999/features_3_0.csv \\
        --threshold 0.80 \\
        --top 10

    python error_analysis.py \\
        --model models/1000-1499/bullet.pkl \\
        --data features/1000-1499/features_1_0.csv \\
        --output results/errors_1000_1499_bullet.csv
        """
    )
    parser.add_argument('--model', required=True, help='Path to saved model .pkl file')
    parser.add_argument('--data', required=True, help='Path to features CSV file')
    parser.add_argument('--threshold', type=float, default=0.80,
                        help='Confidence threshold for "confident" predictions (default: 0.80)')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top errors to display (default: 20)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for CSV of all errors (optional)')
    parser.add_argument('--output_image', type=str, default=None,
                        help='Output path for confidence vs error rate plot (optional)')
    
    args = parser.parse_args()
    
    analyze_errors(
        model_path=args.model,
        data_path=args.data,
        threshold=args.threshold,
        top_n=args.top,
        output_path=args.output,
        output_image=args.output_image,
    )


if __name__ == '__main__':
    main()