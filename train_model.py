"""
Chess Win Probability Model
Train and evaluate logistic regression model on extracted features.

Usage:
    python train_model.py features.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import argparse
import pickle


def load_and_prepare_data(csv_path: str):
    """Load CSV and prepare feature matrix and target."""
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"\nTarget distribution:")
    print(df['white_wins'].value_counts(normalize=True))
    
    # Define feature sets for experimentation
    
    # Basic features (no time)
    basic_features = [
        'material_balance',
        'elo_diff',
        'side_to_move',
        'move_number',
    ]
    
    # Material detail features
    material_features = [
        'white_pawns', 'white_knights', 'white_bishops', 'white_rooks', 'white_queens',
        'black_pawns', 'black_knights', 'black_bishops', 'black_rooks', 'black_queens',
    ]
    
    # Castling features
    castling_features = [
        'white_can_castle_kingside', 'white_can_castle_queenside',
        'black_can_castle_kingside', 'black_can_castle_queenside',
    ]
    
    # Time features (may have missing values)
    time_features = [
        'white_time_ratio', 'black_time_ratio', 'time_ratio_diff',
    ]
    
    return df, {
        'basic': basic_features,
        'material': material_features,
        'castling': castling_features,
        'time': time_features,
    }


def train_and_evaluate(df: pd.DataFrame, feature_cols: list[str], model_name: str = "model"):
    """Train logistic regression and evaluate performance."""
    
    # Drop rows with missing values in the selected features
    df_clean = df.dropna(subset=feature_cols + ['white_wins'])
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Features: {feature_cols}")
    print(f"Samples after dropping NaN: {len(df_clean)}")
    
    X = df_clean[feature_cols].values
    y = df_clean['white_wins'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test AUC-ROC: {auc:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Feature importance (coefficients)
    print(f"\nFeature Coefficients:")
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    print(coef_df.to_string(index=False))
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return model, scaler, accuracy, auc


def main():
    parser = argparse.ArgumentParser(description='Train chess win probability model')
    parser.add_argument('input_csv', help='Input CSV file with extracted features')
    parser.add_argument('--save_model', help='Path to save the trained model (pickle)')
    parser.add_argument('--output_report', help='Path to save the results report (text file)')
    
    args = parser.parse_args()
    
    # If saving to a report, redirect print output
    import sys
    from io import StringIO
    
    if args.output_report:
        # Capture all output
        output_buffer = StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_buffer
    
    try:
        df, feature_sets = load_and_prepare_data(args.input_csv)
        
        # Experiment 1: Basic features only
        basic_cols = feature_sets['basic']
        train_and_evaluate(df, basic_cols, "Basic (material + elo + move)")
        
        # Experiment 2: Add detailed material
        material_cols = feature_sets['basic'] + feature_sets['material']
        # Remove material_balance since we have detailed counts
        material_cols = [c for c in material_cols if c != 'material_balance']
        train_and_evaluate(df, material_cols, "Detailed Material")
        
        # Experiment 3: Add castling
        castling_cols = material_cols + feature_sets['castling']
        train_and_evaluate(df, castling_cols, "Material + Castling")
        
        # Experiment 4: Add time (full model)
        full_cols = castling_cols + feature_sets['time']
        model, scaler, acc, auc = train_and_evaluate(df, full_cols, "Full Model (with time)")
        
        # Experiment 5: Just elo and time (no position info)
        elo_time_cols = ['elo_diff', 'white_time_ratio', 'black_time_ratio', 'time_ratio_diff', 'move_number']
        train_and_evaluate(df, elo_time_cols, "ELO + Time Only (no material)")
        
        # Experiment 6: Material balance + elo + move number + time (simple but complete)
        simple_full_cols = ['material_balance', 'elo_diff', 'move_number', 'white_time_ratio', 'black_time_ratio', 'time_ratio_diff']
        train_and_evaluate(df, simple_full_cols, "Simple Full (material balance + elo + move + time)")
        
        # Save the full model if requested
        if args.save_model:
            with open(args.save_model, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': scaler,
                    'features': full_cols,
                }, f)
            print(f"\nModel saved to {args.save_model}")
    
    finally:
        # Restore stdout and save report if requested
        if args.output_report:
            sys.stdout = original_stdout
            report_content = output_buffer.getvalue()
            
            # Print to console as well
            print(report_content)
            
            # Save to file
            with open(args.output_report, 'w') as f:
                f.write(report_content)
            print(f"\nReport saved to {args.output_report}")


if __name__ == '__main__':
    main()