# Chess Win Probability Calculator

Uses logistic regression to predict bullet, blitz, and rapid chess game outcomes from board position, player ratings, and time remaining. Trained on games played on [lichess.org](https://lichess.org).

## Key Findings

- **Elo difference is the strongest predictor** across all rating bands
- **Time pressure affects all players equally** — coefficients are similar from 1000-2500 Elo
- **Material advantage matters slightly less at higher ratings** (0.71 → 0.59 coefficient)

### Model Performance

| Rating Band | N Samples | AUC-ROC |
|-------------|-----------|---------|
| 1000-1500   | ~20,000   | 0.72    |
| 1500-2000   | ~20,000   | 0.72    |
| 2000-2500   | ~20,000   | 0.73    |

### Feature Coefficients (1500-2000 Elo)

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `elo_diff` | 0.91 | Rating advantage is most predictive |
| `material_balance` | 0.60 | Material matters, but less than rating |
| `time_ratio_diff` | 0.40 | Having more time helps |
| `white_time_ratio` | 0.13 | Absolute time remaining |
| `black_time_ratio` | -0.16 | Opponent's time (negative = good for white) |
| `move_number` | -0.11 | Later moves slightly favor the defender |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/jake-p-bennett/chess-win-probability.git
cd chess-win-probability
pip install -r requirements.txt

# Predict win probability for a position
python scripts/predict.py \
    --fen "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4" \
    --white_elo 1600 --black_elo 1550 \
    --white_time 150 --black_time 140 \
    --time_control 180 \
    --verbose
```

## Project Structure

```
chess-win-probability/
├── README.md
├── requirements.txt
├── scripts/
│   ├── download_games.py    # Fetch games from Lichess API
│   ├── extract_features.py  # Extract features from PGN files
│   ├── train_model.py       # Train and evaluate models
│   ├── compare_models.py    # Compare models across datasets
│   └── predict.py           # Make predictions on new positions
├── models/                   # Saved model files (.pkl)
├── results/                  # Output tables and reports
```

## Full Pipeline

### 1. Download Games

```bash
# Download games from players in a rating range
python scripts/download_games.py \
    --rating_range 1500 2000 \
    --num_games 50000 \
    --output data/games_1500_2000.pgn

# Or from specific users
python scripts/download_games.py \
    --users DrNykterstein Hikaru \
    --max_per_user 1000 \
    --output data/games.pgn
```

### 2. Extract Features

```bash
# Extract features from all games
python scripts/extract_features.py \
    data/games_1500_2000.pgn \
    data/features_1500_2000.csv \
    --num_games 20000

# Filter by specific time control
python scripts/extract_features.py \
    data/games.pgn \
    data/features_3_0.csv \
    --num_games 20000 \
    --time_control "180+0"
```

### 3. Train Models

```bash
# Train and evaluate all model variants
python scripts/train_model.py data/features_1500_2000.csv \
    --output_report results/report_1500_2000.txt \
    --save_model models/model_1500_2000.pkl
```

### 4. Compare Across Datasets

```bash
# Generate comparison table for your blog
python scripts/compare_models.py \
    data/features_1000_1500.csv \
    data/features_1500_2000.csv \
    data/features_2000_2500.csv \
    --labels "1000-1500" "1500-2000" "2000-2500" \
    --output_csv results/comparison_by_rating.csv
```

## Data

Games are downloaded from the [Lichess API](https://lichess.org/api). Raw PGN files are not included in this repo due to size — use `download_games.py` to fetch them.

**Data processing notes:**
- Only decisive games (no draws) are used for binary classification
- One random position is sampled per game, excluding first/last 5 moves
- Games with missing Elo or clock data are skipped


## Methodology

### Feature Engineering

| Feature | Description |
|---------|-------------|
| `material_balance` | Centipawn value: Q=900, R=500, B=330, N=320, P=100 (white - black) |
| `elo_diff` | White's Elo minus Black's Elo |
| `move_number` | Current move number in the game |
| `white_time_ratio` | White's remaining time / starting time |
| `black_time_ratio` | Black's remaining time / starting time |
| `time_ratio_diff` | white_time_ratio - black_time_ratio |

### Why Sample One Position Per Game?

Positions within a game are correlated — if you include all positions, your effective sample size is much smaller than it appears, and late-game positions trivially predict outcomes. Sampling one random mid-game position per game gives independent samples with genuine uncertainty.

### Why Exclude First/Last 5 Moves?

- **First 5 moves:** Positions carry very little signal about the outcome of the game.
- **Last 5 moves:** Positions are much more likely to be from games that are nearly decided

## Possible Extensions

- Add Stockfish evaluation as a feature
- Calibration analysis (do 70% predictions win 70% of the time?)
- Neural network with board representation

