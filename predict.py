"""
Chess Win Probability Predictor

Takes a FEN position, player ratings, and time information, and outputs
the probability that White wins.

Usage:
    # Using a saved model
    python predict.py --model model.pkl --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" \
        --white_elo 1500 --black_elo 1500 --white_time 150 --black_time 160 --time_control 180

    # Using hardcoded coefficients (no model file needed)
    python predict.py --fen "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4" \
        --white_elo 1600 --black_elo 1550 --white_time 120 --black_time 100 --time_control 180
"""

import argparse
import pickle
import sys

try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False

import numpy as np


# Piece values for material calculation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
} if CHESS_AVAILABLE else {}

# Simple piece values for manual parsing (when python-chess not available)
PIECE_VALUES_SIMPLE = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': 0,
}

# Default coefficients (update these with your trained model values)
DEFAULT_COEFFICIENTS = {
    'material_balance': 0.65,
    'elo_diff': 0.90,
    'move_number': -0.07,
    'white_time_ratio': 0.15,
    'black_time_ratio': -0.16,
    'time_ratio_diff': 0.40,
}

# Default scaling parameters (update these with your training data stats)
DEFAULT_MEANS = {
    'material_balance': 0.0,
    'elo_diff': 0.0,
    'move_number': 30.0,
    'white_time_ratio': 0.5,
    'black_time_ratio': 0.5,
    'time_ratio_diff': 0.0,
}

DEFAULT_STDS = {
    'material_balance': 300.0,
    'elo_diff': 150.0,
    'move_number': 15.0,
    'white_time_ratio': 0.25,
    'black_time_ratio': 0.25,
    'time_ratio_diff': 0.30,
}


def calculate_material_balance_chess(board: 'chess.Board') -> int:
    """Calculate material balance using python-chess (positive = white advantage)."""
    balance = 0
    for piece_type, value in PIECE_VALUES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        balance += (white_count - black_count) * value
    return balance


def calculate_material_balance_simple(fen: str) -> int:
    """Calculate material balance by parsing FEN string directly."""
    # Get the board part of the FEN (before the first space)
    board_part = fen.split()[0]
    
    balance = 0
    for char in board_part:
        if char in PIECE_VALUES_SIMPLE:
            balance += PIECE_VALUES_SIMPLE[char]
    
    return balance


def get_side_to_move(fen: str) -> int:
    """Return 1 if white to move, 0 if black to move."""
    parts = fen.split()
    if len(parts) >= 2:
        return 1 if parts[1] == 'w' else 0
    return 1  # Default to white


def get_move_number_from_fen(fen: str) -> int:
    """Extract full move number from FEN."""
    parts = fen.split()
    if len(parts) >= 6:
        try:
            return int(parts[5])
        except ValueError:
            pass
    return 1


def sigmoid(x: float) -> float:
    """Sigmoid function for logistic regression."""
    return 1.0 / (1.0 + np.exp(-x))


def extract_features(fen: str, white_elo: int, black_elo: int, 
                     white_time: float, black_time: float, 
                     time_control: float, move_number: int = None) -> dict:
    """Extract features from position and game state."""
    
    # Calculate material balance
    if CHESS_AVAILABLE:
        board = chess.Board(fen)
        material_balance = calculate_material_balance_chess(board)
    else:
        material_balance = calculate_material_balance_simple(fen)
    
    # Get move number from FEN if not provided
    if move_number is None:
        move_number = get_move_number_from_fen(fen)
    
    # Calculate time ratios
    white_time_ratio = white_time / time_control if time_control > 0 else 0.5
    black_time_ratio = black_time / time_control if time_control > 0 else 0.5
    time_ratio_diff = white_time_ratio - black_time_ratio
    
    return {
        'material_balance': material_balance,
        'elo_diff': white_elo - black_elo,
        'move_number': move_number,
        'white_time_ratio': white_time_ratio,
        'black_time_ratio': black_time_ratio,
        'time_ratio_diff': time_ratio_diff,
        'side_to_move': get_side_to_move(fen),
    }


def predict_with_model(features: dict, model_path: str) -> float:
    """Predict win probability using a saved sklearn model."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    model = data['model']
    scaler = data['scaler']
    feature_names = data['features']
    
    # Build feature vector in correct order
    X = np.array([[features[f] for f in feature_names]])
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]
    
    return prob


def predict_with_coefficients(features: dict, 
                               coefficients: dict = None,
                               means: dict = None,
                               stds: dict = None) -> float:
    """Predict win probability using hardcoded coefficients."""
    if coefficients is None:
        coefficients = DEFAULT_COEFFICIENTS
    if means is None:
        means = DEFAULT_MEANS
    if stds is None:
        stds = DEFAULT_STDS
    
    # Scale features and compute log-odds
    log_odds = 0.0
    for feature, coef in coefficients.items():
        if feature in features:
            scaled = (features[feature] - means[feature]) / stds[feature]
            log_odds += coef * scaled
    
    return sigmoid(log_odds)


def predict_win_probability(fen: str, white_elo: int, black_elo: int,
                            white_time: float, black_time: float,
                            time_control: float, move_number: int = None,
                            model_path: str = None) -> float:
    """
    Main prediction function.
    
    Args:
        fen: FEN string of the position
        white_elo: White's Elo rating
        black_elo: Black's Elo rating
        white_time: White's remaining time in seconds
        black_time: Black's remaining time in seconds
        time_control: Starting time in seconds (e.g., 180 for 3+0)
        move_number: Current move number (optional, extracted from FEN if not provided)
        model_path: Path to saved model file (optional, uses hardcoded coefficients if not provided)
    
    Returns:
        Probability that White wins (0.0 to 1.0)
    """
    features = extract_features(fen, white_elo, black_elo, white_time, black_time, 
                                time_control, move_number)
    
    if model_path:
        return predict_with_model(features, model_path)
    else:
        return predict_with_coefficients(features)


def main():
    parser = argparse.ArgumentParser(description='Predict chess win probability from position')
    parser.add_argument('--fen', required=True, help='FEN string of the position')
    parser.add_argument('--white_elo', type=int, required=True, help="White's Elo rating")
    parser.add_argument('--black_elo', type=int, required=True, help="Black's Elo rating")
    parser.add_argument('--white_time', type=float, required=True, help="White's remaining time (seconds)")
    parser.add_argument('--black_time', type=float, required=True, help="Black's remaining time (seconds)")
    parser.add_argument('--time_control', type=float, required=True, help='Starting time (seconds), e.g., 180 for 3+0')
    parser.add_argument('--move_number', type=int, default=None, help='Move number (optional, extracted from FEN if not provided)')
    parser.add_argument('--model', type=str, default=None, help='Path to saved model .pkl file (optional)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed feature breakdown')
    
    args = parser.parse_args()
    
    # Extract features
    features = extract_features(
        args.fen, args.white_elo, args.black_elo,
        args.white_time, args.black_time, args.time_control,
        args.move_number
    )
    
    # Predict
    if args.model:
        prob = predict_with_model(features, args.model)
    else:
        prob = predict_with_coefficients(features)
    
    # Output
    if args.verbose:
        print("\n=== Position Analysis ===")
        print(f"FEN: {args.fen}")
        print(f"Side to move: {'White' if features['side_to_move'] == 1 else 'Black'}")
        print(f"\n=== Features ===")
        print(f"Material balance: {features['material_balance']} centipawns")
        print(f"Elo difference (W-B): {features['elo_diff']}")
        print(f"Move number: {features['move_number']}")
        print(f"White time ratio: {features['white_time_ratio']:.2%}")
        print(f"Black time ratio: {features['black_time_ratio']:.2%}")
        print(f"Time ratio diff: {features['time_ratio_diff']:.2%}")
        print(f"\n=== Prediction ===")
    
    print(f"White win probability: {prob:.1%}")
    print(f"Black win probability: {1-prob:.1%}")


if __name__ == '__main__':
    main()