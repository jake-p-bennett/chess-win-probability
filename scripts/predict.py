"""
Chess Win Probability Predictor

Takes a FEN position, player ratings, and time information, and outputs
the probability that White wins.

Usage:
    python predict.py --model models/model_1500_2000.pkl \
        --fen "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4" \
        --white_elo 1600 --black_elo 1550 \
        --white_time "2:30" --black_time "2:15" \
        --time_control "3+0"
"""

import argparse
import pickle
import sys
import re

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


def parse_time(time_str: str) -> float:
    """
    Parse a time string into seconds.
    
    Accepted formats:
        "2:30" or "2:30.5"  -> 2 minutes 30 seconds (150 or 150.5 seconds)
        "1:02:30"           -> 1 hour 2 minutes 30 seconds
        "90"                -> 90 seconds
        "90.5"              -> 90.5 seconds
    
    Returns:
        Time in seconds as a float
    
    Raises:
        ValueError if format is invalid
    """
    time_str = time_str.strip()
    
    # Try pure number first (seconds)
    try:
        return float(time_str)
    except ValueError:
        pass
    
    # Try M:SS or MM:SS or H:MM:SS format
    parts = time_str.split(':')
    
    if len(parts) == 2:
        # M:SS or MM:SS
        try:
            minutes = int(parts[0])
            seconds = float(parts[1])
            if seconds < 0 or seconds >= 60:
                raise ValueError(f"Seconds must be between 0 and 59, got {seconds}")
            if minutes < 0:
                raise ValueError(f"Minutes cannot be negative, got {minutes}")
            return minutes * 60 + seconds
        except ValueError as e:
            if "Seconds must be" in str(e) or "Minutes cannot" in str(e):
                raise
            raise ValueError(
                f"Invalid time format: '{time_str}'. "
                f"Expected formats: '2:30' (M:SS), '90' (seconds), or '1:02:30' (H:MM:SS)"
            )
    
    elif len(parts) == 3:
        # H:MM:SS
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            if seconds < 0 or seconds >= 60:
                raise ValueError(f"Seconds must be between 0 and 59, got {seconds}")
            if minutes < 0 or minutes >= 60:
                raise ValueError(f"Minutes must be between 0 and 59, got {minutes}")
            if hours < 0:
                raise ValueError(f"Hours cannot be negative, got {hours}")
            return hours * 3600 + minutes * 60 + seconds
        except ValueError as e:
            if "must be between" in str(e) or "cannot be negative" in str(e):
                raise
            raise ValueError(
                f"Invalid time format: '{time_str}'. "
                f"Expected formats: '2:30' (M:SS), '90' (seconds), or '1:02:30' (H:MM:SS)"
            )
    
    else:
        raise ValueError(
            f"Invalid time format: '{time_str}'. "
            f"Expected formats: '2:30' (M:SS), '90' (seconds), or '1:02:30' (H:MM:SS)"
        )


def parse_time_control(tc_str: str) -> float:
    """
    Parse a time control string into starting seconds.
    
    Accepted formats:
        "3+0"   -> 3 minutes = 180 seconds
        "5+3"   -> 5 minutes = 300 seconds (increment ignored for starting time)
        "10+0"  -> 10 minutes = 600 seconds
        "180"   -> 180 seconds
    
    Returns:
        Starting time in seconds as a float
    
    Raises:
        ValueError if format is invalid
    """
    tc_str = tc_str.strip()
    
    # Try M+I format (e.g., "3+0", "5+3")
    if '+' in tc_str:
        parts = tc_str.split('+')
        if len(parts) != 2:
            raise ValueError(
                f"Invalid time control format: '{tc_str}'. "
                f"Expected formats: '3+0', '5+3', '10+0', or '180' (seconds)"
            )
        try:
            minutes = float(parts[0])
            # We ignore the increment for starting time calculation
            return minutes * 60
        except ValueError:
            raise ValueError(
                f"Invalid time control format: '{tc_str}'. "
                f"Expected formats: '3+0', '5+3', '10+0', or '180' (seconds)"
            )
    
    # Try pure number (seconds)
    try:
        return float(tc_str)
    except ValueError:
        raise ValueError(
            f"Invalid time control format: '{tc_str}'. "
            f"Expected formats: '3+0', '5+3', '10+0', or '180' (seconds)"
        )


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
    return 1


def get_move_number_from_fen(fen: str) -> int:
    """Extract full move number from FEN."""
    parts = fen.split()
    if len(parts) >= 6:
        try:
            return int(parts[5])
        except ValueError:
            pass
    return 1


def format_time(seconds: float) -> str:
    """Format seconds as M:SS for display."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    if secs == int(secs):
        return f"{minutes}:{int(secs):02d}"
    else:
        return f"{minutes}:{secs:05.2f}"


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
    }


def predict_with_model(features: dict, model_path: str) -> float:
    """Predict win probability using a saved sklearn model."""
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    model = data['model']
    scaler = data['scaler']
    feature_names = data['features']
    
    # Check that all required features are available
    missing_features = [f for f in feature_names if f not in features]
    if missing_features:
        print(f"Error: Model requires features not available: {missing_features}")
        sys.exit(1)
    
    # Build feature vector in correct order
    X = np.array([[features[f] for f in feature_names]])
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]
    
    return prob


def main():
    parser = argparse.ArgumentParser(
        description='Predict chess win probability from position',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict.py --model models/model_1500_2000.pkl \\
        --fen "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1" \\
        --white_elo 1600 --black_elo 1550 \\
        --white_time "2:30" --black_time "2:15" \\
        --time_control "3+0"

Time formats:
    --white_time "2:30"     2 minutes 30 seconds
    --white_time "90"       90 seconds
    --white_time "1:02:30"  1 hour 2 minutes 30 seconds
    
    --time_control "3+0"    3 minute game (no increment)
    --time_control "5+3"    5 minutes + 3 second increment
    --time_control "180"    180 seconds
        """
    )
    parser.add_argument('--model', required=True, help='Path to saved model .pkl file')
    parser.add_argument('--fen', required=True, help='FEN string of the position')
    parser.add_argument('--white_elo', type=int, required=True, help="White's Elo rating")
    parser.add_argument('--black_elo', type=int, required=True, help="Black's Elo rating")
    parser.add_argument('--white_time', required=True, help="White's remaining time (e.g., '2:30' or '150')")
    parser.add_argument('--black_time', required=True, help="Black's remaining time (e.g., '2:15' or '135')")
    parser.add_argument('--time_control', required=True, help="Time control (e.g., '3+0', '5+3', or '180')")
    parser.add_argument('--move_number', type=int, default=None, help='Move number (optional, extracted from FEN if not provided)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed feature breakdown')
    
    args = parser.parse_args()
    
    # Parse time arguments
    try:
        white_time = parse_time(args.white_time)
    except ValueError as e:
        print(f"Error parsing --white_time: {e}")
        sys.exit(1)
    
    try:
        black_time = parse_time(args.black_time)
    except ValueError as e:
        print(f"Error parsing --black_time: {e}")
        sys.exit(1)
    
    try:
        time_control = parse_time_control(args.time_control)
    except ValueError as e:
        print(f"Error parsing --time_control: {e}")
        sys.exit(1)
    
    # Validate times make sense
    if white_time > time_control:
        print(f"Warning: White's time ({format_time(white_time)}) exceeds starting time ({format_time(time_control)})")
    if black_time > time_control:
        print(f"Warning: Black's time ({format_time(black_time)}) exceeds starting time ({format_time(time_control)})")
    
    # Extract features
    features = extract_features(
        args.fen, args.white_elo, args.black_elo,
        white_time, black_time, time_control,
        args.move_number
    )
    
    # Predict
    prob = predict_with_model(features, args.model)
    
    # Output
    if args.verbose:
        print("\n=== Position Analysis ===")
        print(f"FEN: {args.fen}")
        print(f"Side to move: {'White' if features['side_to_move'] == 1 else 'Black'}")
        print(f"\n=== Ratings ===")
        print(f"White Elo: {args.white_elo}")
        print(f"Black Elo: {args.black_elo}")
        print(f"Elo difference (W-B): {features['elo_diff']}")
        print(f"\n=== Time ===")
        print(f"Time control: {args.time_control} ({format_time(time_control)})")
        print(f"White time: {format_time(white_time)} ({features['white_time_ratio']:.1%} remaining)")
        print(f"Black time: {format_time(black_time)} ({features['black_time_ratio']:.1%} remaining)")
        print(f"\n=== Features ===")
        print(f"Material balance: {features['material_balance']} centipawns")
        print(f"Move number: {features['move_number']}")
        print(f"Time ratio diff: {features['time_ratio_diff']:+.1%}")
        print(f"\n=== Prediction ===")
    
    print(f"White win probability: {prob:.1%}")
    print(f"Black win probability: {1-prob:.1%}")


if __name__ == '__main__':
    main()