"""
Chess Win Probability Calculator
Extract features from Lichess PGN database for win probability modeling.

Usage:
    python extract_features.py input.pgn output.csv --num_games 10000
"""

import chess
import chess.pgn
import csv
import random
import argparse
from dataclasses import dataclass
from typing import Optional
import io


@dataclass
class PositionFeatures:
    """Features extracted from a single chess position."""
    # Material (positive = white advantage)
    material_balance: int  # centipawn-style: Q=900, R=500, B=330, N=320, P=100
    white_pawns: int
    white_knights: int
    white_bishops: int
    white_rooks: int
    white_queens: int
    black_pawns: int
    black_knights: int
    black_bishops: int
    black_rooks: int
    black_queens: int
    
    # Positional
    side_to_move: int  # 1 = white, 0 = black
    move_number: int
    white_can_castle_kingside: int
    white_can_castle_queenside: int
    black_can_castle_kingside: int
    black_can_castle_queenside: int
    
    # Player info
    white_elo: int
    black_elo: int
    elo_diff: int  # white_elo - black_elo
    
    # Time (if available)
    white_time_remaining: Optional[float]  # seconds
    black_time_remaining: Optional[float]
    white_time_ratio: Optional[float]  # time_remaining / starting_time
    black_time_ratio: Optional[float]
    time_ratio_diff: Optional[float]  # white_ratio - black_ratio
    
    # Target
    white_wins: int  # 1 = white wins, 0 = black wins
    
    # Metadata
    game_url: Optional[str] = None  # Lichess game URL


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def calculate_material_balance(board: chess.Board) -> int:
    """Calculate material balance in centipawns (positive = white advantage)."""
    balance = 0
    for piece_type, value in PIECE_VALUES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        balance += (white_count - black_count) * value
    return balance


def count_pieces(board: chess.Board, color: chess.Color) -> dict:
    """Count pieces for a given color."""
    return {
        'pawns': len(board.pieces(chess.PAWN, color)),
        'knights': len(board.pieces(chess.KNIGHT, color)),
        'bishops': len(board.pieces(chess.BISHOP, color)),
        'rooks': len(board.pieces(chess.ROOK, color)),
        'queens': len(board.pieces(chess.QUEEN, color)),
    }


def parse_time_control(time_control: str) -> Optional[tuple[int, int]]:
    """
    Parse time control string like '180+0' or '300+3'.
    Returns (base_seconds, increment_seconds) or None if unparseable.
    """
    if not time_control or time_control == '-':
        return None
    try:
        parts = time_control.split('+')
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except ValueError:
        pass
    return None


def parse_clock(comment: str) -> Optional[float]:
    """
    Parse clock time from a move comment like '[%clk 0:02:45]'.
    Returns time in seconds or None.
    """
    if '[%clk' not in comment:
        return None
    try:
        start = comment.index('[%clk') + 6
        end = comment.index(']', start)
        time_str = comment[start:end].strip()
        parts = time_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def extract_features_from_position(
    board: chess.Board,
    white_elo: int,
    black_elo: int,
    move_number: int,
    white_time: Optional[float],
    black_time: Optional[float],
    starting_time: Optional[int],
    white_wins: int,
    game_url: Optional[str] = None
) -> PositionFeatures:
    """Extract all features from a position."""
    
    white_pieces = count_pieces(board, chess.WHITE)
    black_pieces = count_pieces(board, chess.BLACK)
    
    # Calculate time ratios if we have time info
    white_time_ratio = None
    black_time_ratio = None
    time_ratio_diff = None
    
    if white_time is not None and black_time is not None and starting_time:
        white_time_ratio = white_time / starting_time
        black_time_ratio = black_time / starting_time
        time_ratio_diff = white_time_ratio - black_time_ratio
    
    return PositionFeatures(
        material_balance=calculate_material_balance(board),
        white_pawns=white_pieces['pawns'],
        white_knights=white_pieces['knights'],
        white_bishops=white_pieces['bishops'],
        white_rooks=white_pieces['rooks'],
        white_queens=white_pieces['queens'],
        black_pawns=black_pieces['pawns'],
        black_knights=black_pieces['knights'],
        black_bishops=black_pieces['bishops'],
        black_rooks=black_pieces['rooks'],
        black_queens=black_pieces['queens'],
        side_to_move=1 if board.turn == chess.WHITE else 0,
        move_number=move_number,
        white_can_castle_kingside=int(board.has_kingside_castling_rights(chess.WHITE)),
        white_can_castle_queenside=int(board.has_queenside_castling_rights(chess.WHITE)),
        black_can_castle_kingside=int(board.has_kingside_castling_rights(chess.BLACK)),
        black_can_castle_queenside=int(board.has_queenside_castling_rights(chess.BLACK)),
        white_elo=white_elo,
        black_elo=black_elo,
        elo_diff=white_elo - black_elo,
        white_time_remaining=white_time,
        black_time_remaining=black_time,
        white_time_ratio=white_time_ratio,
        black_time_ratio=black_time_ratio,
        time_ratio_diff=time_ratio_diff,
        white_wins=white_wins,
        game_url=game_url,
    )


def process_game(game: chess.pgn.Game, time_control_filter: str = None) -> Optional[PositionFeatures]:
    """
    Process a single game and extract features from a random position.
    Returns None if the game should be skipped (draw, missing info, etc.)
    
    Args:
        game: The chess game to process
        time_control_filter: If provided, only process games with this exact time control
                            (e.g., "180+0" for 3+0, "300+0" for 5+0, "600+0" for 10+0)
    """
    # Get result
    result = game.headers.get('Result', '*')
    if result == '1-0':
        white_wins = 1
    elif result == '0-1':
        white_wins = 0
    else:
        # Skip draws and unfinished games
        return None
    
    # Get game URL (Lichess stores this in Site header)
    game_url = game.headers.get('Site', None)
    
    # Get time control
    time_control = game.headers.get('TimeControl', '')
    
    # Filter by time control if specified
    if time_control_filter is not None and time_control != time_control_filter:
        return None
    
    time_info = parse_time_control(time_control)
    starting_time = time_info[0] if time_info else None
    
    # Get ELOs
    try:
        white_elo = int(game.headers.get('WhiteElo', 0))
        black_elo = int(game.headers.get('BlackElo', 0))
        if white_elo == 0 or black_elo == 0:
            return None
    except ValueError:
        return None
    
    # Collect all positions with their clock times
    positions = []
    board = game.board()
    node = game
    move_number = 0
    
    white_time = starting_time
    black_time = starting_time
    
    while node.variations:
        node = node.variations[0]
        move_number += 1
        board.push(node.move)
        
        # Parse clock from comment
        if node.comment:
            clock = parse_clock(node.comment)
            if clock is not None:
                # Clock is for the player who just moved
                if board.turn == chess.BLACK:  # White just moved
                    white_time = clock
                else:
                    black_time = clock
        
        positions.append({
            'fen': board.fen(),
            'move_number': move_number,
            'white_time': white_time,
            'black_time': black_time,
        })
    
    if len(positions) < 10:
        # Skip very short games
        return None
    
    # Sample a random position (avoiding first 5 and last 5 moves)
    start_idx = min(5, len(positions) // 4)
    end_idx = max(start_idx + 1, len(positions) - 5)
    
    if start_idx >= end_idx:
        return None
        
    sampled = random.choice(positions[start_idx:end_idx])
    
    board = chess.Board(sampled['fen'])
    
    return extract_features_from_position(
        board=board,
        white_elo=white_elo,
        black_elo=black_elo,
        move_number=sampled['move_number'],
        white_time=sampled['white_time'],
        black_time=sampled['black_time'],
        starting_time=starting_time,
        white_wins=white_wins,
        game_url=game_url,
    )


def process_pgn_file(pgn_path: str, output_path: str, num_games: int = 10000, 
                     time_control_filter: str = None, verbose: bool = True):
    """Process a PGN file and write features to CSV.
    
    Args:
        pgn_path: Path to input PGN file
        output_path: Path to output CSV file
        num_games: Number of games to extract
        time_control_filter: If provided, only process games with this exact time control
                            (e.g., "180+0" for 3+0, "300+0" for 5+0, "600+0" for 10+0)
        verbose: Whether to print progress
    """
    
    if verbose and time_control_filter:
        print(f"Filtering for time control: {time_control_filter}")
    
    fieldnames = [
        'material_balance', 
        'white_pawns', 'white_knights', 'white_bishops', 'white_rooks', 'white_queens',
        'black_pawns', 'black_knights', 'black_bishops', 'black_rooks', 'black_queens',
        'side_to_move', 'move_number',
        'white_can_castle_kingside', 'white_can_castle_queenside',
        'black_can_castle_kingside', 'black_can_castle_queenside',
        'white_elo', 'black_elo', 'elo_diff',
        'white_time_remaining', 'black_time_remaining',
        'white_time_ratio', 'black_time_ratio', 'time_ratio_diff',
        'white_wins',
        'game_url'
    ]
    
    with open(pgn_path, 'r') as pgn_file, open(output_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        games_processed = 0
        games_written = 0
        
        while games_written < num_games:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            games_processed += 1
            
            features = process_game(game, time_control_filter)
            if features is not None:
                writer.writerow(features.__dict__)
                games_written += 1
                
                if verbose and games_written % 1000 == 0:
                    print(f"Processed {games_processed} games, wrote {games_written} samples")
        
        if verbose:
            print(f"\nDone! Processed {games_processed} games, wrote {games_written} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract features from Lichess PGN for win probability modeling')
    parser.add_argument('input_pgn', help='Input PGN file path')
    parser.add_argument('output_csv', help='Output CSV file path')
    parser.add_argument('--num_games', type=int, default=10000, help='Number of games to sample (default: 10000)')
    parser.add_argument('--time_control', type=str, default=None, 
                        help='Filter for specific time control (e.g., "180+0" for 3+0, "300+0" for 5+0, "600+0" for 10+0)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    
    args = parser.parse_args()
    
    process_pgn_file(args.input_pgn, args.output_csv, args.num_games, 
                     time_control_filter=args.time_control, verbose=not args.quiet)


if __name__ == '__main__':
    main()