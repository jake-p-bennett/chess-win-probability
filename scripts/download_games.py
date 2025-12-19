"""
Download games from Lichess API.

This script provides several methods to get blitz games:
1. Download games from specific users
2. Download from players in a specific rating range
3. Download from tournament participants

Usage:
    # Download blitz games from a list of usernames
    python download_games.py --users DrNykterstein Hikaru --max_per_user 500 --output games.pgn
    
    # Download from players rated 1400-1800 (finds players via recent tournaments)
    python download_games.py --rating_range 1400 1800 --num_games 50000 --output games.pgn
    
    # Download from top N players
    python download_games.py --top_players 20 --max_per_user 500 --output games.pgn
"""

import requests
import argparse
import time
import sys
import random


# Lichess API base URL
BASE_URL = "https://lichess.org"

# Be respectful of rate limits
REQUEST_DELAY = 1.5  # seconds between requests


def download_user_games(
    username: str,
    max_games: int = 500,
    perf_type: str = "blitz",
    with_clocks: bool = True,
) -> str:
    """
    Download games for a specific user.
    
    Args:
        username: Lichess username
        max_games: Maximum number of games to download
        perf_type: Game type (blitz, rapid, bullet, classical)
        with_clocks: Include clock times in PGN
    
    Returns:
        PGN string of games
    """
    url = f"{BASE_URL}/api/games/user/{username}"
    params = {
        "max": max_games,
        "perfType": perf_type,
        "clocks": str(with_clocks).lower(),
        "rated": "true",
        "pgnInJson": "false",
    }
    
    headers = {
        "Accept": "application/x-chess-pgn",
    }
    
    print(f"Downloading up to {max_games} {perf_type} games from {username}...")
    
    response = requests.get(url, params=params, headers=headers, stream=True)
    
    if response.status_code == 429:
        print("Rate limited! Please wait a minute and try again.")
        return ""
    
    response.raise_for_status()
    
    # Stream the response to handle large downloads
    pgn_data = []
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            # Decode bytes to string
            if isinstance(chunk, bytes):
                chunk = chunk.decode('utf-8')
            pgn_data.append(chunk)
    
    pgn = "".join(pgn_data)
    
    # Filter out non-standard variants (keep only games without Variant tag or with Variant "Standard")
    games = pgn.split("\n\n[Event")
    filtered_games = []
    for i, game in enumerate(games):
        if i > 0:
            game = "[Event" + game
        # Skip games with non-standard variants
        if '[Variant "' in game and '[Variant "Standard"]' not in game:
            continue
        filtered_games.append(game)
    
    pgn = "\n\n".join(filtered_games)
    
    game_count = pgn.count('[Event "')
    print(f"  Downloaded {game_count} games from {username}")
    
    return pgn


def download_from_multiple_users(
    usernames: list[str],
    max_per_user: int = 500,
    perf_type: str = "blitz",
    with_clocks: bool = True,
) -> str:
    """Download games from multiple users with rate limiting."""
    all_pgns = []
    
    for i, username in enumerate(usernames):
        if i > 0:
            print(f"Waiting {REQUEST_DELAY}s before next request...")
            time.sleep(REQUEST_DELAY)
        
        try:
            pgn = download_user_games(username, max_per_user, perf_type, with_clocks)
            if pgn:
                all_pgns.append(pgn)
        except requests.RequestException as e:
            print(f"  Error downloading from {username}: {e}")
    
    return "\n\n".join(all_pgns)


def get_top_players(perf_type: str = "blitz", count: int = 50) -> list[str]:
    """Get usernames of top players in a given category."""
    url = f"{BASE_URL}/api/player/top/{count}/{perf_type}"
    
    headers = {
        "Accept": "application/vnd.lichess.v3+json",
    }
    
    print(f"Fetching top {count} {perf_type} players...")
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    usernames = [user["username"] for user in data["users"]]
    
    print(f"  Found {len(usernames)} players")
    return usernames


def get_swiss_tournament_players(tournament_id: str) -> list[dict]:
    """Get players from a Swiss tournament with their ratings."""
    url = f"{BASE_URL}/api/swiss/{tournament_id}/results"
    
    headers = {
        "Accept": "application/x-ndjson",
    }
    
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    players = []
    for line in response.iter_lines(decode_unicode=True):
        if line:
            import json
            player = json.loads(line)
            players.append({
                "username": player.get("username"),
                "rating": player.get("rating", 0),
            })
    
    return players


def get_arena_tournament_players(tournament_id: str) -> list[dict]:
    """Get players from an arena tournament with their ratings."""
    url = f"{BASE_URL}/api/tournament/{tournament_id}/results"
    
    headers = {
        "Accept": "application/x-ndjson",
    }
    
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    players = []
    for line in response.iter_lines(decode_unicode=True):
        if line:
            import json
            player = json.loads(line)
            players.append({
                "username": player.get("username"),
                "rating": player.get("rating", 0),
            })
    
    return players


def get_recent_tournaments(perf_type: str = "blitz", count: int = 20) -> list[str]:
    """Get IDs of recent arena tournaments."""
    # The tournament API endpoint for listing tournaments
    url = f"{BASE_URL}/api/tournament"
    
    headers = {
        "Accept": "application/json",
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    
    # Filter for the right perf type and get finished tournaments
    tournament_ids = []
    
    for t in data.get("finished", []):
        if t.get("perf", {}).get("key") == perf_type:
            tournament_ids.append(t["id"])
            if len(tournament_ids) >= count:
                break
    
    return tournament_ids


def get_leaderboard(perf_type: str = "blitz", count: int = 200) -> list[dict]:
    """Get top players from the leaderboard with their ratings."""
    url = f"{BASE_URL}/api/player/top/{count}/{perf_type}"
    
    headers = {
        "Accept": "application/vnd.lichess.v3+json",
    }
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    players = []
    for user in data.get("users", []):
        players.append({
            "username": user.get("username"),
            "rating": user.get("perfs", {}).get(perf_type, {}).get("rating", 0),
        })
    
    return players


def find_players_in_rating_range(
    min_rating: int,
    max_rating: int,
    perf_type: str = "blitz",
    target_count: int = 200,
) -> list[str]:
    """
    Find players within a rating range by scraping recent tournaments.
    For high ratings (2400+), also checks the leaderboard.
    
    This is a bit slow but effective way to find players at various rating levels.
    """
    print(f"Finding players rated {min_rating}-{max_rating} in {perf_type}...")
    
    players_found = set()
    
    # For high ratings, start with the leaderboard
    if min_rating >= 2400:
        print("  High rating range detected, fetching from leaderboard...")
        time.sleep(REQUEST_DELAY)
        try:
            leaderboard = get_leaderboard(perf_type, count=200)
            for p in leaderboard:
                if min_rating <= p["rating"] <= max_rating:
                    players_found.add(p["username"])
            print(f"  Found {len(players_found)} players from leaderboard")
        except requests.RequestException as e:
            print(f"  Error fetching leaderboard: {e}")
    
    # If we still need more players, scan tournaments
    if len(players_found) < target_count:
        print("  Fetching recent tournaments...")
        tournament_ids = get_recent_tournaments(perf_type, count=50)
        print(f"  Found {len(tournament_ids)} recent {perf_type} tournaments")
        
        for i, tid in enumerate(tournament_ids):
            if len(players_found) >= target_count:
                break
                
            try:
                time.sleep(REQUEST_DELAY)
                print(f"  Scanning tournament {i+1}/{len(tournament_ids)} ({tid})...")
                
                players = get_arena_tournament_players(tid)
                
                for p in players:
                    if min_rating <= p["rating"] <= max_rating:
                        players_found.add(p["username"])
                        
                print(f"    Found {len(players_found)} players so far")
                
            except requests.RequestException as e:
                print(f"    Error: {e}")
                continue
    
    result = list(players_found)
    random.shuffle(result)  # Randomize order
    
    print(f"  Total: {len(result)} players in rating range {min_rating}-{max_rating}")
    
    if len(result) < target_count:
        print(f"  Warning: Only found {len(result)} players, fewer than target {target_count}")
    
    return result[:target_count]


def download_games_by_ids(game_ids: list[str], with_clocks: bool = True) -> str:
    """
    Download specific games by their IDs.
    
    Args:
        game_ids: List of Lichess game IDs
        with_clocks: Include clock times
    
    Returns:
        PGN string
    """
    url = f"{BASE_URL}/api/games/export/_ids"
    
    params = {
        "clocks": str(with_clocks).lower(),
    }
    
    headers = {
        "Accept": "application/x-chess-pgn",
    }
    
    # API accepts up to 300 IDs at a time
    batch_size = 300
    all_pgns = []
    
    for i in range(0, len(game_ids), batch_size):
        batch = game_ids[i:i + batch_size]
        ids_str = ",".join(batch)
        
        print(f"Downloading batch {i // batch_size + 1} ({len(batch)} games)...")
        
        response = requests.post(url, params=params, headers=headers, data=ids_str)
        
        if response.status_code == 429:
            print("Rate limited! Waiting 60 seconds...")
            time.sleep(60)
            response = requests.post(url, params=params, headers=headers, data=ids_str)
        
        response.raise_for_status()
        all_pgns.append(response.text)
        
        if i + batch_size < len(game_ids):
            time.sleep(REQUEST_DELAY)
    
    return "\n\n".join(all_pgns)


def main():
    parser = argparse.ArgumentParser(description="Download games from Lichess API")
    
    parser.add_argument(
        "--users",
        nargs="+",
        help="List of usernames to download games from",
    )
    parser.add_argument(
        "--top_players",
        type=int,
        metavar="N",
        help="Download from top N players in the category",
    )
    parser.add_argument(
        "--rating_range",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        help="Find players in this rating range (e.g., --rating_range 1400 1800)",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=10000,
        help="Target total number of games when using --rating_range (default: 10000)",
    )
    parser.add_argument(
        "--max_per_user",
        type=int,
        default=500,
        help="Maximum games per user (default: 500)",
    )
    parser.add_argument(
        "--perf_type",
        default="blitz",
        choices=["bullet", "blitz", "rapid", "classical"],
        help="Game type (default: blitz)",
    )
    parser.add_argument(
        "--output",
        default="games.pgn",
        help="Output PGN file (default: games.pgn)",
    )
    parser.add_argument(
        "--no_clocks",
        action="store_true",
        help="Don't include clock times (faster download)",
    )
    
    args = parser.parse_args()
    
    with_clocks = not args.no_clocks
    
    # Determine which users to download from
    usernames = []
    max_per_user = args.max_per_user
    
    if args.users:
        usernames = args.users
    elif args.top_players:
        usernames = get_top_players(args.perf_type, args.top_players)
        time.sleep(REQUEST_DELAY)
    elif args.rating_range:
        min_rating, max_rating = args.rating_range
        # Calculate how many players we need
        # Assume we get ~80% of max_per_user on average
        estimated_games_per_player = max_per_user * 0.8
        num_players_needed = int(args.num_games / estimated_games_per_player) + 10
        
        usernames = find_players_in_rating_range(
            min_rating,
            max_rating,
            args.perf_type,
            target_count=num_players_needed,
        )
        time.sleep(REQUEST_DELAY)
    else:
        print("No users specified. Use --users, --top_players, or --rating_range")
        print("\nExample usage:")
        print("  python download_games.py --users DrNykterstein --max_per_user 1000 --output games.pgn")
        print("  python download_games.py --top_players 10 --max_per_user 200 --output games.pgn")
        print("  python download_games.py --rating_range 1400 1800 --num_games 50000 --output games.pgn")
        sys.exit(1)
    
    if not usernames:
        print("No players found matching criteria.")
        sys.exit(1)
    
    # Download games
    pgn = download_from_multiple_users(
        usernames,
        max_per_user,
        args.perf_type,
        with_clocks,
    )
    
    # Save to file
    if pgn:
        with open(args.output, "w") as f:
            f.write(pgn)
        
        total_games = pgn.count('[Event "')
        print(f"\nSaved {total_games} games to {args.output}")
    else:
        print("No games downloaded.")


if __name__ == "__main__":
    main()