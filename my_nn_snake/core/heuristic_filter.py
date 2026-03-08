from collections import deque

MOVES = {
    "up":    (0,  1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": ( 1, 0),
}

def _next_head(head, move):
    dx, dy = MOVES[move]
    return {"x": head["x"] + dx, "y": head["y"] + dy}

def _on_board(pos, width, height):
    return 0 <= pos["x"] < width and 0 <= pos["y"] < height

def _build_occupied(board, exclude_tails=True):
    """
    Return a set of (x, y) tuples that are occupied by snake bodies.
    By default, excludes the very last tail segment because it will
    vacate the square on the next turn (unless the snake just ate).
    """
    occupied = set()
    for snake in board["snakes"]:
        body = snake["body"]
        # Skip the tail (it will move) unless it's the same position as
        # the second-to-last segment (i.e. the snake just ate and grew)
        skip_tail = (
            len(body) >= 2 and
            body[-1]["x"] != body[-2]["x"] or
            body[-1]["y"] != body[-2]["y"]
        )
        limit = len(body) - 1 if exclude_tails and skip_tail else len(body)
        for part in body[:limit]:
            occupied.add((part["x"], part["y"]))
    return occupied

def _flood_fill_area(start, board, occupied):
    """
    BFS flood fill from `start` position.
    Returns the number of reachable empty squares.
    """
    width = board["width"]
    height = board["height"]
    start_t = (start["x"], start["y"])
    if start_t in occupied:
        return 0

    visited = {start_t}
    queue = deque([start_t])
    count = 0

    while queue:
        x, y = queue.popleft()
        count += 1
        for dx, dy in MOVES.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                nxt = (nx, ny)
                if nxt not in visited and nxt not in occupied:
                    visited.add(nxt)
                    queue.append(nxt)
    return count

def get_safe_moves(data):
    """
    Main entry point.  Returns a dict:
        {
          "up":    True/False,
          "down":  True/False,
          "left":  True/False,
          "right": True/False,
        }
    A move is True when it is considered "safe":
      1. Does not go out of bounds.
      2. Does not collide with any snake body (tails excluded).
      3. Does not walk into a square adjacent to a larger-or-equal enemy head
         (avoidable head-to-head).
      4. Leaves at least `min_area` reachable squares (flood fill).
    """
    board = data["board"]
    you   = data["you"]
    head  = you["head"]
    my_length = you["length"]

    width  = board["width"]
    height = board["height"]

    occupied = _build_occupied(board, exclude_tails=True)

    # Squares that an equal-or-larger enemy could move into next turn
    danger_squares = set()
    for snake in board["snakes"]:
        if snake["id"] == you["id"]:
            continue
        if snake["length"] >= my_length:
            for dx, dy in MOVES.values():
                nx = snake["head"]["x"] + dx
                ny = snake["head"]["y"] + dy
                if 0 <= nx < width and 0 <= ny < height:
                    danger_squares.add((nx, ny))

    # Minimum flood-fill area threshold: at least our own length
    # so we always have room to survive.
    min_area = max(1, my_length // 2)

    result = {}
    for move in MOVES:
        nxt = _next_head(head, move)

        # 1. In bounds?
        if not _on_board(nxt, width, height):
            result[move] = False
            continue

        nxt_t = (nxt["x"], nxt["y"])

        # 2. Body collision?
        if nxt_t in occupied:
            result[move] = False
            continue

        # 3. Dangerous head-to-head square?
        if nxt_t in danger_squares:
            result[move] = False
            continue

        # 4. Flood-fill reachability
        # Temporarily mark the next head as occupied for the fill
        occupied.add(nxt_t)
        area = _flood_fill_area(nxt, board, occupied)
        occupied.discard(nxt_t)

        result[move] = (area >= min_area)

    # Safety fallback: if ALL moves are blocked by flood fill,
    # relax and allow any non-wall, non-body move (avoid instant death only).
    if not any(result.values()):
        for move in MOVES:
            nxt = _next_head(head, move)
            nxt_t = (nxt["x"], nxt["y"])
            result[move] = (
                _on_board(nxt, width, height) and
                nxt_t not in occupied
            )

    return result


def apply_mask(logits, safe_moves, MOVE_IDX=None):
    """
    Zero-out (set to -1e9) logits for unsafe moves.
    logits: list[float] of length 4  [up, down, left, right]
    safe_moves: dict from get_safe_moves()
    Returns a new list with unsafe logits masked to -inf.
    """
    if MOVE_IDX is None:
        MOVE_IDX = {"up": 0, "down": 1, "left": 2, "right": 3}

    masked = list(logits)
    for move, idx in MOVE_IDX.items():
        if not safe_moves.get(move, False):
            masked[idx] = -1e9
    return masked
