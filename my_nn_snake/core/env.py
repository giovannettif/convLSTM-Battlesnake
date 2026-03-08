"""
env.py — Fast in-memory Battlesnake simulator for training.

Implements a simplified Gym-like interface:
    env = BattleSnakeEnv(width=11, height=11, num_snakes=2)
    obs   = env.reset()
    obs, rewards, done, info = env.step(actions)

Actions: dict {snake_id: move_str}  where move_str in {"up","down","left","right"}
Rewards: dict {snake_id: float}
Obs: dict {snake_id: {"board": data}}  — a dict in the same format as the live API
"""

import random
from copy import deepcopy

MOVES = {
    "up":    (0,  1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": ( 1, 0),
}

MOVE_NAMES = list(MOVES.keys())
MOVE_IDX = {m: i for i, m in enumerate(MOVE_NAMES)}


class Snake:
    def __init__(self, sid, body, health=100):
        self.id = sid
        self.body = list(body)   # list of {"x":int, "y":int}, head first
        self.health = health
        self.alive = True
        self.just_ate = False

    @property
    def head(self):
        return self.body[0]

    @property
    def length(self):
        return len(self.body)

    def move(self, direction):
        dx, dy = MOVES[direction]
        new_head = {"x": self.head["x"] + dx, "y": self.head["y"] + dy}
        self.body.insert(0, new_head)
        if not self.just_ate:
            self.body.pop()        # remove tail
        self.just_ate = False
        self.health -= 1


class BattleSnakeEnv:
    """
    Lightweight Battlesnake simulator for reinforcement-learning training.
    
    Key parameters
    --------------
    width, height : board dimensions
    num_snakes    : number of snakes (all controlled by the agent during self-play)
    food_spawn_chance : % chance of spawning food each turn (0-100)
    min_food      : minimum food on board at all times
    hazard_damage : HP drained per turn when inside a hazard cell
    royale_shrink_every : turns between hazard-ring expansions (0 = disabled)
    royale_start_turn   : turn at which royale mode kicks in
    max_turns     : episode cap (0 = no cap)
    """

    def __init__(
        self,
        width=11, height=11,
        num_snakes=2,
        food_spawn_chance=15,
        min_food=1,
        hazard_damage=14,
        royale_shrink_every=0,
        royale_start_turn=1000,
        max_turns=0,
    ):
        self.width = width
        self.height = height
        self.num_snakes = num_snakes
        self.food_spawn_chance = food_spawn_chance
        self.min_food = min_food
        self.hazard_damage = hazard_damage
        self.royale_shrink_every = royale_shrink_every
        self.royale_start_turn = royale_start_turn
        self.max_turns = max_turns

        # Will be populated on reset()
        self.snakes = {}
        self.food = []
        self.hazards = set()
        self.turn = 0
        self._hazard_ring = 0   # how many rings of hazard have been applied

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reinitialise and return initial observations."""
        self.turn = 0
        self._hazard_ring = 0
        self.hazards = set()

        # Place snakes at random starting positions (cardinal 4 + corners)
        starts = self._random_starts(self.num_snakes)
        self.snakes = {}
        for i, start in enumerate(starts):
            sid = f"snake_{i}"
            # Start with 3-segment stack on top of each other (standard)
            body = [start, start.copy(), start.copy()]
            self.snakes[sid] = Snake(sid, body)

        # Seed food
        self.food = []
        self._maybe_spawn_food(force_min=True)

        return self._get_observations()

    def step(self, actions: dict):
        """
        actions: {snake_id: direction_str}
        Returns: (observations, rewards, done, info)
        """
        self.turn += 1

        # 1. Move all snakes
        for sid, snake in self.snakes.items():
            if not snake.alive:
                continue
            direction = actions.get(sid, random.choice(MOVE_NAMES))
            snake.move(direction)

        # 2. Eliminate out-of-bounds
        for sid, snake in self.snakes.items():
            if not snake.alive:
                continue
            hx, hy = snake.head["x"], snake.head["y"]
            if not (0 <= hx < self.width and 0 <= hy < self.height):
                snake.alive = False
                snake.health = 0

        # 3. Eliminate body collisions
        all_bodies = set()
        # Build set of non-head body positions
        for sid, s in self.snakes.items():
            if s.alive:
                for seg in s.body[1:]:
                    all_bodies.add((seg["x"], seg["y"]))

        for sid, snake in self.snakes.items():
            if not snake.alive:
                continue
            if (snake.head["x"], snake.head["y"]) in all_bodies:
                snake.alive = False
                snake.health = 0

        # 4. Head-to-head collisions
        heads = {}
        for sid, snake in self.snakes.items():
            if snake.alive:
                key = (snake.head["x"], snake.head["y"])
                if key in heads:
                    other_sid = heads[key]
                    other = self.snakes[other_sid]
                    # Shorter snake dies; tie = both die
                    if snake.length < other.length:
                        snake.alive = False
                    elif snake.length > other.length:
                        other.alive = False
                    else:
                        snake.alive = False
                        other.alive = False
                else:
                    heads[key] = sid

        # 5. Food consumption
        food_set = {(f["x"], f["y"]) for f in self.food}
        eaten = set()
        for sid, snake in self.snakes.items():
            if not snake.alive:
                continue
            hk = (snake.head["x"], snake.head["y"])
            if hk in food_set:
                snake.health = 100
                snake.just_ate = True
                # Grow: re-append the current tail
                snake.body.append(deepcopy(snake.body[-1]))
                eaten.add(hk)

        self.food = [f for f in self.food if (f["x"], f["y"]) not in eaten]

        # 6. Hazard damage
        for sid, snake in self.snakes.items():
            if not snake.alive:
                continue
            if (snake.head["x"], snake.head["y"]) in self.hazards:
                snake.health -= self.hazard_damage
            if snake.health <= 0:
                snake.alive = False

        # 7. Starvation
        for sid, snake in self.snakes.items():
            if snake.alive and snake.health <= 0:
                snake.alive = False

        # 8. Royale hazard expansion
        if (self.royale_shrink_every > 0 and
                self.turn >= self.royale_start_turn and
                (self.turn - self.royale_start_turn) % self.royale_shrink_every == 0):
            self._expand_hazard_ring()

        # 9. Spawn food
        self._maybe_spawn_food()

        # 10. Compute rewards
        alive_snakes = [s for s in self.snakes.values() if s.alive]
        rewards = {}
        for sid, snake in self.snakes.items():
            if not snake.alive:
                rewards[sid] = -1.0
            else:
                if self.num_snakes > 1:
                    if len(alive_snakes) == 1 and alive_snakes[0].id == sid:
                        rewards[sid] = 1.0
                    else:
                        rewards[sid] = 0.01
                else:
                    # Solo survival: just reward staying alive
                    rewards[sid] = 0.1
                
                # Boost reward for eating to help early learning
                if snake.just_ate:
                    rewards[sid] += 1.0

        if self.num_snakes > 1:
            done = len(alive_snakes) <= 1
        else:
            done = len(alive_snakes) == 0

        if self.max_turns > 0 and self.turn >= self.max_turns:
            done = True

        info = {
            "turn": self.turn,
            "alive": [s.id for s in alive_snakes],
        }

        return self._get_observations(), rewards, done, info

    def render(self):
        """Returns an ASCII representation of the board."""
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        
        for x, y in self.hazards:
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[self.height - 1 - y][x] = "X"
                
        for f in self.food:
            x, y = f["x"], f["y"]
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[self.height - 1 - y][x] = "o"
                
        for sid, snake in self.snakes.items():
            if not snake.alive: continue
            for i, body in enumerate(snake.body):
                x, y = body["x"], body["y"]
                if 0 <= x < self.width and 0 <= y < self.height:
                    if i == 0:
                        grid[self.height - 1 - y][x] = "@" # head
                    else:
                        grid[self.height - 1 - y][x] = "#" # body

        lines = [f"Turn {self.turn}"]
        for row in grid:
            lines.append(" ".join(row))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Observation builder — matches the live Battlesnake API format
    # so StateEncoder can consume it directly
    # ------------------------------------------------------------------

    def _get_observations(self):
        obs = {}
        board_snakes = self._snake_list_dicts()
        for sid, snake in self.snakes.items():
            obs[sid] = {
                "game": {
                    "id": "sim",
                    "ruleset": {
                        "name": "standard",
                        "version": "sim",
                        "settings": {
                            "foodSpawnChance": self.food_spawn_chance,
                            "minimumFood": self.min_food,
                            "hazardDamagePerTurn": self.hazard_damage,
                            "royale": {"shrinkEveryNTurns": self.royale_shrink_every},
                        }
                    },
                    "map": "standard",
                    "timeout": 500,
                    "source": "",
                },
                "turn": self.turn,
                "board": {
                    "height": self.height,
                    "width": self.width,
                    "snakes": board_snakes,
                    "food": list(self.food),
                    "hazards": [{"x": x, "y": y} for x, y in self.hazards],
                },
                "you": self._snake_dict(snake),
            }
        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snake_dict(self, snake):
        return {
            "id": snake.id,
            "name": snake.id,
            "latency": "0",
            "health": snake.health,
            "body": [dict(p) for p in snake.body],
            "head": dict(snake.head),
            "length": snake.length,
            "shout": "",
            "squad": "",
            "customizations": {"color": "#888888", "head": "default", "tail": "default"},
        }

    def _snake_list_dicts(self):
        return [self._snake_dict(s) for s in self.snakes.values() if s.alive]

    def _random_starts(self, n):
        """Pick n distinct random positions on the board."""
        all_cells = [
            {"x": x, "y": y}
            for x in range(self.width)
            for y in range(self.height)
        ]
        random.shuffle(all_cells)
        return all_cells[:n]

    def _empty_cells(self):
        occupied = set()
        for s in self.snakes.values():
            for seg in s.body:
                occupied.add((seg["x"], seg["y"]))
        food_pos = {(f["x"], f["y"]) for f in self.food}
        return [
            {"x": x, "y": y}
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied and (x, y) not in food_pos and (x, y) not in self.hazards
        ]

    def _maybe_spawn_food(self, force_min=False):
        if force_min or len(self.food) < self.min_food:
            while len(self.food) < self.min_food:
                empties = self._empty_cells()
                if not empties:
                    break
                self.food.append(random.choice(empties))

        # Only spawn random extra food if we are below min_food
        # (This makes min_food act as a hard cap for training)
        if len(self.food) < self.min_food:
            if random.randint(1, 100) <= self.food_spawn_chance:
                empties = self._empty_cells()
                if empties:
                    self.food.append(random.choice(empties))

    def _expand_hazard_ring(self):
        """
        Flood hazard inward from the edges, one ring at a time.
        Ring 0 = edges, ring 1 = one inside, etc.
        """
        r = self._hazard_ring
        for x in range(self.width):
            for y in range(self.height):
                if x <= r or x >= self.width - 1 - r or y <= r or y >= self.height - 1 - r:
                    self.hazards.add((x, y))
        self._hazard_ring += 1
