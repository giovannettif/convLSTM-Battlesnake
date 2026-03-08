import numpy as np
import math

class StateEncoder:
    """
    Converts a Battlesnake JSON board state into tensors for the Neural Network.
    Designed to support variable board sizes (7x7 to 25x25).
    """
    
    def __init__(self, history_length=8):
        self.history_length = history_length
        # 16 Feature Planes defined in NeuralNetworkArchitecture.md
        # Self Channels (4)
        # 0: Self Head
        # 1: Self Body
        # 2: Self Tail
        # 3: Self Body Age / Distance-to-tail map
        
        # Enemy Channels (5)
        # 4: All enemy heads
        # 5: All enemy bodies
        # 6: All enemy tails
        # 7: Larger-or-equal enemy heads (lethal to us)
        # 8: Smaller enemy heads (we can kill them)
        
        # Objective Channels (3)
        # 9: Food
        # 10: Hazards
        # 11: Contested squares / Predicted enemy next-step danger
        
        # Global-state / Broadcast Planes (4)
        # 12: Self health map (normalized 0-1 across entire board)
        # 13: Self length map 
        # 14: Longest enemy length map
        # 15: Turn-progress / hunger-pressure map
        
        self.num_channels = 16
        
    def encode_board(self, data):
        """
        Converts a single turn's JSON data into a 3D numpy array (Channels, Height, Width).
        """
        board = data["board"]
        width = board["width"]
        height = board["height"]
        you = data["you"]
        my_length = you["length"]
        
        # Initialize empty tensor for this frame
        tensor = np.zeros((self.num_channels, height, width), dtype=np.float32)
        
        # --- SELF CHANNELS ---
        # 0: Our head
        hy, hx = you["head"]["y"], you["head"]["x"]
        if 0 <= hx < width and 0 <= hy < height:
            tensor[0, hy, hx] = 1.0
        
        # 1: Our body & 3: Body Age Map
        body_parts = you["body"]
        for i in range(1, len(body_parts) - 1): # Exclude head and tail
            part = body_parts[i]
            px, py = part["x"], part["y"]
            if 0 <= px < width and 0 <= py < height:
                tensor[1, py, px] = 1.0
                dist_to_tail = len(body_parts) - 1 - i
                tensor[3, py, px] = max(0.01, 1.0 - (dist_to_tail / max(1, my_length)))
            
        # 2: Our tail
        if len(body_parts) > 1:
            tail = body_parts[-1]
            tx, ty = tail["x"], tail["y"]
            if 0 <= tx < width and 0 <= ty < height:
                tensor[2, ty, tx] = 1.0
        
        # --- ENEMY CHANNELS ---
        longest_enemy_len = 0
        for snake in board["snakes"]:
            if snake["id"] == you["id"]:
                continue
                
            enemy_len = snake["length"]
            longest_enemy_len = max(longest_enemy_len, enemy_len)
            
            # 4: All enemy heads
            head_x, head_y = snake["head"]["x"], snake["head"]["y"]
            if 0 <= head_x < width and 0 <= head_y < height:
                tensor[4, head_y, head_x] = 1.0
                
                # 7 & 8: Relative head size logic
                if enemy_len >= my_length:
                    tensor[7, head_y, head_x] = 1.0 # Lethal
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = head_x + dx, head_y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            tensor[11, ny, nx] = 1.0
                else:
                    tensor[8, head_y, head_x] = 1.0 # Vulnerable
            
            # 5: All enemy bodies
            enemy_body = snake["body"]
            for i in range(1, len(enemy_body) - 1):
                part = enemy_body[i]
                ex, ey = part["x"], part["y"]
                if 0 <= ex < width and 0 <= ey < height:
                    tensor[5, ey, ex] = 1.0
                
            # 6: All enemy tails
            if len(enemy_body) > 1:
                enemy_tail = enemy_body[-1]
                etx, ety = enemy_tail["x"], enemy_tail["y"]
                if 0 <= etx < width and 0 <= ety < height:
                    tensor[6, ety, etx] = 1.0
            
        # --- OBJECTIVE CHANNELS ---
        # 9: Food
        for food in board["food"]:
            tensor[9, food["y"], food["x"]] = 1.0
            
        # 10: Hazards
        for hazard in board.get("hazards", []):
            tensor[10, hazard["y"], hazard["x"]] = 1.0
            
        # --- GLOBAL BROADCAST PLANES ---
        # Instead of single values, we flood the entire plane with the scalar.
        # This allows the CNN to consider this context purely spatially.
        
        # 12: Self health
        health_val = you["health"] / 100.0
        tensor[12, :, :] = health_val
        
        # 13: Self length map (normalized to an arbitrary long length, say 30)
        len_val = min(1.0, my_length / 30.0)
        tensor[13, :, :] = len_val
        
        # 14: Longest enemy length
        enemy_len_val = min(1.0, longest_enemy_len / 30.0)
        tensor[14, :, :] = enemy_len_val
        
        # 15: Turn progress
        turn_val = min(1.0, data["turn"] / 1000.0) # 1000 is the expected royale shrink start
        tensor[15, :, :] = turn_val
            
        return tensor

    def encode_entity_stream(self, data):
        """
        Extracts per-snake features for the Entity Stream (GRU input).
        Returns a list of feature vectors, one for each snake.
        Features: [is_us, rel_health, rel_length, dist_to_our_head]
        """
        you = data["you"]
        my_length = you["length"]
        my_head_x, my_head_y = you["head"]["x"], you["head"]["y"]
        
        entities = []
        for snake in data["board"]["snakes"]:
            is_us = 1.0 if snake["id"] == you["id"] else 0.0
            rel_health = snake["health"] / 100.0
            rel_length = min(3.0, snake["length"] / max(1, my_length)) # Ratio capped at 3x our size
            
            # Manhattan distance to our head
            dist_x = abs(snake["head"]["x"] - my_head_x)
            dist_y = abs(snake["head"]["y"] - my_head_y)
            # Normalize by rough max board size (25)
            rel_dist = min(1.0, (dist_x + dist_y) / 25.0) 
            
            # Turn orientation logic (which way are they facing?)
            # 0=Up, 1=Right, 2=Down, 3=Left (Requires body analysis)
            dir_up, dir_rt, dir_dn, dir_lt = 0.0, 0.0, 0.0, 0.0
            if len(snake["body"]) > 1:
                hx, hy = snake["head"]["x"], snake["head"]["y"]
                nx, ny = snake["body"][1]["x"], snake["body"][1]["y"]
                if hy > ny: dir_up = 1.0
                elif hy < ny: dir_dn = 1.0
                elif hx > nx: dir_rt = 1.0
                elif hx < nx: dir_lt = 1.0
                
            feature_vec = [is_us, rel_health, rel_length, rel_dist, dir_up, dir_rt, dir_dn, dir_lt]
            entities.append(np.array(feature_vec, dtype=np.float32))
            
        # Pad or slice to a fixed number of max snakes (e.g. 8)
        max_snakes = 8
        num_features = len(entities[0]) if entities else 8
        padded_entities = np.zeros((max_snakes, num_features), dtype=np.float32)
        
        for i in range(min(max_snakes, len(entities))):
            padded_entities[i] = entities[i]
            
        return padded_entities

    def get_scalar_context(self, data):
        """
        Extracts global scalar features for the final Fusion MLP.
        """
        you = data["you"]
        board = data["board"]
        
        health = you["health"] / 100.0
        length = min(1.0, you["length"] / 30.0)
        turn = min(1.0, data["turn"] / 1000.0)
        snakes_alive = min(1.0, len(board["snakes"]) / 8.0)
        
        longest_enemy = 0
        for snake in board["snakes"]:
            if snake["id"] != you["id"]:
                longest_enemy = max(longest_enemy, snake["length"])
                
        enemy_len = min(1.0, longest_enemy / 30.0)
        
        # Includes ruleset flag proxy (if shrinkEveryNTurns is present, it's Royale)
        is_royale = 1.0 if "royale" in data["game"]["ruleset"]["settings"] else 0.0
                
        return np.array([health, length, enemy_len, snakes_alive, turn, is_royale], dtype=np.float32)
