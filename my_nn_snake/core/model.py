import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ConvLSTMCell(nn.Module):
    """
    A basic Convolutional LSTM cell.
    Preserves spatial dimensions while adding temporal memory.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)  
        
        # Conv layer output (4 * hidden_channels)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class BattleSnakeNet(nn.Module):
    def __init__(self, board_channels=16, entity_features=8, scalar_features=6):
        super(BattleSnakeNet, self).__init__()
        
        # ----------------------------------------------------
        # 1. BOARD STREAM (Spatial + Temporal)
        # ----------------------------------------------------
        self.board_cnn = nn.Sequential(
            nn.Conv2d(board_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        
        # ConvLSTM expects temporal sequence. 
        self.board_lstm = ConvLSTMCell(input_dim=64, hidden_dim=128, kernel_size=(3, 3))
        
        # Refinement after LSTM
        self.board_refinement = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        # ----------------------------------------------------
        # 2. ENTITY STREAM (Identity + Temporal)
        # ----------------------------------------------------
        # small MLP for each snake feature vector
        self.entity_mlp = nn.Sequential(
            nn.Linear(entity_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # GRU for temporal identity memory
        # batch_first=True -> (batch, seq, feature)
        self.entity_gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        
        # Cross-snake attention
        # A simple Self-Attention layer among the snakes
        self.entity_attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.entity_pool = nn.AdaptiveAvgPool1d(1) # Pool across snakes
        
        # ----------------------------------------------------
        # 3. SCALAR STREAM
        # ----------------------------------------------------
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # ----------------------------------------------------
        # 4. FUSION & HEADS
        # ----------------------------------------------------
        # Fusion input size: (Head Conv state: 128) + (Globally Pooled Conv state: 128) 
        #                    + (Entity Context: 64) + (Scalar Context: 64) = 384
        fusion_dim = 128 + 128 + 64 + 64
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Policy Head (4 actions: Up, Down, Left, Right)
        self.policy_head = nn.Linear(256, 4)
        
        # Auxiliary Heads
        self.value_head = nn.Linear(256, 1)
        self.food_urgency_head = nn.Linear(256, 1)
        self.kill_opportunity_head = nn.Linear(256, 1)
        self.territory_head = nn.Linear(256, 1)
        
    def forward(self, board_seq, entity_seq, scalar_context, head_positions):
        """
        Arguments:
        - board_seq: (Batch, SeqLen=8, Channels=16, H, W)
        - entity_seq: (Batch, SeqLen=8, NumSnakes=8, Features=8)
        - scalar_context: (Batch, Features=6)  (just for current turn)
        - head_positions: (Batch, 2) The (x, y) coordinates of our head in the final turn.
        """
        batch_size, seq_len, c, h, w = board_seq.size()
        num_snakes = entity_seq.size(2)
        device = board_seq.device
        
        # --- 1. BOARD STREAM ---
        # Initialize ConvLSTM hidden state
        h_lstm = torch.zeros(batch_size, 128, h, w, device=device)
        c_lstm = torch.zeros(batch_size, 128, h, w, device=device)
        
        for t in range(seq_len):
            board_t = board_seq[:, t, :, :, :] # (Batch, C, H, W)
            cnn_features = self.board_cnn(board_t) # (Batch, 64, H, W)
            h_lstm, c_lstm = self.board_lstm(cnn_features, (h_lstm, c_lstm))
            
        # Refine the final temporal representation
        board_refined = self.board_refinement(h_lstm) # (Batch, 128, H, W)
        
        # Extract features specifically at our head position
        # head_positions is shape (Batch, 2) holding [x, y]
        head_features = torch.zeros(batch_size, 128, device=device)
        for i in range(batch_size):
            hx = max(0, min(w - 1, int(head_positions[i, 0])))
            hy = max(0, min(h - 1, int(head_positions[i, 1])))
            head_features[i] = board_refined[i, :, hy, hx]
            
        # Extract global board summary
        global_board_features = F.adaptive_avg_pool2d(board_refined, (1, 1)).view(batch_size, -1) # (Batch, 128)
        
        # --- 2. ENTITY STREAM ---
        # Reshape to push NumSnakes into the Batch dimension to process via MLP/GRU
        entity_flat = entity_seq.view(batch_size * num_snakes, seq_len, -1) # (Batch*NumSnakes, SeqLen=8, Features=8)
        
        # Apply MLP to every frame before GRU
        # (Batch*NumSnakes, 8, 64)
        entity_mlp_out = self.entity_mlp(entity_flat) 
        
        # Pass through GRU and take final hidden state
        output, h_n = self.entity_gru(entity_mlp_out)
        # h_n shape: (1, Batch*NumSnakes, 64)
        entity_hidden = h_n.squeeze(0).view(batch_size, num_snakes, 64) # (Batch, NumSnakes, 64)
        
        # Cross-snake attention
        attn_out, _ = self.entity_attention(entity_hidden, entity_hidden, entity_hidden) # (Batch, NumSnakes, 64)
        
        # Pool across snakes to get a single entity context vector
        # Permute to (Batch, 64, NumSnakes) for adaptive pool
        entity_context = self.entity_pool(attn_out.permute(0, 2, 1)).squeeze(-1) # (Batch, 64)
        
        # --- 3. SCALAR STREAM ---
        scalar_features = self.scalar_mlp(scalar_context) # (Batch, 64)
        
        # --- 4. FUSION ---
        fused = torch.cat([head_features, global_board_features, entity_context, scalar_features], dim=1) # (Batch, 384)
        shared_repr = self.fusion_mlp(fused) # (Batch, 256)
        
        # --- 5. HEADS ---
        policy_logits = self.policy_head(shared_repr) # (Batch, 4)
        value = self.value_head(shared_repr) # (Batch, 1)
        food_urgency = self.food_urgency_head(shared_repr)
        kill_opp = self.kill_opportunity_head(shared_repr)
        territory = self.territory_head(shared_repr)
        
        return policy_logits, value, food_urgency, kill_opp, territory
