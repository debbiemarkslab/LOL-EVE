import torch
from torch import nn

class AdaptiveLocalPositionEmbedding(nn.Module):
    def __init__(self, d_model, sequence_start=4, max_len=1007):
        super().__init__()
        self.control_embeddings = nn.Embedding(sequence_start, d_model)
        self.sequence_embeddings = nn.Embedding(max_len - sequence_start, d_model)
        self.sequence_start = sequence_start
        
    def forward(self, x, input_ids, tokenizer):
        batch_size, seq_len, _ = x.size()
        position_embeddings = torch.zeros_like(x)
        
        # Apply control token embeddings (absolute positions)
        for i in range(min(self.sequence_start, seq_len)):
            position_embeddings[:, i] = self.control_embeddings(torch.tensor(i, device=x.device))
        
        # Find sequence boundaries using start token
        start_token = tokenizer.convert_tokens_to_ids('start')
        start_positions = (input_ids == start_token).nonzero(as_tuple=True)
        
        # Apply sequence token embeddings (relative positions)
        for idx, batch_idx in enumerate(start_positions[0]):
            pos = start_positions[1][idx].item()
            
            # Skip if start token is after sequence_start
            if pos < self.sequence_start:
                continue
                
            # For each position after start token, use relative position
            for i in range(pos, seq_len):
                rel_pos = i - pos  # Relative position
                if rel_pos < self.sequence_embeddings.num_embeddings:
                    position_embeddings[batch_idx, i] = self.sequence_embeddings(torch.tensor(rel_pos, device=x.device))
        
        return x + position_embeddings

class RoPEPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding as described in https://arxiv.org/abs/2104.09864
    Adapted for the LOL-EVE architecture
    """
    def __init__(self, d_model, sequence_start=4, max_len=1007, base=10000.0):
        super().__init__()
        self.sequence_start = sequence_start
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.control_embeddings = nn.Embedding(sequence_start, d_model)
        
        # Create register buffer for faster access
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model)),
        )
    
    def _get_rotary_embeddings(self, seq_len):
        """Generate rotary embeddings for the given sequence length"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Create cos and sin embeddings
        cos_pos = emb.cos()[None, :, None, :]
        sin_pos = emb.sin()[None, :, None, :]
        
        return cos_pos, sin_pos
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _apply_rotary_embeddings(self, x, cos, sin):
        """Apply rotary embeddings to input tensor x"""
        # Reshape for multi-head attention if needed
        seq_len = x.shape[1]
        cos = cos[:, :seq_len]  # [1, seq_len, 1, dim]
        sin = sin[:, :seq_len]  # [1, seq_len, 1, dim]
        
        # Reshape x if needed (depends on whether this is used in self-attention)
        # For embedding layer, shape is [batch, seq_len, d_model]
        # For multi-head attention with heads=1, shape is [batch, seq_len, 1, d_model]
        if len(x.shape) == 3:
            x = x.unsqueeze(2)
            apply_rotary = True
            
            # Apply rotary embeddings
            x_rope = (x * cos) + (self._rotate_half(x) * sin)
            
            # Reshape back
            return x_rope.squeeze(2)
        else:
            # For already reshaped inputs (e.g., in attention layers)
            return (x * cos) + (self._rotate_half(x) * sin)
    
    def forward(self, x, input_ids, tokenizer):
        """
        Apply position embeddings to input tensors
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            input_ids: Input token IDs
            tokenizer: Tokenizer instance
            
        Returns:
            Tensor with position embeddings applied
        """
        batch_size, seq_len, _ = x.size()
        position_embeddings = torch.zeros_like(x)
        
        # Apply regular embeddings for control tokens
        for i in range(min(self.sequence_start, seq_len)):
            position_embeddings[:, i] = self.control_embeddings(torch.tensor(i, device=x.device))
        
        # Get rotary embeddings
        cos_pos, sin_pos = self._get_rotary_embeddings(seq_len - self.sequence_start)
        
        # Apply rotary embeddings for sequence tokens
        sequence_embeddings = x[:, self.sequence_start:]
        sequence_embeddings = self._apply_rotary_embeddings(sequence_embeddings, cos_pos, sin_pos)
        
        # Combine control and sequence embeddings
        final_embeddings = x.clone()
        final_embeddings[:, :self.sequence_start] = x[:, :self.sequence_start] + position_embeddings[:, :self.sequence_start]
        final_embeddings[:, self.sequence_start:] = sequence_embeddings
        
        return final_embeddings