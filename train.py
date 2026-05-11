from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functinal as F

class Block(nn.)









@dataclass
#model hyperparameters
class GPTConfig:
    block_size: int  = 256
    vocab_size:int = 65
    n_layer:int =6
    n_head:int =6
    n_embd:int =384

class GPT(nn.Module):
     def __init__(self, config):
        super().__init__()    
        self.config = config


        self.transformer = nn.ModuleDict(dict(

            wte = nn.Embedding(config.vocab_size, config.n_embd),# weights of token emb
            wpe =nn.Embedding(config.block_size, config.n_embd),# weights of positional emb
            h =nn.ModuleList([Block(config) for _ in range(config.n_layer)]),# hidden layer
            ln_f = nn.LayerNorm(config.n_embd), #lang modeling head, 
                                               # conv final vector size 384 -> scores over 65         
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)