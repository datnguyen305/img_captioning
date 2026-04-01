from torch import nn
from models.modules.utils import clones
import torch 
from models.modules.ocr_layer import OCREncoderLayer

class OCREncoder(nn.Module):
    def __init__(self, config, word_embed = None):
        super().__init__()
        # head, d_model, d_kv, d_ff
        
        self.config = config
        self.word_embed = word_embed
        self.layers = clones(OCREncoderLayer(self.config.head, self.config.d_model, self.config.d_kv, self.config.d_ff), 3)
        self.norm = nn.LayerNorm(self.config.d_model)

    def forward(self, inputs, mask):
        break_probs = []
        if self.word_embed is not None:
            x = self.word_embed(inputs)
            print(x.shape)
        else:
            x = inputs
        group_prob = 0.0
        for layer in self.layers:
            x, group_prob, break_prob = layer(x, mask, group_prob)
            break_probs.append(break_prob)

        x = self.norm(x)
        break_probs = torch.stack(break_probs, dim=1)

        return x, break_probs