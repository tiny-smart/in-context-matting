from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
import numpy as np
import abc


LOW_RESOURCE = False 


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, ensemble_size=1, token_batch_size=1):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet, ensemble_size, token_batch_size)
            else:
                h = attn.shape[0]
                # attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
                attn = self.forward(attn, is_cross, place_in_unet, ensemble_size, token_batch_size)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, ensemble_size=1, token_batch_size=1):
        num_head = attn.shape[0]//token_batch_size
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if self.store_res is not None:
            if attn.shape[1] in self.store_res and (is_cross is False):
                attn = attn.reshape(-1, ensemble_size, *attn.shape[1:])
                attn = attn.mean(dim=1)
                attn = attn.reshape(-1,num_head , *attn.shape[1:])
                attn = attn.mean(dim=1)
                self.step_store[key].append(attn)
        elif attn.shape[1] <= 48 ** 2 and (is_cross is False):  # avoid memory overhead
            attn = attn.reshape(-1, ensemble_size, *attn.shape[1:])
            attn = attn.mean(dim=1)
            attn = attn.reshape(-1,num_head , *attn.shape[1:])
            attn = attn.mean(dim=1)
            self.step_store[key].append(attn)

        torch.cuda.empty_cache()

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        del self.step_store
        torch.cuda.empty_cache()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self,store_res = None):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        store_res = [store_res] if isinstance(store_res, int) else list(store_res) 
        self.store_res = []
        for res in store_res:
            self.store_res.append(res**2)
        
