from primera.utils.pretrain_preprocess import *
from torch.utils.data import DataLoader, Dataset, IterableDataset
from pathlib import Path
import torch
from random import shuffle
import random
import os
from nltk.tokenize import sent_tokenize
import re
import sys
class PretrainDataset(Dataset):
    def __init__(self,data,tokenizer,max_input_len=4096,max_output_len=512,non_mask_ratio=0.5):
        self.data=data
        self.max_input_len=max_input_len
        self.max_output_len=max_output_len
        self.tokenizer=tokenizer
        self.non_mask_ratio=non_mask_ratio
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,index):
        row=self.data.loc[index]
        data=get_src_tgt_and_mask(row["truncated_docs"],row["selected_sents"],self.tokenizer,self.max_input_len,
                          self.max_output_len,self.non_mask_ratio)
        return data