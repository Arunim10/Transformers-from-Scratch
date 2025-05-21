import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset, Dataset

class BillingualDataset(Dataset):

    def __add__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        ## Convert SOS token into a input id
        self.sos_token = torch.tensor([tokenizer_src.token_to_id(['[SOS]'])],dtype=torch.int64) ## dtype is long bcoz vocab can be greater than 32 bit
        self.eos_token = torch.tensor([tokenizer_src.token_to_id(['[EOS]'])],dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id(['[PAD]'])],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        ## splitting sentences into words and then into numbers and then finally getting ids of that in the vocabulary
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids ## array of ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        ## Padding
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 ## -2 is added to include <SOS> and <EOS> tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 ## here in the decoder side, we only include the <SOS> token

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        

        ## Add SOS and EOS to src_text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens,dtype=torch.int64)
            ]
        )

        ## add sos to dec input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens,dtype=torch.int64)
            ]
        )

        ## add eos to label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens,dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input, ## (seq_len)
            'decoder_input': decoder_input, ## (seq_len)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), ## (1,1,seq_len), here we dont want the attention to focus on pad tokens 
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), ## (1,1,seq_len) & (1,seq_len,seq_len)
            'label': label, ## (seq_len)
            'src_text': src_text, ## for visualization
            'tgt_text': tgt_text  ## visualization
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal = 1).type(torch.int64)
    return mask==0