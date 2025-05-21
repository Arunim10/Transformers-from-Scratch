#tWILIO 2YR526RMCUCWMBCJALTCLECP

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BillingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path,get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel,BPE
from tokenizers.trainers import WordLevelTrainer,BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras

from torch.utils.tensorboard.writer import SummaryWriter

# import warnings
from tqdm import tqdm
 
from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
## The key difference between yield and return is that yield allows the function to return a value while saving its state so that it can be resumed from where it left off later. This makes it possible to generate a sequence of values over time instead of computing all of them at once.


def get_or_build_tokenizer(config, ds, lang): ## ds->dataset, lang->language
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        unk_token = "[UNK]"
        special_tokens = ["[UNK]","[PAD]","[SOS]","[EOS]"]
        tokenizer = Tokenizer(BPE(unk_token=unk_token)) ## unk_token will replace the those unknown words, which are not in their vocabulary, with 'UNK'
        tokenizer.pre_tokenizer.PreTokenizer = Whitespace() ## this means we split by whitespace
        trainer = BpeTrainer(special_tokens = special_tokens, min_frequency = 2) ## this will train the tokenizer and the tokenized form will also have special tokens like UNK, PAD, SOS, EOS and min_freq = 2 means for a word to be included in the vocabulary , it should appear atleast 2 times in the text.
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')
    """
      def load_dataset(dataset_name, lang_pair, split='train'):
        # # Implementation to load the dataset
        # # The actual code would go here, handling the loading and processing of the dataset
        # # It might involve downloading data, reading files, preprocessing, etc.
        # pass  # Placeholder for the actual implementation
      The parameters of the load_dataset function are as follows:

      dataset_name: A string representing the name of the dataset to load. In this case, it is 'opus_books'.

      lang_pair: A string representing the language pair to be loaded. The format of this string is likely 'source_language-target_language'. The specific source and target languages will be determined by the values stored in the config dictionary, accessed through config["lang_src"] and config["lang_tgt"].

      split: A keyword argument that specifies which split of the dataset to load. The default value is 'train', indicating the training split. Depending on the dataset, other possible values could be 'test', 'validation', or similar.

    """
    ## Build tokenizer
    tokenizer_src = get_or_build_tokenizer(config,ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw, config['lang_tgt'])

    ## train -> 90% , test -> 10%
    train_ds_size = int(0.9 * len(ds_raw)) 
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds = BillingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BillingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src,len(src_ids))
        max_len_tgt = max(max_len_tgt,len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1,shuffle=False) ## batch size = 1 bcoz I want to process each sentence one by one
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'],config['d_model'])
    return model

def train_model(config):
    ## define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    ## Tensorboard to visualize loss charts 
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader,desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            
            encoder_input = batch['encoder_input'].to(device) ## (batch,seq_len)
            decoder_input = batch['decoder_input'].to(device) ## (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) ## (b,1,1,seq_len)
            decoder_mask = batch['decoder_mask'].to(device) ## (b,1,1,seq_len) ## for hiding pad tokens
            decoder_mask = batch['decoder_mask'].to(device) ## (b,1,seq_len,seq_len) ## for hiding pad tokens and also future tokens of each word

            ## Runs tensors through transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask,decoder_input,decoder_mask) ## (b,seq_len,d_model)
            proj_output = model.project(decoder_output) ## (b,seq_len,tgt_vocab_size)

            label = batch['label'].to(device) ## (b,seq_len)

            ## (b, seq_len, tgt_vocab_size) --> (b*seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size_), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            ## log the loss
            writer.add_scalar('train loss', loss.item(),global_step)

            ## backpropagate the loss
            loss.backward()

            ## update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1 ## used by tensorboard to keep track of the loss

        # save the model at the end of every epoch
        model_filename = get_weights_file_path(config,f"{epoch:02d}")
        torch.save({
             'epoch': epoch,
             'model_state_dict':model.state_dict(),
             'global_step': global_step,
             'optimizer_state_dict': optimizer.state_dict()
        },model_filename)

if __name__ == "main":
    # warnings.filterwarnings('ignore')
    try:
    # Your PyTorch code
        print("Config \n")
        config = get_config()
        print("Training \n")
        train_model(config)
    except Exception as e:
        print(f"Error: {e}")
    
                    