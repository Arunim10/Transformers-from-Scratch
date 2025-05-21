import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self,d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) ## in the paper, its written we multiply the embedding layer weights by sqrt(d_model) so we are doing that here
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        ## Create a matrix of shape (seq_len,d_model) to store PE
        self.pe = torch.zeros(seq_len,d_model)

        ## Create a vector of shape (seq_len,1) representing position of words inside a sentence
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) ## (seq_len,1)

        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))  ## the divisible term is same but we have used log to get numerical stability

        ## Apply the sin to even position
        self.pe[:,0::2] = torch.sin(position * div_term) ## pe[:,0;:2] means all rows and columns are 0,2,4,6,8,10

        ## Apply cos to odd position
        self.pe[:,1::2] = torch.cos(position * div_term) ## ## pe[:,1;:2] means all rows and columns are 1,3,5,7,9

        self.pe = self.pe.unsqueeze(0) ## (1,seq_len,d_model) bcoz we will input batch of sequences here, so unsqueezed in 0 position

        self.register_buffer('pe',self.pe) ## its used to save the pe in the file and not as some parameter
        ## Buffers are tensors that should be included in the state of the module but should not be optimized (updated) during training, unlike model parameters. Buffers are typically used to store running statistics, fixed weights, or any other persistent data that is required for the functioning of the model.

    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False) ## this will make positional encoding to not be learned
        return self.dropout(x) 
    ## During forward propagation, nn.Dropout() randomly sets a fraction of the input elements to zero with a probability p. 
     
class LayerNormalization(nn.Module):

    def __init__(self,eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) ## only one element "1" in alpha  ## this is multiplied
        self.bias = nn.Parameter(torch.zeros(1))  ## only one element "0" in bias ## this is added
        ## Parameter means it keeps the variables learnable
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True) # keepdim will keep the dimensions of input tensors for ex mean of x with shape(2,3) along dim=1 with keepdim=True will have resulting tensor with shape (2,1)
        std = x.std(dim=-1,keepdim=True)
        return self.alpha * (x-mean) / (std+self.eps) + self.bias 
    
class FeedForwardBlock(nn.Module):

    def __init__(self,d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        # (Batch,seq_len,d_model) --> (Batch,seq_len,d_ff) --> (Batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model: int,h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model,d_model) ## wq
        self.w_k = nn.Linear(d_model,d_model) ## wk
        self.w_v = nn.Linear(d_model,d_model) ## wv

        self.w_o = nn.Linear(d_model,d_model) ##  shape of w_o => (h * dv,d_model) == (h*dk,d_model) == (d_model,d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod ## this function is static means we can  directly call this function using MultiHeadAttentionBlock.attention without creating any instance of this class
    def attention(query,key,value,mask,dropout: nn.Dropout):
        d_k = query.shape[-1] 

        ## (batch,h,seq_len,d_k) --> (batch,h,seq_len,seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / (math.sqrt(d_k))
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
            ## In PyTorch, the masked_fill_ function is used to replace elements in a tensor based on a given mask. This function modifies the tensor in-place. It fills the elements in the original tensor with a specified value where the corresponding mask is True, and leaves the elements unchanged where the mask is False.

        attention_scores = attention_scores.softmax(dim=-1) ## (batch,h,seq_len,seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value) , attention_scores ## (final attention scores which is the output of attention => (batch,h,seq_len,d_k), attention_scores are returned for visualization purposes => (batch,h,seq_len,seq_len))

    def forward(self, q, k, v, mask):
        query = self.w_q(q) ## (Batch,seq_len,d_model) --> (batch,seq_len,d_model)
        key = self.w_k(k) ## (Batch,seq_len,d_model) --> (batch,seq_len,d_model)
        value = self.w_v(v) ## (Batch,seq_len,d_model) --> (batch,seq_len,d_model)

        ## (batch,seq_len,d_model) --> (batch,seq_len,h,d_k) --transpose--> (batch,h,seq_len,d_k) => we want each head to see (seq_len,d_k) this matrix. 
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        ## x shape => (batch,h,seq_len,d_k) --transpose--> (batch,seq_len,h,d_k) --> (batch,seq_len,h*d_k)==(batch,seq_len,d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)
        ## Tensor operations in PyTorch sometimes result in non-contiguous tensors. For example, slicing, transposing, or certain operations involving strides can lead to non-contiguous memory layout. In such cases, it can be beneficial to use the contiguous function to make sure the tensor's memory is organized in a contiguous manner, as some operations require contiguous tensors for efficient computation.

        return self.w_o(x)
    
## DOUBT: THIS CLASS IS NOT UNDERSTOOD, UNDERSTAND IT WHEN IMPLEMENTION IS DONE 
class ResidualConnection(nn.Module):

    def __init__(self,dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer): ## sublayer is the previous layer
        return x + self.dropout(sublayer(self.norm(x))) ## in actual paper, first sublayer is applied to x and then its normalized but in many implementations of the paper, the video author saw this implementation so we will stick with this implementation


class EncoderBlock(nn.Module):

    def __init__(self,self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    ## DOUBT : WHY LAMBDA FUNCTION IS USED HERE AND WE ARE GIVING 4 INPUTS (X,X,X,SRC_MASK) HERE BUT IN THE RESIDUAL_CONNECTIONS FORWARD FUNCTION , SUBLAYER ACCEPTS ONLY ONE INPUT I.E.. NORM(X) , SO HOWS THIS CODE WORKING HERE ? I AM NOT GETTING HOW THE NORMALIZATION IS APPLIED IN THE OUTPUT OF MULTI-HEAD-ATTENTION BLOCK ?
    def forward(self,x,src_mask): ## this src_mask includes those interactions which should be neglected by the block and this also includes interactions of paddings with the sentence words
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x,self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self,layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask) ## the output of one encoder is the input of next encoder
        return self.norm(x) ## finally after normalising the output of final encoder , we return the final output of encoder
    

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask): ## src_mask is for encoder where mask is applied for english(input) language and tgt_mask is for decoder where mask is applied for spanish(target) language
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask)) ## query from decoder and keys , values from encoder
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        ## (batch,seq_len,d_model) --> (batch,seq_len,vocab_size)
        return torch.log_softmax(self.proj(x),dim=-1) ## log_softmax for numerical stability

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        print("Inside Decoder")
        print(encoder_output.shape," ",src_mask.shape," ",tgt.shape," ",tgt_mask.shape)
        tgt = self.tgt_embed(tgt)
        print(tgt.shape)
        tgt = self.tgt_pos(tgt)
        print(tgt.shape)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)
    
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 1, h: int = 1, dropout: float = 0.1, d_ff:int = 512) -> Transformer:  ## src_seq_len and tgt_seq_len in our case will be same , but they can be different also
    ## Create embedding layers
    src_embed = InputEmbeddings(d_model,src_vocab_size)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)

    ## Create positional encoding layers
    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_len,dropout)

    ## Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    ## Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,h,dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    ## Create Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    ## Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    ## Create the Transformer
    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    ## Initialize the parameters using xavier_uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  ## its used for faster training

    return transformer

# transformer = build_transformer(10,10,20,20)
# for p in transformer.parameters():
#     print(p.shape)