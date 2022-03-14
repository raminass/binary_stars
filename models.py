import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from torch import nn, Tensor

################################################################################
############################## MLP Models ######################################
################################################################################

############### Initial MLP Model - Optuna #####################################
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs, n_output, trial):
        super(MLP, self).__init__()
        l1 = trial.suggest_int("l1", 128, 512)
        l2 = trial.suggest_int("l2", 32, 128)
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, l1)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(l1, l2)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(l2, n_output)
        nn.init.xavier_uniform_(self.hidden3.weight)
        # self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        # X = self.act3(X)
        return X

############### Final MLP Model - using layer sizes from Optuna ################
class MLP_final(nn.Module):
    # define model elements
    def __init__(self, n_inputs, n_output):
        super().__init__()
        l1 = 400
        l2 = 100
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, l1)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(l1, l2)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(l2, n_output)
        nn.init.xavier_uniform_(self.hidden3.weight)
        # self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        # X = self.act3(X)
        return X

################################################################################
################  Transformer-based Model ######################################
################################################################################

class xformer(nn.Module):

    # Constructor
    def __init__(
        self,
        target_labels,
        input_dim=10197,
        dim_model=512,
        pool=10,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout_p=0.1,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        '''self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )'''

        # Pool & prep src for transformer
        self.pool = nn.MaxPool1d(pool, stride=pool)
        self.pool_out = (input_dim-(pool-1)-1)//pool+1
        self.src_in = nn.Linear(self.pool_out, dim_model)

        # Embed 
        #self.embedding = nn.Embedding(target_labels, dim_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )

        # Output layer (to take on transformer dimensions)
        self.target_in = nn.Linear(target_labels, self.dim_model)
        self.out = nn.Linear(self.dim_model, target_labels)

    def forward(
        self,
        src,
        tgt,
    ):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        #src = self.embedding(src) * math.sqrt(self.dim_model)
        #tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        #src = self.positional_encoder(src)
        #tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        #src = src.permute(1, 0, 2)
        #tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        src = self.pool(src)
        src = self.src_in(src)
        src = torch.unsqueeze(src, 0)
        tgt = self.target_in(tgt)
        tgt=torch.unsqueeze(tgt, 0)
        transformer_out = self.transformer(src, tgt)
        out = self.out(transformer_out)

        return out.squeeze()

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, n_output: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, d_hid)
        self.flat = torch.nn.Flatten(1)
        self.init_weights()

        # Last hidden layer and output
        self.hidden3 = nn.Linear(50*d_hid, n_output)
        nn.init.xavier_uniform_(self.hidden3.weight)


    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.d_model)
        batch_size = src.shape[0]
        src = torch.reshape(src[:,:10000].T, (50,batch_size,200))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, None)
        output = self.decoder(output)
        output = self.flat(output.transpose(0,1))
        output =  self.hidden3(output)
        return output


################################################################################
#################### Attention-based Model #####################################
################################################################################

######################### Simple 1D attention layer ############################
class Attention1D(nn.Module):

    # Constructor
    def __init__(
        self,
        input_dim=200,
        dim_model=256,
        dim_out= 100,
    ):
        super().__init__()

        # LAYERS
        # W_k (Dx X Dq)
        self.key_linear = nn.Linear(input_dim, dim_model)
        # W_q (Dx X Dq)
        self.query_linear = nn.Linear(input_dim, dim_model)
        # W_v (Dx X Dv)
        self.value_linear = nn.Linear(input_dim, dim_out)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        # Create key, query, and value
        K = self.key_linear(X)
        Q = self.query_linear(X)
        V = self.value_linear(X)

        # Scale factor
        D_q = K.shape[1]

        # Matmuls
        E = torch.matmul(K, Q.t())/math.sqrt(D_q)
        A = self.softmax(E)
        Y = torch.matmul(A, V)
        return Y

############### Multihead attention - 4 heads ##################################
## Combines four Attention1D layers in to one
class Fourhead_attention1D(nn.Module):
    # Constructor
    def __init__(
        self,
        input_dim=200,
        dim_model=256,
        dim_out= 100,
    ):
        super().__init__()

        # Create layer for each head

        self.head1=Attention1D(input_dim, dim_model=dim_model, dim_out= dim_model)
        self.head2=Attention1D(input_dim, dim_model=dim_model, dim_out= dim_model)
        self.head3=Attention1D(input_dim, dim_model=dim_model, dim_out= dim_model)
        self.head4=Attention1D(input_dim, dim_model=dim_model, dim_out= dim_model)

    def forward(self, X):
        
        a = self.head1(X)
        b = self.head2(X)
        c = self.head3(X)
        d = self.head4(X)
        out = torch.hstack((a, b, c, d))

        return out

############### Multihead attention block - uses 4-heads, FC layer #############
## Uses FC layer to return output shape of fourhead attention to same as initial input
## Also adds residual connection to create the block
class MHA_block(nn.Module):

    # Constructor
    def __init__(
        self,
        dim = 128,
        num_heads=4,
    ):
        super().__init__()

        # INFO
        # Create layer for each head
        self.mhattention = Fourhead_attention1D(input_dim=dim, dim_model=dim, dim_out= dim)
        
        self.linear = nn.Linear(dim*num_heads, dim)

    def forward(self, X):
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        
        out = self.mhattention(X)
        out = F.relu(self.linear(out))
        out = out+X
      
        return out

#################### model_x 7 MHA Block layers followed by FC layers ##########
class AttentionBlockModel(nn.Module):

    # Constructor
    def __init__(
        self,
        target_labels,
        input_dim=10197,
        dim_model=512,
        num_heads=8,
        dropout_p=0.2,
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout(p=dropout_p)

        self.linear1 = nn.Linear(input_dim, 500)
        self.linear2 = nn.Linear(500,128)

        self.MHA1 = MHA_block(dim=128, num_heads=4)
        self.MHA2 = MHA_block(dim=128, num_heads=4)
        self.MHA3 = MHA_block(dim=128, num_heads=4)
        self.MHA4 = MHA_block(dim=128, num_heads=4)
        self.MHA5 = MHA_block(dim=128, num_heads=4)
        self.MHA6 = MHA_block(dim=128, num_heads=4)
        self.MHA7 = MHA_block(dim=128, num_heads=4)

        self.linear3 = nn.Linear(128,50)
        self.linear4 = nn.Linear(50, target_labels)


    def forward(self, X):

        #Start with 2 FC layers, with dropout
        X = F.relu(self.linear1(X))
        X = self.dropout(X)
        X = F.relu(self.linear2(X))
        X = self.dropout(X)

        # Seven MHA Blocks
        X = self.MHA1(X)
        X = self.MHA2(X)
        X = self.MHA3(X)
        X = self.MHA4(X)
        X = self.MHA5(X)
        X = self.MHA6(X)
        X = self.MHA7(X)

        # Two FC layers at end before output
        out = F.relu(self.linear3(X))
        out = self.linear4(out)

        return out

################################################################################
###################### 1D Convolutional Models #################################
################################################################################

# model definition
class ConvolutionalNet(nn.Module):

    # define model elements
    def __init__(self, n_params):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 16, 5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 5, stride=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 5, stride=5, padding=2)
        self.conv4= nn.Conv1d(64, 64, 5, stride=5, padding=2)
        self.conv_final= nn.Conv1d(64, 5, 1, stride=1, padding=0)

        # Linear layers
        self.linear1 = nn.Linear(410,100)
        self.linear2 = nn.Linear(100, n_params)
        
    # forward propagate input
    def forward(self, X):

        # unsqueeze to fit dimension for nn.Conv1d
        X = torch.unsqueeze(X, 1)

        # Four convolutional layers
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))

        # 1D convolution, then flatten for FC layers
        X = F.relu(self.conv_final(X))
        X = torch.flatten(X, start_dim=1)

        # Two FC layers to give output
        X = F.relu(self.linear1(X))
        out = self.linear2(X)

        return out