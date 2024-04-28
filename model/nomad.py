import torch
from torch import nn
import torch.nn.functional as F
import timm
import math

class PositionalEncoding(nn.Module):
    '''
    references: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class VisionEncoder(nn.Module):
    def __init__(self, d_model=256):
        super(VisionEncoder, self).__init__()

        self.obs_encoder = timm.create_model('efficientnet_b0', pretrained=False, num_classes=d_model)
        self.goal_encoder = timm.create_model('efficientnet_b0', pretrained=False, num_classes=d_model)
        self.goal_encoder.conv_stem=nn.Conv2d(6, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len = 7)

        #구현체 그대로 gelu를 사용하고 dimension은 NoMAD논문을 따랐습니다.
        self.mhsa_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                     nhead=4,
                                                     dim_feedforward=256*4,
                                                     activation='gelu',
                                                     batch_first=True,
                                                     norm_first=True)

        self.transformer_encoder = nn.TransformerEncoder(self.mhsa_layer, num_layers=4)

        goal_mask = torch.tensor([[0.,0,0,0,0,0,0],
                                 [0,0,0,0,0,0,1]])
        self.register_buffer('goal_mask', goal_mask)

        goal_normalize = torch.tensor([[1.,1,1,1,1,1,1],
                                       [1.16,1.16,1.16,1.16,1.16,1.16,0]]).T
        self.register_buffer('goal_normalize',goal_normalize)




    def forward(self,obs_imgs, goal_img, goal_mask):
        '''
        obs_imgs = batchsize, time, channel, 96, 96
        goal_img = batchsize,channel,96,96
        obs_imgs flow = x
        goal_img flow = y
        context vector flow = ct
        '''
        goal_mask_arr = torch.index_select(self.goal_mask, 0, goal_mask).permute(1,0)
        goal_normalize = torch.index_select(self.goal_normalize, 1, goal_mask).unsqueeze(0).permute(2,1,0).permute(1,0,2)

        #ViNT 논문에 따라 early fusion을 진행합니다.
        latest_img = obs_imgs[:,-1,:,:]
        fusion_img = torch.cat((goal_img, latest_img), dim=1)
        y = self.goal_encoder(fusion_img)#batch, dim

        obs_imgs = obs_imgs.view(-1,3,96,96)
        x = self.obs_encoder(obs_imgs)  #batch*time, 256
        x = torch.split(x,6,dim=0) #batch,time,256
        x = torch.stack(x)

        y = y.unsqueeze(1)
        xy = torch.cat((x,y),dim=1)
        xy = xy.permute(1,0,2)
        ct = self.positional_encoding(xy)


        ct = self.transformer_encoder(xy,src_key_padding_mask=goal_mask_arr)
        ct = ct*goal_normalize

        ct = torch.mean(ct,dim=0)

        return ct


class DistPredNet(nn.Module):
    def __init__(self, d_model=256):
        super(DistPredNet, self).__init__()
        self.d_model = d_model
        self.network = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//4),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.d_model//4, self.d_model//16),
            nn.ReLU(),
            nn.Linear(self.d_model//16, 1,),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.network(x)
        return x
    

class NoisePredNet(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model =d_model
        self.positional_encoding = PositionalEncoding(d_model=self.d_model, max_len = 2)
        self.time_embedding = nn.Embedding(10,self.d_model)

        self.noise_embedding  = nn.Sequential(nn.Flatten(1,-1),
                                              nn.Linear(16,self.d_model),
                                              nn.GELU())

        self.mhsa_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                     nhead=4,
                                                     dim_feedforward=256*4,
                                                     activation='gelu',
                                                     batch_first=True,
                                                     norm_first=True)

        self.transformer_encoder = nn.TransformerEncoder(self.mhsa_layer, num_layers=4)

        self.ffn = nn.Sequential(nn.GELU(),
                                 nn.Linear(self.d_model,64),
                                 nn.GELU(),
                                 nn.Linear(64,16))

    def forward(self, noise, ct, time):
        noise = self.noise_embedding(noise)
        time  = self.time_embedding(time)

        ct = ct+time

        x = torch.stack([noise,ct],dim=1).permute(1,0,2)

        #x = self.positional_encoding(x)
        x = self.transformer_encoder(x)

        x = x.permute(1,0,2)
        x = x[:,0]
        x = self.ffn(x)

        x = x.view(-1,8,2)

        return x