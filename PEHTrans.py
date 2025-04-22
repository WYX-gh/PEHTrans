import torch
import torch.nn as nn
from timm.models.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x,):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4,  
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(
            dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
        )

    def forward(self, x): 
        x = x + self.drop_path(self.attn(self.norm1(x)))  
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    



class ResidualConvBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),  
        nn.InstanceNorm3d(out_dim),
        activation,
        ResidualConvBlock(out_dim, out_dim, stride=1),  
        activation,
        ResidualConvBlock(out_dim, out_dim, stride=1),  
        activation
    )

def conv_3d_NoDown(in_dim, out_dim, activation):
    return nn.Sequential(
        ResidualConvBlock(in_dim, out_dim, stride=1),  
        activation,
        ResidualConvBlock(out_dim, out_dim, stride=1),  
        activation,
         ResidualConvBlock(out_dim, out_dim, stride=1),  
        activation
    )

def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=(1, 2, 2), padding=1),  
        nn.InstanceNorm3d(out_dim),
        activation,
        ResidualConvBlock(in_dim, out_dim, stride=1), 
        activation,
        ResidualConvBlock(out_dim, out_dim, stride=1),  
        activation
    )


def conv_trans_block_z_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,
        #nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False)
        nn.ConvTranspose3d(out_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation
        )

def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,
        nn.ConvTranspose3d(out_dim, out_dim, kernel_size=3, stride=(1,2,2), padding=1, output_padding=(0,1,1)), 
        #nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        nn.InstanceNorm3d(out_dim),
        activation,
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation
        )

class TokenSeg(nn.Module):
    def __init__(self, inch=2, outch=2, downlayer=3, base_channel=32, hidden_size=256, imgsize=[64,192,192], TransformerLayerNum=2):
        super().__init__()
        self.downlayer = downlayer
        self.base_channel = base_channel
        self.hidden_size = hidden_size
        self.imgsize = imgsize
        self.downlayer = downlayer
        activation = nn.LeakyReLU(0.2)
        self.outch=outch

        self.modalities = inch 

        self.encoder1_layer1 = conv_block_2_3d(self.base_channel, self.base_channel, activation)
        self.encoder1_layer2 = conv_block_3d(self.base_channel, self.base_channel * 2, activation)
        self.encoder1_layer3 = conv_block_3d(self.base_channel * 2, self.base_channel * 4, activation)


        self.encoder2_layer1 = conv_block_2_3d(self.base_channel, self.base_channel, activation)
        self.encoder2_layer2 = conv_block_3d(self.base_channel, self.base_channel * 2, activation)
        self.encoder2_layer3 = conv_block_3d(self.base_channel * 2, self.base_channel * 4, activation)


        self.transformer_blocks = nn.ModuleList([
            nn.ModuleList([TransformerBlock(dim=hidden_size, num_heads=8, mlp_ratio=4) for _ in range(TransformerLayerNum)]),
            nn.ModuleList([TransformerBlock(dim=hidden_size, num_heads=8, mlp_ratio=4) for _ in range(TransformerLayerNum)]),
            nn.ModuleList([TransformerBlock(dim=hidden_size, num_heads=8, mlp_ratio=4) for _ in range(TransformerLayerNum)])
        ])



        self.mlp_adjust_c1 = nn.Linear(base_channel*8, hidden_size)
        self.mlp_adjust_c2 = nn.Linear(base_channel*16, hidden_size)
        self.mlp_adjust_c3 = nn.Linear(base_channel*32, hidden_size)

 
        self.up_c2 = nn.Upsample(scale_factor=(2,2,2), mode='trilinear', align_corners=False)# conv_trans_block_3d(hidden_size, hidden_size, activation)
        self.up_c3 = nn.Upsample(scale_factor=(4,4,4), mode='trilinear', align_corners=False)#conv_trans_block_3d(hidden_size, hidden_size, activation)

 
        self.fusion = nn.Sequential(nn.Conv3d(hidden_size*3, 256, kernel_size=1))
        

        self.downsample1 = conv_block_3d(self.base_channel*8, self.base_channel*16, activation)
        self.downsample2 = conv_block_3d(self.base_channel*16, self.base_channel*32, activation)


        self.conv_3d_NoDown1 = conv_3d_NoDown(1, self.base_channel, activation)
        self.conv_3d_NoDown2 = conv_3d_NoDown(1, self.base_channel, activation)
        self.conv_3d_NoDown3 = conv_3d_NoDown(self.base_channel * 3 , self.base_channel * 3 , activation)


        self.trans_1 = conv_trans_block_z_3d(self.base_channel * 16, self.base_channel * 4, activation)
        self.trans_2 = conv_trans_block_z_3d(self.base_channel * 8, self.base_channel * 4, activation)
        self.trans_3 = conv_trans_block_3d(self.base_channel * 4 + self.base_channel * 2, self.base_channel, activation)


        self.decoder_out1=nn.Conv3d(self.base_channel*4+self.base_channel*2,self.outch,3,1,1)
        self.decoder_out2=nn.Conv3d(self.base_channel*8,self.outch,3,1,1)
        self.decoder_out3=nn.Conv3d(self.base_channel*16,self.outch,3,1,1)
        self.decoder_out4=nn.Conv3d(self.base_channel*8,self.outch,3,1,1)


    def transformer_layer(self, feature, transformer_blocks, mlp_layer):
        B, C, D, H, W = feature.shape
        x_seq = feature.view(B, C, -1).permute(0, 2, 1)
        mlp_layer = mlp_layer.to(feature.device) 
        x_seq = mlp_layer(x_seq)
        for block in transformer_blocks:
            x_seq = block(x_seq)
        return x_seq.permute(0, 2, 1).view(B, -1, D, H, W)
    
       
    def forward(self, x):


        modality1, modality2 = torch.chunk(x, 2, dim=1)


        modality1_conv = self.conv_3d_NoDown1(modality1)
        modality2_conv = self.conv_3d_NoDown2(modality2)


        x1 = self.encoder1_layer1(modality1_conv)
        x1_layer1 = x1  
        x1 = self.encoder1_layer2(x1)
        x1_layer2 = x1 
        x1 = self.encoder1_layer3(x1)
        x1_layer3 = x1  

  
        x2 = self.encoder2_layer1(modality2_conv)
        x2_layer1 = x2  
        x2 = self.encoder2_layer2(x2)
        x2_layer2 = x2  
        x2 = self.encoder2_layer3(x2)
        x2_layer3 = x2  


        x1 = torch.cat([x1_layer3, x2_layer3], dim=1)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)


        c1 = self.transformer_layer(x1, self.transformer_blocks[0], self.mlp_adjust_c1 ) 
        c2 = self.transformer_layer(x2, self.transformer_blocks[1], self.mlp_adjust_c2  )
        c3 = self.transformer_layer(x3, self.transformer_blocks[2], self.mlp_adjust_c3  )
        out4=c2


        c2 = self.up_c2(c2)  
        c3 = self.up_c3(c3)  

        fused = torch.cat([c1, c2, c3,], dim=1)
        trans_fused_feature = self.fusion(fused) 
        x=torch.cat([trans_fused_feature,x1_layer3,x2_layer3],dim=1)
        out3 = x
        

        x = self.trans_1(x) 
        x = torch.cat([x, x1_layer2, x2_layer2], dim=1)  
        out2 = x


        x = self.trans_2(x) 
        x = torch.cat([x, x1_layer1, x2_layer1], dim=1) 
        out1 = x


        x = self.trans_3(x)  
        x = torch.cat([x, modality1_conv, modality2_conv], dim=1)  
        x = self.conv_3d_NoDown3(x) 


        x = self.out(x)

        x1=self.decoder_out1(out1)
        x2=self.decoder_out2(out2)
        x3=self.decoder_out3(out3)
        x4=self.decoder_out4(out4)
       
        return [x,x1,x2,x3,x4]
