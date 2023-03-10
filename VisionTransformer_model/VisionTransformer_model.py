import torch
from torch import nn
from functools import partial
# --------------------------------------- #
#（1）patch embedding
'''
img_size=224 : 输入图像的宽高
patch_size=16 ： 每个patch的宽高，也是卷积核的尺寸和步长
in_c=3 ： 输入图像的通道数
embed_dim=768 ： 卷积输出通道数
'''
# --------------------------------------- #
class patchembed(nn.Module):
    # 初始化
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768):
        super(patchembed, self).__init__()
        
        # 输入图像的尺寸224*224
        self.img_size = (img_size, img_size)
        # 每个patch的大小16*16
        self.patch_size = (patch_size, patch_size)
        # 将输入图像划分成14*14个patch
        self.grid_size = (img_size//patch_size, img_size//patch_size)
        # 一共有14*14个patch
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 使用16*16的卷积切分图像，将图像分成14*14个
        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
        
        # 定义标准化方法，给LN传入默认参数eps
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(embed_dim)
        
        
    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        B, C, H, W = inputs.shape
        
        # 如果输入图像的宽高不等于224*224就报错
        assert H==self.img_size[0] and W==self.img_size[1], 'input shape does not match 224*224'
        
        # 卷积层切分patch [b,3,224,224]==>[b,768,14,14]
        x = self.proj(inputs)
        # 展平 [b,768,14,14]==>[b,768,14*14]
        x = x.flatten(start_dim=2, end_dim=-1)  # 将索引为 start_dim 和 end_dim 之间（包括该位置）的数量相乘
        # 维度调整 [b,768,14*14]==>[b,14*14,768]
        x = x.transpose(1, 2)  # 实现一个张量的两个轴之间的维度转换
        # 标准化
        x = self.norm(x)
        
        return x
 # --------------------------------------- #
#（2）类别标签和位置标签
'''
embed_dim : 代表patchembed层输出的通道数
'''
# --------------------------------------- #
class class_token_pos_embed(nn.Module):
    # 初始化
    def __init__(self, embed_dim):
        super(class_token_pos_embed, self).__init__()
        
        # patchembed层将图像划分的patch个数==14*14
        num_patches = patchembed().num_patches
        
        self.num_tokens = 1  # 类别标签
        
        # 创建可学习的类别标签 [1,1,768]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 创建可学习的位置编码 [1,196+1,768]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+self.num_tokens, embed_dim))
        
        # 权重以正态分布初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
 
    # 前向传播
    def forward(self, x):  # 输入特征图的shape=[b,196,768]
        
        # 类别标签扩充维度 [1,1,768]==>[b,1,768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
 
        # 将类别标签添加到特征图中 [b,1,768]+[b,196,768]==>[b,197,768]
        x = torch.cat((cls_token, x), dim=1)
        
        # 添加位置编码 [b,197,768]+[1,197,768]==>[b,197,768]
        x = x + self.pos_embed
        
        return x
# --------------------------------------- #
#（3）多头注意力模块
'''
dim : 代表输入特征图的通道数
num_heads : 多头注意力中heads的个数
qkv_bias ： 生成qkv时是否使用偏置 
atten_drop_ratio ：qk计算完之后的dropout层
proj_drop_ratio ： qkv计算完成之后的dropout层
'''
# --------------------------------------- #
class attention(nn.Module):
    # 初始化
    def __init__(self, dim, num_heads=12, qkv_bias=False, atten_drop_ratio=0., proj_drop_ratio=0.):
        super(attention, self).__init__()
        
        # 多头注意力的数量
        self.num_heads = num_heads  
        # 将生成的qkv均分成num_heads个。得到每个head的qkv对应的通道数。
        head_dim = dim // num_heads
        # 公式中的分母
        self.scale = head_dim ** -0.5
        
        # 通过一个全连接层计算qkv
        self.qkv = nn.Linear(in_features=dim, out_features=dim*3, bias=qkv_bias)
        # dropout层
        self.atten_drop = nn.Dropout(atten_drop_ratio)
        
        # 再qkv计算完之后通过一个全连接提取特征
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        # dropout层
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    
    # 前向传播
    def forward(self, inputs):
        # 获取输入图像的shape=[b,197,768]
        B, N, C = inputs.shape
        
        # 将输入特征图经过全连接层生成qkv [b,197,768]==>[b,197,768*3]
        qkv = self.qkv(inputs)
 
        # 维度调整 [b,197,768*3]==>[b, 197, 3, 12, 768//12]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C//self.num_heads)
        # 维度重排==> [3, B, 12, 197, 768//12]
        qkv = qkv.permute(2,0,3,1,4)
        # 切片提取q、k、v的值，单个的shape=[B, 12, 197, 768//12]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 针对每个head计算 ==> [B, 12, 197, 197] 
        atten = (q @ k.transpose(-2,-1)) * self.scale  # @ 代表在多维tensor的最后两个维度矩阵相乘
        # 对计算结果的每一行经过softmax
        atten = atten.softmax(dim=-1)
        # dropout层
        atten = self.atten_drop(atten)
        
        # softmax后的结果和v加权 ==> [B, 12, 197, 768//12]
        x = atten @ v
        # 通道重排 ==> [B, 197, 12, 768//12]
        x = x.transpose(1,2)
        # 维度调整 ==> [B, 197, 768]
        x = x.reshape(B,N,C)
        
        # 通过全连接层融合特征 ==> [B, 197, 768]
        x = self.proj(x)
        # dropout层
        x = self.proj_drop(x)
        
        return x

        # [b,197,3072]==>[b,197,768]
        x = self.fc2(x)
        x = self.drop(x)
        
        return x
# --------------------------------------- #
#（4）MLP多层感知器
'''
in_features : 输入特征图的通道数
hidden_features : 第一个全连接层上升通道数
out_features : 第二个全连接层的下降的通道数
drop : 全连接层后面的dropout层的杀死神经元的概率
'''
# --------------------------------------- #  
# --------------------------------------- #
#（4）MLP多层感知器
'''
in_features : 输入特征图的通道数
hidden_features : 第一个全连接层上升通道数
out_features : 第二个全连接层的下降的通道数
drop : 全连接层后面的dropout层的杀死神经元的概率
'''
# --------------------------------------- #  
class MLP(nn.Module):
    # 初始化
    def __init__(self, in_features, hidden_features, out_features=None, drop=0.):
        super(MLP, self).__init__()
        
        # MLP的输出通道数默认等于输入通道数
        out_features = out_features or in_features
        # 第一个全连接层上升通道数
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        # GeLU激活函数
        self.act = nn.GELU()
        # 第二个全连接下降通道数
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        # dropout层
        self.drop = nn.Dropout(drop)
    
    # 前向传播
    def forward(self, inputs):
        
        # [b,197,768]==>[b,197,3072]
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x)
        
        # [b,197,3072]==>[b,197,768]
        x = self.fc2(x)
        x = self.drop(x)
        
        return x
# --------------------------------------- #
#（5）Encoder Block
'''
dim : 该模块的输入特征图个数
mlp_ratio ： MLP中第一个全连接层上升的通道数
drop_ratio : 该模块的dropout层的杀死神经元的概率
'''
# --------------------------------------- # 
class encoder_block(nn.Module):
    # 初始化
    def __init__(self, dim, mlp_ratio=4., drop_ratio=0.):
        super(encoder_block, self).__init__()
        
        # LayerNormalization层
        self.norm1 = nn.LayerNorm(dim)
        # 实例化多头注意力
        self.atten = attention(dim)
        # dropout
        self.drop = nn.Dropout()
        
        # LayerNormalization层
        self.norm2 = nn.LayerNorm(dim)
        # MLP中第一个全连接层上升的通道数
        hidden_features = int(dim * mlp_ratio)
        # MLP多层感知器
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features)
        
    # 前向传播
    def forward(self, inputs):
        
        # [b,197,768]==>[b,197,768]
        x = self.norm1(inputs)
        x = self.atten(x)
        x = self.drop(x)
        feat1 = x + inputs  # 残差连接
        
        # [b,197,768]==>[b,197,768]
        x = self.norm2(feat1)
        x = self.mlp(x)
        x = self.drop(x)
        feat2 = x + feat1  # 残差连接
        
        return feat2
# --------------------------------------- #
#（6）主干网络
'''
num_class: 分类数
depth : 重复堆叠encoder_block的次数
drop_ratio : 位置编码后的dropout层
embed_dim : patchembed层输出通道数
'''
# --------------------------------------- # 
class VIT(nn.Module):
    # 初始化
    def __init__(self, num_classes=1000, depth=12, drop_ratio=0., embed_dim=768):
        super(VIT, self).__init__()
        
        self.num_classes = num_classes  # 分类类别数
        
        # 实例化patchembed层
        self.patchembed = patchembed()
        
        # 实例化类别标签和位置编码
        self.cls_pos_embed = class_token_pos_embed(embed_dim=embed_dim)        
 
        # 位置编码后做dropout
        self.pos_drop = nn.Dropout(drop_ratio)
        
        # 在列表中添加12个encoder_block
        self.blocks = nn.Sequential(*[encoder_block(dim=embed_dim) for _ in range(depth)])
        
        # 定义LayerNormalization标准化方法
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # 经过12个encoder之后的标准化层
        self.norm = norm_layer(embed_dim)
        
        # 分类层
        self.head = nn.Linear(in_features=embed_dim, out_features=num_classes)
        
        # 权值初始化
        for m in self.modules():
            # 对卷积层使用kaiming初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # 对偏置初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # 对标准化层初始化
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # 对全连接层初始化
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
 
    
    # 前向传播
    def forward(self, inputs):
        
        # 先将输入传递给patchembed [b,3,224,224]==>[b,196,768]
        x = self.patchembed(inputs)
        
        # 对特征图添加类别标签和位置编码
        x = self.cls_pos_embed(x)
        
        # dropout层
        x = self.pos_drop(x)
        
        # 经过12个encoder层==>[b,197,768]
        x = self.blocks(x)
        
        # LN标准化层
        x = self.norm(x)
        
        # 提取类别标签的输出,因为在cat时将类别标签放在最前面
        x = x[:, 0]  # [b,197,768]==>[b,768]
 
        # 全连接层分类 [b,768]==>[b,1000]
        x = self.head(x)
        
        return x
