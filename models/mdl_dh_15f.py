from torch import nn
import torch
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np
from transformers import AutoModel, AutoConfig
import torch.utils.checkpoint

from typing import Tuple, Union, Optional
import typing
from torch import Tensor
# from transformers.models.speech_to_text import Speech2TextConfig, Speech2TextForConditionalGeneration
# from transformers.models.speech_to_text.modeling_speech_to_text import shift_tokens_right, Speech2TextDecoder
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding
from timm.layers.norm_act import BatchNormAct2d


'''
self = Net(cfg)
'''


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q = q.unsqueeze(1)
    k = k.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.squeeze(1)
    k_embed = k_embed.squeeze(1)
    return q_embed, k_embed

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        #self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        '''
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        '''
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos[:, :, :kv_seq_len], sin[:, :, :kv_seq_len])
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights.masked_fill_(attention_mask, torch.finfo(attn_weights.dtype).min)


        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class FeedForwardModule(nn.Module):
    """
    Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of squeezeformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        
        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        
        return x


class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: False
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class DepthwiseConv2d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 2
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 2-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: int = 2,
        padding: int = 0,
    ) -> None:
        super(DepthwiseConv2d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConvModule(nn.Module):
    """
    Convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by squeezeformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)

    # mask_pad = mask.bool().unsqueeze(1)
    def forward(self, x, mask_pad):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        Reference for masking : https://github.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/built-in/audio/Wenet_Conformer_for_Pytorch/wenet/transformer/convolution.py#L26
        """
        # mask batch padding
        x = x.transpose(1, 2)
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        # torch.Size([4, 128, 384])
        x_bn = x.permute(0,2,1).reshape(-1, x.shape[1])
        mask_bn = mask_pad.view(-1)
        x_bn[mask_bn] = self.bn(x_bn[mask_bn])
        x = x_bn.view(x.permute(0,2,1).shape).permute(0,2,1)
        '''
        x = self.bn(x)
        '''
        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        x = x.transpose(1, 2)
        return x

def make_scale(encoder_dim):
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None,None,:])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None,None,:])
    return scale, bias

class SqueezeformerBlock(nn.Module):
    """
    SqueezeformerBlock is a simpler block structure similar to the standard Transformer block,
    where the MHA and convolution modules are each directly followed by a single feed forward module.

    Args:
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by squeezeformer block.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super(SqueezeformerBlock, self).__init__()
        
        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)
        
        self.mhsa_llama = LlamaAttention(LlamaConfig(hidden_size = encoder_dim, 
                                       num_attention_heads = num_attention_heads, 
                                       max_position_embeddings = 384))
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        
        self.ff_mhsa = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
        
        # Attention_mask = (bsz, self.num_heads, q_len, kv_seq_len)   
            
            
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)
        
    def forward(self, x, cos, sin, mask):
        mask_pad = ( mask).long().bool().unsqueeze(1)
        mask_pad = ~( mask_pad.permute(0, 2,1) * mask_pad)
        mask_flat = mask.view(-1).bool()
        bs, slen, nfeats = x.shape
        
        residual = x
        x = x * self.scale_mhsa.to(x.dtype) + self.bias_mhsa.to(x.dtype)
        x = residual + self.mhsa_llama(x, cos, sin, attention_mask = mask_pad.unsqueeze(1) )[0]
        # Skip pad #1
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        x = self.ln_mhsa(x) #casts to fp32
        residual = x
        #32
        x = x * self.scale_ff_mhsa.to(x.dtype) + self.bias_ff_mhsa.to(x.dtype)
        #32
        x = residual + self.ff_mhsa(x) 
        #32
        x = self.ln_ff_mhsa(x) #casts to fp32
        
        
        # Unskip pad #1
#         print(x_skip[mask_flat].dtype, x[0].dtype)
        x_skip[mask_flat] = x[0].to(x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)
        residual = x
        # torch.Size([16, 384, 128])
        x = x * self.scale_conv.to(x.dtype) + self.bias_conv.to(x.dtype)
        x = residual + self.conv(x, mask_pad = mask.bool().unsqueeze(1))
        # Skip pad #2
        x_skip = x.view(-1, x.shape[-1])
        x = x_skip[mask_flat].unsqueeze(0)
        
        x = self.ln_conv(x)
        
        
        residual = x
        x = x * self.scale_ff_conv.to(x.dtype) + self.bias_ff_conv.to(x.dtype)
        x = residual + self.ff_conv(x)
        x = self.ln_ff_conv(x)
        
        # Unskip pad #2
        x_skip[mask_flat] = x[0].to(x_skip.dtype)
        x = x_skip.view(bs, slen, nfeats)  
        
        
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def glorot_uniform(parameter):
    nn.init.xavier_uniform_(parameter.data, gain=1.0)


class FeedForward(nn.Module):
    def __init__(self, d_input, d_model, dim_feedforward, dropout):
        super().__init__()
        # Implementation of Feedforward model
        layer_norm_eps  = 1e-5
        self.linear1 = nn.Linear(d_input, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        
    def forward(self, x):
        x = self.norm1(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        x = self.norm2(self.dropout(x))
        return x

class LinearHead(nn.Module):
    def __init__(self, input_dim, output_dim,drop_out=0.5):
        super(LinearHead, self).__init__()
        dropouts = [drop_out*0.2*i for i in range(1,6)]
        self.dropout1 = nn.Dropout(dropouts[0])
        self.dropout2 = nn.Dropout(dropouts[1])
        self.dropout3 = nn.Dropout(dropouts[2])
        self.dropout4 = nn.Dropout(dropouts[3])
        self.dropout5 = nn.Dropout(dropouts[4])
        self.classifier = nn.Linear(input_dim, output_dim)
        glorot_uniform(self.classifier.weight)
        
    def forward(self, x, target=None):
        # x is B x S x C
        logits1 = self.classifier(self.dropout1(x))
        logits2 = self.classifier(self.dropout2(x))
        logits3 = self.classifier(self.dropout3(x))
        logits4 = self.classifier(self.dropout4(x))
        logits5 = self.classifier(self.dropout5(x))
                              
        logits = torch.stack([logits1, logits2, logits3, logits4, logits5]).mean(0)
        return logits

# n_pos, dim = cfg.max_length, self.backbone.config.hidden_size
def create_sinusoidal_embeddings(n_pos, dim, pow_ = 10000):
    out = torch.zeros(n_pos, dim)
    position_enc = np.array([[pos / np.power(pow_, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out2 = torch.nn.Embedding(n_pos, dim)
    out2.load_state_dict({'weight':out})
    out2.requires_grad = False
    return out2

# from PIL import Image
# Image.fromarray(((position_enc + 1)*100).astype(np.uint8))

class MCRMSE(nn.Module):
    def __init__(self):
        super(MCRMSE, self).__init__()
    
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, logits, targets):
        
        loss = torch.sqrt(self.mse(logits, targets).mean(0)).mean()
        return loss
    

class FeatureExtractor(nn.Module):
    def __init__(self, ksize = 21, out_dim = 128, conv_ch = 3):
        super().__init__()   
        
        assert ksize%2 == 1
        self.stem_linear = nn.Linear(out_dim,out_dim,bias=False)
        self.stem_bn = nn.BatchNorm1d(out_dim, momentum=0.95)
        self.conv_stem = nn.Conv1d(conv_ch, out_dim, kernel_size=ksize, stride=1, padding=ksize//2, bias=False)
        self.bn_conv = nn.BatchNorm1d(out_dim, momentum=0.95)
        
    def forward(self, data, mask):
        
        '''
        data, mask = feat_enc, batch['attention_mask'].bool()
        '''
        xc = data
        xc = self.conv_stem(xc)
        xc = self.bn_conv(xc)
        
        m = mask.to(torch.bool)  
        x = self.stem_linear(xc.permute(0, 2, 1))
        
        # Batchnorm without pads
        bs,slen,nfeat = x.shape
        x = x.view(-1, nfeat)
        x_bn = x[mask.view(-1)==1].unsqueeze(0)
        x_bn = self.stem_bn(x_bn.permute(0,2,1)).permute(0,2,1)
        x[mask.view(-1)==1] = x_bn[0]
        x = x.view(bs,slen,nfeat)
        # Padding mask
        x = x.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
        
        return x
    
    
class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg
        
        if self.cfg.offline_inference:
            config = AutoConfig.from_pretrained(cfg.backbone, **cfg.backbone_cfg)
            self.backbone = AutoModel.from_config(config)
        else:
            config = AutoConfig.from_pretrained(cfg.backbone, **cfg.backbone_cfg)
            self.backbone = AutoModel.from_pretrained(cfg.backbone, config=config)
            
        if self.cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        
        requires_grad = False
        for nm,p in self.backbone.named_parameters():
            if not cfg.freeze_embeddings:
                if 'embeddings' in nm:
                    print(p.requires_grad, nm)
                    continue
            if cfg.freeze_until in nm:
                requires_grad = True
            p.requires_grad = requires_grad
        
            # print(p.requires_grad, nm)
                
        # Image.fromarray((100 * (self.time_emb.weight + 1 )).to(torch.uint8).numpy())
        
        #cfg.feat_dim = 256
        #cfg.feat_init_ksize = 21
        
        self.feature_extractor = FeatureExtractor(ksize =cfg.feat_init_ksize, out_dim=cfg.feat_dim, conv_ch = 2)
        self.feature_extractor2 = FeatureExtractor(ksize =cfg.feat_init_ksize, out_dim=cfg.feat_dim, conv_ch = 1)

        rotary_emb = LlamaRotaryEmbedding(cfg.feat_dim * 2 // 4, max_position_embeddings=self.cfg.max_times)
        self.cos = torch.nn.parameter.Parameter(rotary_emb.cos_cached, requires_grad=False)#[:, :, :seq_len, ...]#.to(dtype=x.dtype)
        self.sin = torch.nn.parameter.Parameter(rotary_emb.sin_cached, requires_grad=False)#[:, :, :seq_len, ...]#.to(dtype=x.dtype)
        
        
        self.mha0 = SqueezeformerBlock(
            encoder_dim=cfg.feat_dim * 2,
            num_attention_heads=4,# num_attention_heads,
            feed_forward_expansion_factor=1,#feed_forward_expansion_factor,
            conv_expansion_factor=2,#conv_expansion_factor,
            feed_forward_dropout_p=0.,#feed_forward_dropout_p,
            attention_dropout_p=0.,#attention_dropout_p,
            conv_dropout_p=0.,#conv_dropout_p,
            conv_kernel_size=31,#conv_kernel_size,
        )
        
        self.feat_dim = cfg.feat_dim * 2
        
        self.mha1 = nn.MultiheadAttention(self.backbone.config.hidden_size + self.feat_dim, 4, batch_first = True)
        self.mha2 = nn.MultiheadAttention(self.mha1.kdim, 4, batch_first = True)
        self.head = LinearHead(self.mha1.kdim, 1)
        self.aux_head1 = LinearHead(self.backbone.config.hidden_size, 1)
        self.aux_head2 = LinearHead(self.feat_dim, 1)
        
        self.dropout = cfg.dropout_mha2
        
#         self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn = MCRMSE()
        self.return_logits = True
        
        print('n_params:',count_parameters(self))
        
    def forward(self, batch):
        
        x = self.backbone(input_ids=batch['input_ids'],
                          attention_mask=batch['attention_mask']).last_hidden_state
        
        cum_enc = torch.log(1+batch['cumtimes']) / 8.
        up_enc = torch.log(1+batch['uptimes']) / 8.
        cursor = torch.log(1+batch['cursor']) / 8.
        feat_enc = torch.stack((cum_enc, up_enc, cursor)).permute(1, 0, 2).float()
        feat_enc = torch.nan_to_num(feat_enc)
        feat_enc, feat_enc2 = feat_enc[:,:2], feat_enc[:,2:]
        feat_enc = self.feature_extractor(feat_enc, batch['attention_mask'].bool())
        feat_enc2 = self.feature_extractor2(feat_enc2, batch['attention_mask'].bool())
        feat_enc = torch.cat((feat_enc, feat_enc2), -1)
        feat_out = self.mha0(feat_enc, self.cos, self.sin, batch['attention_mask'].bool())
        
        # feat_out = self.ffn0(feat_out)
        
        out0 = torch.cat((feat_out, x), -1)
        out1, _ = self.mha1(out0, out0, out0, key_padding_mask=~batch['attention_mask'].bool())
        # out1, _ = self.mha1(x, feat_enc, feat_enc, key_padding_mask=~batch['attention_mask'].bool())
        out1 = self.dropout(out1)
        out2, _ = self.mha2(out1[:,:1], out1, out1, key_padding_mask=~batch['attention_mask'].bool())
        
        logits = self.head(out2[:,0]) # torch.Size([4, 1656, 9])       
#         preds = logits.softmax(1)
        output = {'idx': batch['idx']} 
        if 'target' in batch.keys():
            
            logits_aux1 = self.aux_head1(x[:,0]) # torch.Size([4, 1656, 9])    
            logits_aux2 = self.aux_head2(feat_out[:,0]) # torch.Size([4, 1656, 9])  
            loss = self.loss_fn(logits,batch['target'].unsqueeze(1))
            loss_aux1 = self.loss_fn(logits_aux1,batch['target'].unsqueeze(1))
            loss_aux2 = self.loss_fn(logits_aux2,batch['target'].unsqueeze(1))
#             loss = (loss * batch['attention_mask']).sum() /  batch['attention_mask'].sum()
            loss = loss + 0.1 * loss_aux1  + 0.1 * loss_aux2
            output.update({'loss':loss, 
                           'loss_txt':loss_aux1, 
                           'loss_feats':loss_aux2, 
                   })

        if (not self.training ) & self.return_logits:
            output['logits'] = logits
            
#             output['offsets'] = batch['offsets']

        return output
