import torch
from torch import nn
from torch.nn import functional as F
import modules.utils as utils
import modules.modules
import modules.utils as utils
import utils
from utils import f0_to_coarse
from reference_tool import vencoder, vdecoder

class SynthesizerTrn(nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 gin_channels, ssl_dim, n_speakers, sampling_rate=44100, vol_embedding=False,
                 vocoder_name="nsf-hifigan", use_depthwise_conv=False, use_automatic_f0_prediction=True,
                 flow_share_parameter=False, n_flow_layer=4, n_layers_trans_flow=3,
                 use_transformer_flow=False, **kwargs):

        super().__init__()
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.use_depthwise_conv = use_depthwise_conv
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        self.n_layers_trans_flow = n_layers_trans_flow
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        if vol_embedding:
           self.emb_vol = nn.Linear(1, hidden_channels)

        self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
            "use_depthwise_conv":use_depthwise_conv
        }
        
        modules.set_Conv1dModel(self.use_depthwise_conv)

        if vocoder_name == "nsf-hifigan":
            from reference_tool.vdecoder.hifigan.models import Generator
            self.dec = Generator(h=hps)

        self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(inter_channels, hidden_channels, filter_channels, n_heads, n_layers_trans_flow, 5, p_dropout, n_flow_layer,  gin_channels=gin_channels, share_parameter= flow_share_parameter)
        else:
            self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, n_flow_layer, gin_channels=gin_channels, share_parameter= flow_share_parameter)
        if self.use_automatic_f0_prediction:
            self.f0_decoder = F0Decoder(
                1,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                spk_channels=gin_channels
            )
        self.emb_uv = nn.Embedding(2, hidden_channels)
        self.character_mix = False

def forward(self, c, f0, uv, spec, g=None, c_lengths=None, spec_lengths=None, vol = None):
        # 将g转换为embedding
        g = self.emb_g(g).transpose(1,2)
        # 卷积层
        # vol proj
        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0
        # ssl prenet
        x_mask = torch.unsqueeze(utils.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1,2) + vol
        # f0 predict
        lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
        norm_lf0 = utils.normalize_f0(lf0, x_mask, uv)
        pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
        # encoder
        z_ptemp, m_p, logs_p, _ = self.enc_p(x, x_mask, f0=f0_to_coarse(f0))
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        # flow
        z_p = self.flow(z, spec_mask, g=g)
        # 随机分割
        z_slice, pitch_slice, ids_slice = utils.rand_slice_segments_with_pitch(z, f0, spec_lengths, self.segment_size)
        # nsf decoder
        o = self.dec(z_slice, g=g, f0=pitch_slice)
        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0

   def infer(self, c, f0, uv, g=None, noice_scale=0.35, seed=52468, predict_f0=False, vol = None):

        '''
        :param c: [N, S, B, H]
        :param f0: [N, S, B, 1]
        :param uv: [N, S, B, 1]
        :param g: [B, H]
        :param noice_scale:
        :param seed:
        :param predict_f0:
        :param vol: [N, S, 1, H]
        :return:
        '''

        if c.device == torch.device("cuda"):
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)
        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if self.character_mix and len(g) > 1:   # [N, S]  *  [S, B, 1, H]
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
            g = g * self.speaker_map  # [N, S, B, 1, H]
            g = g * self.speaker_map  # [N, S, B, 1, 1]
            g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)
        x_mask = torch.unsqueeze(utils.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        # vol proj
        vol = self.emb_vol(vol[:,:,None]).transpose(1,2) if vol is not None and self.vol_embedding else 0
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol
        # f0 predict
        lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
        norm_lf0 = utils.normalize_f0(lf0, x_mask, uv, random_scale=False)
        pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
        f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        
        # encoder
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), noice_scale=noice_scale)
        # flow
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        # decoder
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o, f0
