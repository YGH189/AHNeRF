from typing import Callable, Collection, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn   #GGGGGGGGGGGGGG
from .network_swinir import SwinIR
from .Tozx import BiLevelRoutingAttention_nchw

def positional_encoding(positions, freqs):
    """
    Return positional_encoding results with frequency freqs.
    """
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class General_MLP1(torch.nn.Module):
    """
    A general MLP module with potential input including time position encoding(PE): t_pe, feature PE: fea_pe, 3D position PE: pos_pe,
    view direction PE: view_pe.

    pe > 0: use PE with frequency = pe.
    pe < 0: not use this feautre.
    pe = 0: only use original value.
    """

    def __init__(
        self,
        inChanel: int,
        outChanel: int,
        t_pe: int = 6,
        fea_pe: int = 6,
        pos_pe: int = 6,
        view_pe: int = 6,
        featureC: int = 128,
        n_layers: int = 3,
        use_sigmoid: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()

        self.in_mlpC = inChanel
        self.use_t = t_pe >= 0
        self.use_fea = fea_pe >= 0
        self.use_pos = pos_pe >= 0
        self.use_view = view_pe >= 0
        self.t_pe = t_pe
        self.fea_pe = fea_pe
        self.pos_pe = pos_pe
        self.view_pe = view_pe
        self.use_sigmoid = use_sigmoid

        # Whether use these features as inputs
        if self.use_t:
            self.in_mlpC += 1 + 2 * t_pe * 1
        if self.use_fea:
            self.in_mlpC += 2 * fea_pe * inChanel
        if self.use_pos:
            self.in_mlpC += 3 + 2 * pos_pe * 3
        if self.use_view:
            self.in_mlpC += 3 + 2 * view_pe * 3

        assert n_layers >= 2  # Assert at least two layers of MLP
        layers = [torch.nn.Linear(self.in_mlpC, featureC), torch.nn.ReLU(inplace=True)]

        for _ in range(n_layers - 2):
            layers += [torch.nn.Linear(featureC, featureC), torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Linear(featureC, outChanel)]
        self.mlp = torch.nn.Sequential(*layers)

        if zero_init:
            torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(
        self,
        pts: torch.Tensor,
        viewdirs: torch.Tensor,
        features: torch.Tensor,
        frame_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        MLP forward.
        """
        # Collect input data
        indata = [features]
        if self.use_t:
            indata += [frame_time]
            if self.t_pe > 0:
                indata += [positional_encoding(frame_time, self.t_pe)]
        if self.use_fea:
            if self.fea_pe > 0:
                indata += [positional_encoding(features, self.fea_pe)]
        if self.use_pos:
            indata += [pts]
            if self.pos_pe > 0:
                indata += [positional_encoding(pts, self.pos_pe)]
        if self.use_view:
            indata += [viewdirs]
            if self.view_pe > 0:
                indata += [positional_encoding(viewdirs, self.view_pe)]
        mlp_in = torch.cat(indata, dim=-1)

        rgb = self.mlp(mlp_in)
        if self.use_sigmoid:
            rgb = torch.sigmoid(rgb)

        return rgb

'''swinir'''
class General_MLP(torch.nn.Module):
    """
    A general MLP module with potential input including time position encoding(PE): t_pe, feature PE: fea_pe, 3D position PE: pos_pe,
    view direction PE: view_pe.

    pe > 0: use PE with frequency = pe.
    pe < 0: not use this feautre.
    pe = 0: only use original value.
    """

    def __init__(
            self,
            inChanel: int,
            outChanel: int,
            t_pe: int = 6,
            fea_pe: int = 6,
            pos_pe: int = 6,
            view_pe: int = 6,
            featureC: int = 128,
            n_layers: int = 3,
            use_sigmoid: bool = True,
            zero_init: bool = True,
    ):
        super().__init__()

        self.in_mlpC = inChanel
        self.use_t = t_pe >= 0
        self.use_fea = fea_pe >= 0
        self.use_pos = pos_pe >= 0
        self.use_view = view_pe >= 0
        self.t_pe = t_pe
        self.fea_pe = fea_pe
        self.pos_pe = pos_pe
        self.view_pe = view_pe
        self.use_sigmoid = use_sigmoid

        # Whether use these features as inputs
        if self.use_t:
            self.in_mlpC += 1 + 2 * t_pe * 1
        if self.use_fea:
            self.in_mlpC += 2 * fea_pe * inChanel
        if self.use_pos:
            self.in_mlpC += 3 + 2 * pos_pe * 3
        if self.use_view:
            self.in_mlpC += 3 + 2 * view_pe * 3

        assert n_layers >= 2  # Assert at least two layers of MLP
        layers = [torch.nn.Linear(self.in_mlpC, featureC), torch.nn.ReLU(inplace=True)]

        for _ in range(n_layers - 2):
            layers += [torch.nn.Linear(featureC, featureC), torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Linear(featureC, outChanel)]
        self.mlp = torch.nn.Sequential(*layers)

        if zero_init:
            torch.nn.init.constant_(self.mlp[-1].bias, 0)
#GGGGGGGG
        upscale = 1
        window_size = 1
        height = (142 // upscale // window_size + 1) * window_size
        width = (150 // upscale // window_size + 1) * window_size

        self.swinir_model = SwinIR(upscale=1, img_size=(height, width), in_chans=1,
                                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                                   embed_dim=1, num_heads=[1, 1, 1, 1], mlp_ratio=1, upsampler='pixelshuffledirect')

        self.swinir_fc_layers = [torch.nn.Linear(150, 142), torch.nn.ReLU(inplace=True)]
        self.swinir_fc_layers += [torch.nn.Linear(142, 3)]
        self.swinir_fc = torch.nn.Sequential(*self.swinir_fc_layers)

        # self.mlp_fc_layer = torch.nn.Linear(60, 1)

    def forward(
            self,
            pts: torch.Tensor,
            viewdirs: torch.Tensor,
            features: torch.Tensor,
            frame_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        MLP forward.
        """
        # Collect input data
        indata = [features]
        if self.use_t:
            indata += [frame_time]
            if self.t_pe > 0:
                indata += [positional_encoding(frame_time, self.t_pe)]
        if self.use_fea:
            if self.fea_pe > 0:
                indata += [positional_encoding(features, self.fea_pe)]
        if self.use_pos:
            indata += [pts]
            if self.pos_pe > 0:
                indata += [positional_encoding(pts, self.pos_pe)]
        if self.use_view:
            indata += [viewdirs]
            if self.view_pe > 0:
                indata += [positional_encoding(viewdirs, self.view_pe)]
        mlp_in = torch.cat(indata, dim=-1)
#GGGG
        mlp_in = mlp_in.reshape((mlp_in.shape[0] * mlp_in.shape[1], 1))
        # mlp_in = mlp_in.repeat(1, 60)

        # mlp_in = self.swinir_model.layers[0].residual_group.blocks[0](mlp_in, mlp_in_size)

        rgb = self.swinir_model.layers[0].residual_group.blocks[0].mlp(mlp_in)

        # mlp_in = self.swinir_model(mlp_in)
        # mlp_in = mlp_in.squeeze()

        # rgb = self.mlp_fc_layer(rgb)
        # mlp_in = mlp_in[:, :, :1]
        rgb = rgb.reshape((-1, 150))
        rgb = self.swinir_fc(rgb)

        # rgb = self.mlp(mlp_in)
        if self.use_sigmoid:
            rgb = torch.sigmoid(rgb)

        return rgb


class General_MLP2(torch.nn.Module):
    """
    General MLP + BiLevelRoutingAttention_nchw
    """

    def __init__(
            self,
            inChanel: int,
            outChanel: int,
            t_pe: int = 6,
            fea_pe: int = 6,
            pos_pe: int = 6,
            view_pe: int = 6,
            featureC: int = 128,
            n_layers: int = 3,
            use_sigmoid: bool = True,
            zero_init: bool = True,
    ):
        super().__init__()

        self.in_mlpC = inChanel
        self.use_t = t_pe >= 0
        self.use_fea = fea_pe >= 0
        self.use_pos = pos_pe >= 0
        self.use_view = view_pe >= 0
        self.t_pe = t_pe
        self.fea_pe = fea_pe
        self.pos_pe = pos_pe
        self.view_pe = view_pe
        self.use_sigmoid = use_sigmoid

        # 计算拼接后的通道数（这就是 attention 的 dim）
        if self.use_t:
            self.in_mlpC += 1 + 2 * t_pe * 1
        if self.use_fea:
            self.in_mlpC += 2 * fea_pe * inChanel
        if self.use_pos:
            self.in_mlpC += 3 + 2 * pos_pe * 3
        if self.use_view:
            self.in_mlpC += 3 + 2 * view_pe * 3

        # MLP 部分（保持和原版 General_MLP 一致）
        assert n_layers >= 2
        layers = [torch.nn.Linear(self.in_mlpC, featureC), torch.nn.ReLU(inplace=True)]
        for _ in range(n_layers - 2):
            layers += [torch.nn.Linear(featureC, featureC), torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Linear(featureC, outChanel)]
        self.mlp = torch.nn.Sequential(*layers)

        if zero_init:
            torch.nn.init.constant_(self.mlp[-1].bias, 0)

        # ------------------ BiLevelRoutingAttention 部分 ------------------
        dim = self.in_mlpC  # 注意力通道数 = 拼接后的特征维度
        self.biLive_model = BiLevelRoutingAttention_nchw(
            dim=dim,
            num_heads=8,
            n_win=7,
            qk_scale=None,
            topk=4,
            side_dwconv=3,
            auto_pad=False,
            attn_backend='torch'
        )

    def forward(
            self,
            pts: torch.Tensor,
            viewdirs: torch.Tensor,
            features: torch.Tensor,
            frame_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        MLP forward + BiLevelRoutingAttention.
        Expected mlp_in shape: [B, N, C], where N = H * W.
        """
        # --------- 1. 先按原逻辑拼接所有输入 ----------
        indata = [features]
        if self.use_t:
            indata += [frame_time]
            if self.t_pe > 0:
                indata += [positional_encoding(frame_time, self.t_pe)]
        if self.use_fea:
            if self.fea_pe > 0:
                indata += [positional_encoding(features, self.fea_pe)]
        if self.use_pos:
            indata += [pts]
            if self.pos_pe > 0:
                indata += [positional_encoding(pts, self.pos_pe)]
        if self.use_view:
            indata += [viewdirs]
            if self.view_pe > 0:
                indata += [positional_encoding(viewdirs, self.view_pe)]

        # [B, N, C]
        mlp_in = torch.cat(indata, dim=-1)
        B, N, C = mlp_in.shape
        assert C == self.in_mlpC, f"channel mismatch: {C} vs {self.in_mlpC}"

        # --------- 2. reshape 成 [B, C, H, W] 以适配 NCHW 注意力 ----------
        # 这里简单假设 N = H * W，取一个方形；如果你有自己的 H,W 就直接改这里
        H = W = int(N ** 0.5)
        assert H * W == N, "当前假设 N = H*W 不成立，请自行指定 H,W"

        x = mlp_in.transpose(1, 2).contiguous().view(B, C, H, W)  # [B, C, H, W]

        # --------- 3. 经过 BiLevelRoutingAttention ----------
        x = self.biLive_model(x)  # 仍然是 [B, C, H, W]

        # --------- 4. reshape 回 [B, N, C] 再丢给 MLP ----------
        x = x.view(B, C, N).transpose(1, 2).contiguous()  # [B, N, C]

        rgb = self.mlp(x)  # [B, N, outChanel]
        if self.use_sigmoid:
            rgb = torch.sigmoid(rgb)

        return rgb
