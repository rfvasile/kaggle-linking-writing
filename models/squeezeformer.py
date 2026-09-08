import torch
from torch import Tensor, nn


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    """Gated linear unit: half the channels gate the other half."""

    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class DepthwiseConv1d(nn.Module):
    """One filter per input channel — `groups=in_channels` keeps channels from mixing."""

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


class PointwiseConv1d(nn.Module):
    """kernel_size=1 conv — mixes channels at a single timestep, used to match dimensions."""

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


class FeedForwardModule(nn.Module):
    """Linear(d, 4d) -> Swish -> Dropout -> Linear(4d, d) -> Dropout. Unchanged since Vaswani 2017.

    Inputs/Outputs: (batch, time, dim).
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        return x


def make_scale(encoder_dim: int) -> tuple[nn.Parameter, nn.Parameter]:
    """Learned per-channel scale/bias, shaped [1,1,C] to broadcast over (B,T,C)."""
    scale = torch.nn.Parameter(torch.tensor([1.0] * encoder_dim)[None, None, :])
    bias = torch.nn.Parameter(torch.tensor([0.0] * encoder_dim)[None, None, :])
    return scale, bias


class ConvModule(nn.Module):
    """pointwise -> GLU -> depthwise -> BN -> Swish -> pointwise -> dropout.

    Inputs:
        x (batch, time, dim)
        mask_pad (batch, 1, time): True at real positions.
    Outputs:
        (batch, time, dim), pad positions zeroed.
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

    def forward(self, x: Tensor, mask_pad: Tensor) -> Tensor:
        x = x.transpose(1, 2)  # (B,T,C) -> (B,C,T), the layout Conv1d wants
        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)

        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)

        # BatchNorm must see real positions only, otherwise the pad frames bias the
        # batch statistics (and the running stats) toward zero.
        b, c, t = x.shape
        flat_x = x.transpose(1, 2).flatten(0, 1)  # (B,C,T) -> (B*T, C)
        flat_mask = mask_pad.flatten()  # True at real positions
        normed = torch.zeros_like(flat_x)
        normed[flat_mask] = self.bn(flat_x[flat_mask])
        x = normed.view(b, t, c).transpose(1, 2)  # (B*T,C) -> (B,C,T)

        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)

        if mask_pad.size(2) > 0:  # time > 0
            x = x.masked_fill(~mask_pad, 0.0)
        return x.transpose(1, 2)  # (B,C,T) -> (B,T,C)
