import torch as pt
import torch_geometric as ptg


class GATEncoder(pt.nn.Module):
    """
    Graph Attention Network (GAT) encoder.

    Args:
    ----
        - in_channels (int): Number of input channels.
        - hidden_channels (int): Number of hidden channels.
        - num_layers (int): Number of GAT layers.
        - out_channels (int): Number of output channels.
        - dropout (float): Dropout rate.
        - gat_heads (int): Number of GAT heads.
        - n_blocks (int): Number of GAT blocks.
    """

    def __init__(
        self,
        in_channels: int = 450,
        hidden_channels: int = 128,
        num_layers: int = 2,
        out_channels: int = 64,
        dropout: float = 0.2,
        gat_heads: int = 4,
        n_blocks: int = 2,
        num_nodes: int = 128,
        out_dim: int = 1024,
    ):
        super().__init__()
        self.gat_blocks = pt.nn.ModuleList()
        for b in range(n_blocks):
            ich = in_channels if b == 0 else hidden_channels
            och = hidden_channels if b < n_blocks - 1 else out_channels
            gat = ptg.nn.models.GAT(
                ich,
                hidden_channels,
                num_layers,
                out_channels=och,
                dropout=dropout,
                norm="graph",
                jk="cat",
                v2=True,
                heads=gat_heads,
            )
            bnm = pt.nn.BatchNorm1d(och)
            self.gat_blocks.extend((gat, bnm))
        self.inter_block_act = pt.nn.ReLU()
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        self.fc_layers = pt.nn.Sequential(
            pt.nn.Linear(out_channels * num_nodes, out_dim),
            pt.nn.ReLU(),
        )

    def _forward(self, x: pt.Tensor, edge_index: pt.Tensor = None) -> pt.Tensor:
        for b in range(0, len(self.gat_blocks), 2):
            gat, bnm = self.gat_blocks[b], self.gat_blocks[b + 1]
            residual = x
            x = gat(x, edge_index)
            x = bnm(x)
            if b not in [0, len(self.gat_blocks) - 2]:
                x = self.inter_block_act(x + residual)
            else:
                x = self.inter_block_act(x)
        x = x.view(-1)
        x = self.fc_layers(x)
        return x

    def forward(self, x: pt.Tensor, edge_index: pt.Tensor) -> pt.Tensor:
        batch_size = x.size(0)
        output = pt.zeros(batch_size, self.out_dim).to(x.device)
        for i, (x_, edge_index_) in enumerate(zip(x, edge_index)):
            output[i] = self._forward(x_, edge_index_)
        return output


class EEGEncoderSimple(pt.nn.Module):
    def __init__(
        self,
        eeg_shape: tuple[int, int],
        outc1: int = 128,
        outc2: int = 32,
        lat_dim: int = 1024,
        device: str = "cuda",
    ):
        super(EEGEncoderSimple, self).__init__()
        nec = eeg_shape[0]
        row = pt.arange(nec).view(-1, 1).repeat(1, nec).view(-1)
        col = pt.arange(nec).view(-1, 1).repeat(nec, 1).view(-1)
        self.eins = pt.stack([row, col], dim=0)
        self.conv_spat = ptg.nn.GATv2Conv(
            in_channels=-1,
            out_channels=outc1,
            heads=4,
            dropout=0.1,
            concat=False,
        )

        row = pt.arange(outc1).view(-1, 1).repeat(1, outc1).view(-1)
        col = pt.arange(outc1).view(-1, 1).repeat(outc1, 1).view(-1)
        self.eint = pt.stack([row, col], dim=0)
        self.conv_temp = ptg.nn.GATv2Conv(
            in_channels=-1,
            out_channels=outc2,
            heads=4,
            dropout=0.1,
            concat=False,
        )
        self.lat_dim = lat_dim

        self.fc1 = pt.nn.Linear(outc1 * outc2, lat_dim)
        self.fc2 = pt.nn.Linear(lat_dim, lat_dim)
        self.fc3 = pt.nn.Linear(lat_dim, lat_dim)
        self.act_fn = pt.nn.LeakyReLU()
        self.fin_ac = pt.nn.LeakyReLU()

    def _forward(self, x: pt.Tensor) -> pt.Tensor:
        x = self.conv_spat(x, self.eins)
        x = x.T
        x = self.conv_temp(x, self.eint)
        x = x.view(-1)
        x = self.fc1(x)
        xr = self.act_fn(x)
        x = self.fc2(xr)
        x = x + xr
        x = self.fc3(x)
        x = self.act_fn(x)
        x = self.fin_ac(x)
        return x

    def forward(self, x: pt.Tensor, ein: pt.Tensor) -> pt.Tensor:
        out = pt.empty(x.size(0), self.lat_dim, device=x.device)
        for i in range(x.size(0)):
            out[i] = self._forward(x[i])
        return out

    def to(self, *args, **kwargs):
        model = super(EEGEncoderSimple, self).to(*args, **kwargs)
        self.eint = self.eint.to(*args, **kwargs)
        self.eins = self.eins.to(*args, **kwargs)
        return model


class EEGChannelNet(pt.nn.Module):
    def __init__(self, latent_dim=1024):
        super(EEGChannelNet, self).__init__()
        temp_dil = [1, 2, 4, 8, 16]
        temp_pad = [16, 32, 64, 128, 256]
        self.temp_block = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.Conv2d(
                        in_channels=1,
                        out_channels=10,
                        kernel_size=(1, 33),
                        stride=(1, 2),
                        dilation=(1, d),
                        padding=(0, p),
                    ),
                    pt.nn.BatchNorm2d(10),
                    pt.nn.ReLU(),
                )
                for d, p in zip(temp_dil, temp_pad)
            ]
        )
        spat_krn = [128, 64, 32, 16]
        spat_pad = [63, 31, 15, 7]
        self.spat_block = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.Conv2d(
                        in_channels=50,
                        out_channels=50,
                        kernel_size=(k, 1),
                        stride=(2, 1),
                        dilation=(1, 1),
                        padding=(p, 0),
                    ),
                    pt.nn.BatchNorm2d(50),
                    pt.nn.ReLU(),
                )
                for k, p in zip(spat_krn, spat_pad)
            ]
        )
        self.res_blocks = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.Conv2d(
                        in_channels=200,
                        out_channels=200,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        dilation=(1, 1),
                        padding=(1, 1),
                    ),
                    pt.nn.BatchNorm2d(200),
                    pt.nn.ReLU(),
                    pt.nn.Conv2d(
                        in_channels=200,
                        out_channels=200,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        dilation=(1, 1),
                        padding=(1, 1),
                    ),
                    pt.nn.BatchNorm2d(200),
                )
                for _ in range(4)
            ]
        )
        self.stride_layers = pt.nn.ModuleList(
            [
                pt.nn.Sequential(
                    pt.nn.Conv2d(
                        in_channels=200,
                        out_channels=200,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        dilation=(1, 1),
                        padding=(1, 1),
                    ),
                    pt.nn.BatchNorm2d(200),
                    pt.nn.ReLU(),
                )
                for _ in range(4)
            ]
        )
        self.act = pt.nn.ReLU()
        self.fin_act = pt.nn.LeakyReLU()
        self.fin_conv = pt.nn.Conv2d(
            in_channels=200,
            out_channels=50,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(0, 0),
        )
        self.fc1 = pt.nn.Linear(50 * 2 * 13, latent_dim)
        self.fc2 = pt.nn.Linear(latent_dim, latent_dim) 

    def forward(self, x, ein):
        temp_feat = [conv(x) for conv in self.temp_block]
        x = pt.cat(temp_feat, dim=1)
        spat_feat = [conv(x) for conv in self.spat_block]
        x = pt.cat(spat_feat, dim=1)
        for res, stride in zip(self.res_blocks, self.stride_layers):
            x = self.act(res(x) + x)
            x = stride(x)
        x = self.fin_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fin_act(x)
        x = self.fc2(x) 
        x = self.fin_act(x) 
        return x
