import torch as pt
import torch.nn.functional as F


class StochasticPatchMSELoss(pt.nn.modules.loss._Loss):
    def __init__(self, patch_size: int):
        super(StochasticPatchMSELoss, self).__init__()
        self.ps = patch_size

    def forward(self, yt: pt.Tensor, yh: pt.Tensor) -> pt.Tensor:
        oo = pt.randint(0, yt.shape[1] - self.ps, (2,))
        yt = yt[:, oo[0] : oo[0] + self.ps, oo[1] : oo[1] + self.ps]
        yh = yh[:, oo[0] : oo[0] + self.ps, oo[1] : oo[1] + self.ps]
        loss = F.mse_loss(yt, yh)
        return loss
