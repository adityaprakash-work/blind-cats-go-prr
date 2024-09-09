import torch as pt


class GenericImageEncoder(pt.nn.Module):
    def __init__(self):
        super(GenericImageEncoder, self).__init__()
        self.conv_layers = pt.nn.Sequential(
            pt.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            pt.nn.ReLU(),
            pt.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            pt.nn.ReLU(),
            pt.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            pt.nn.ReLU(),
            pt.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            pt.nn.ReLU(),
        )

        self.fc_layers = pt.nn.Sequential(
            pt.nn.Linear(256 * 8 * 8, 2056),
            pt.nn.ReLU(),
            pt.nn.Linear(2056, 1024),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class GenericLatentDecoder(pt.nn.Module):
    def __init__(self):
        super(GenericLatentDecoder, self).__init__()
        self.fc_layers = pt.nn.Sequential(
            pt.nn.Linear(1024, 2056),
            pt.nn.ReLU(),
            pt.nn.Linear(2056, 256 * 8 * 8),
        )

        self.conv_layers = pt.nn.Sequential(
            pt.nn.ConvTranspose2d(
                256,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            pt.nn.ReLU(),
            pt.nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            pt.nn.ReLU(),
            pt.nn.ConvTranspose2d(
                64,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            pt.nn.ReLU(),
            pt.nn.ConvTranspose2d(
                32,
                1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            pt.nn.ReLU(),
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(x.size(0), 256, 8, 8)
        x = self.conv_layers(x)
        return x


class GenericImageEncoder1(pt.nn.Module):
    def __init__(
        self,
        num_input_channels: int = 1,
        base_channel_size: int = 32,
        latent_dim: int = 1024,
        act_fn: object = pt.nn.GELU,
    ):
        super().__init__()
        c_hid = base_channel_size
        self.net = pt.nn.Sequential(
            pt.nn.Conv2d(
                num_input_channels,
                c_hid,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            act_fn(),
            pt.nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            pt.nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            pt.nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            pt.nn.Conv2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            act_fn(),
            pt.nn.Flatten(),
            pt.nn.Linear(2 * c_hid * 16 * 16, latent_dim),
            pt.nn.Tanh(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class GenericLatentDecoder1(pt.nn.Module):
    def __init__(
        self,
        num_input_channels: int = 1,
        base_channel_size: int = 32,
        latent_dim: int = 1024,
        act_fn: object = pt.nn.GELU,
    ):
        super().__init__()
        c_hid = base_channel_size
        self.linear = pt.nn.Sequential(
            pt.nn.Linear(latent_dim, 2 * 16 * 16 * c_hid), act_fn()
        )
        self.net = pt.nn.Sequential(
            pt.nn.ConvTranspose2d(
                2 * c_hid,
                2 * c_hid,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            act_fn(),
            pt.nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            pt.nn.ConvTranspose2d(
                2 * c_hid,
                c_hid,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            act_fn(),
            pt.nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            pt.nn.ConvTranspose2d(
                c_hid,
                num_input_channels,
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),
            pt.nn.Tanh(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 16, 16)
        x = self.net(x)
        x = x / 2 + 0.5
        return x
