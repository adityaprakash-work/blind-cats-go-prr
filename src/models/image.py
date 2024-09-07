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
