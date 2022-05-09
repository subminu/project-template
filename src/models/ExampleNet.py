from torch import nn


class ExampleNet(nn.Module):
    def __init__(
        self,
        cv1_ch: int = 32,
        cv2_ch: int = 64,
        output_size: int = 10,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, cv1_ch, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(cv1_ch, cv2_ch, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 256 * cv2_ch, output_size),
        )

    def forward(self, x):
        return self.model(x)
