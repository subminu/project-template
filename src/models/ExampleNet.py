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
            nn.Conv2d(32, cv2_ch, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1, 7 * 7 * 64, output_size),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)
