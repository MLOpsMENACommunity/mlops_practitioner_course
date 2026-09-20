"""Plate recognizer: grayscale crop [B,1,32,128] -> per-timestep logits [B,32,37] for CTC.

  - ``crnn``      baseline/teacher: conv features + 2-layer BiLSTM (the textbook CRNN)
  - ``conv_ctc``  student: no recurrence; context comes from 1-D convs along the width

Both emit the same 32 time steps, which is what lets the student match the teacher's
per-step character distributions during distillation.
"""

from __future__ import annotations

from torch import Tensor, nn

from src import config

OUTPUT_LAYERS = ("fc",)  # the CTC projection: never pruned, kept high precision


def _cbr(cin: int, cout: int, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, groups=groups, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    """Conv stack -> BiLSTM -> Linear. Height is collapsed to 1 before the sequence model.

    The BiLSTM reads the whole width at once, so with few training crops it learns the
    plate FORMATS and guesses characters it cannot see. Dropout between the LSTM layers
    and before the projection is worth about a point of held-out exact match; the rest of
    that gap closes with training data, not with regularization.
    """

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            _cbr(1, 32), nn.MaxPool2d(2),  # 16 x 64
            _cbr(32, 64), nn.MaxPool2d(2),  # 8 x 32
            _cbr(64, 128), _cbr(128, 128), nn.MaxPool2d((2, 1)),  # 4 x 32
            _cbr(128, 192), nn.MaxPool2d((2, 1)),  # 2 x 32
            nn.Conv2d(192, 192, (2, 1)), nn.BatchNorm2d(192), nn.ReLU(inplace=True),  # 1 x 32
        )  # fmt: skip
        self.rnn = nn.LSTM(192, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(256, config.NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        seq = self.cnn(x).squeeze(2).transpose(1, 2)  # [B, 32, 192]
        return self.fc(self.drop(self.rnn(seq)[0]))


class ConvCTC(nn.Module):
    """LSTM-free student: every op is a conv, so every runtime and quantizer handles it."""

    def __init__(self, width: int = 96) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            _cbr(1, 24), nn.MaxPool2d(2),
            _cbr(24, 48), nn.MaxPool2d(2),
            _cbr(48, 48, groups=48), _cbr(48, width), nn.MaxPool2d((2, 1)),
            _cbr(width, width, groups=width), nn.MaxPool2d((2, 1)),
            nn.Conv2d(width, width, (2, 1)), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )  # fmt: skip
        self.context = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(width, width, 5, padding=2), nn.ReLU(inplace=True),
        )  # fmt: skip
        self.fc = nn.Linear(width, config.NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        seq = self.context(self.cnn(x).squeeze(2))  # [B, width, 32]
        return self.fc(seq.transpose(1, 2))


ARCHS = {"crnn": CRNN, "conv_ctc": ConvCTC}
