import torch
import torch.nn as nn
import torch.onnx


class RideDurationTorchModel(nn.Module):
    """PyTorch version of the ride-duration heuristic.

    A single linear layer over ``[distance_km, passengers]``. Weights are
    initialized to reproduce ``src.model.RideDurationModel``:
    ``duration = distance / 0.5 + passengers * 0.5 = 2*distance + 0.5*passengers``.
    Swap in a real trained model when available.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[2.0, 0.5]]))
            self.linear.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


model = RideDurationTorchModel()
model.eval()  # disable dropout/batchnorm training mode

# Dummy input only defines the input SHAPE — values don't matter
dummy_input = torch.randn(1, 2)  # [distance_km, passengers]

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,  # bundle trained weights into the file
    opset_version=17,  # ONNX operator set version
    input_names=["features"],
    output_names=["duration"],
    dynamic_axes={  # allow variable batch size at inference
        "features": {0: "batch_size"},
        "duration": {0: "batch_size"},
    },
)

# ── Always validate the export ─────────────────────
import onnx  # noqa: E402  (imported at point of use for teaching clarity)

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)  # raises if graph is invalid
print("Inputs: ", [i.name for i in onnx_model.graph.input])
print("Outputs:", [o.name for o in onnx_model.graph.output])
