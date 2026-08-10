"""
LEVEL 4, step 1 — put the model in a versioned model store.

This is the thing Level 1 could not do. There, MODEL_VERSION was a string
constant in the source file: shipping v2 meant restarting the process, rolling
back meant redeploying, and running v1 and v2 side by side for a canary meant
operating two services by hand.

Here the model is an immutable, content-addressed entry in a store. Run this
twice and you get two versions; the service resolves "resnet50:latest", and
pinning a specific tag is a one-line change.

Run:  python level_4_save_model.py
      bentoml models list
"""

import bentoml
import torch
from torchvision.models import ResNet50_Weights, resnet50

model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()

# We store the state_dict, not the pickled nn.Module. bentoml.pytorch.save_model
# would pickle the whole object, and torch>=2.6 refuses to unpickle that under
# its weights_only=True default -- a real failure you will hit in the wild.
# Weights in, architecture in code, is both safer and the portable option.
with bentoml.models.create("resnet50") as ref:
    torch.save(model.state_dict(), ref.path_of("model.pt"))

print(f"saved {ref.tag}")
print("versions now in the store:")
for m in bentoml.models.list("resnet50"):
    print(f"  {m.tag}")
