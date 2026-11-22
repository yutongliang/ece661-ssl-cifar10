#!/bin/bash
# Source from a ResNet-50 job: print the GPU and refuse anything under 22 GiB.
"$python_path" - <<'PY' || exit 1
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
props = torch.cuda.get_device_properties(0)
gib = props.total_memory / 2**30
print(f"GPU: {props.name} | {gib:.1f} GiB")
assert gib >= 22, "GPU has less memory than the 24 GB plan requires"
PY
