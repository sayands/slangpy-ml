# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Third-party imports
import numpy as np
from slangpy.backend import Device
from slangpy.types import NDBuffer

def create_uv_grid(device: Device, resolution: int):
    span = np.linspace(0, 1, resolution, dtype=np.float32)
    uvs_np = np.stack(np.broadcast_arrays(span[None, :], span[:, None]), axis=2)
    uvs = NDBuffer(device, 'float2', shape=(resolution, resolution))
    uvs.copy_from_numpy(uvs_np)
    return uvs