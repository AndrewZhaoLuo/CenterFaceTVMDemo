"""
Example of running ONNX centerface model with onnxruntime
"""

import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from numpy.testing._private.utils import break_cycles
from PIL import Image as img
from tvm import relay

"""Model exports from https://github.com/linghu8812/tensorrt_inference"""

# Try loading into tvm
onnx_model = onnx.load("./centerface.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)

new_width = 640
new_height = 640

test_image = img.open("./crowd_of_people.jpeg")
width, height = test_image.size  # Get dimensions

left = (width - new_width) / 2
top = (height - new_height) / 2
right = (width + new_width) / 2
bottom = (height + new_height) / 2

# Crop the center of the image
test_image = test_image.crop((left, top, right, bottom))

# Default array ordering is HWC, we want CHW, expand_dims to add back batch dim
np_image = np.expand_dims(np.array(test_image).transpose(2, 0, 1), 0)

# preprocessing of image goes here
np_image = (np_image / 255.0).astype("float32")


import time

time_in_ms = []
for i in range(10):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    session = ort.InferenceSession("centerface.onnx", sess_options=so)

    start = time.time()
    result = session.run(None, {"input.1": np_image})
    end = time.time()

    print(result)
    time_in_ms.append((end - start) * 1000)

print(time_in_ms)
