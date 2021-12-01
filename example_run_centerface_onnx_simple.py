import numpy as np
import onnxruntime as ort
from PIL import Image as img

"""
Example of running ONNX centerface model with onnxruntime

centerface.onnx exports from https://github.com/linghu8812/tensorrt_inference
centerface-optimized.onnx is gotten by applying 
    https://github.com/AndrewZhaoLuo/TVM-Sandbox/blob/f1f9f698be2b7a8cc5bcf1167d892cd915eb7ce7/onnx/onnx_optimizer.py#L1
    on centerface.onnx
"""

if __name__ == "__main__":
    new_width = 640
    new_height = 640

    test_image = img.open("crowd_of_people.jpeg")
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

    # Turn off logging
    so = ort.SessionOptions()
    so.log_severity_level = 3

    session = ort.InferenceSession("models/centerface-optimized.onnx", sess_options=so)
    result = session.run(None, {"input.1": np_image})

    print("Benchmarking onnxruntime:")

    import time

    time_in_ms = []
    for i in range(100):
        session = ort.InferenceSession("models/centerface.onnx", sess_options=so)

        start = time.time()
        result = session.run(None, {"input.1": np_image})
        end = time.time()

        time_in_ms.append((end - start) * 1000)

    # avg_time: 61.13584756851196 ms
    print(f"avg_time: {sum(time_in_ms) / len(time_in_ms)} ms")
