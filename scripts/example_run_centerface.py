"""
Example of running centerface models and displaying results
"""

import time

import numpy as np
from PIL import Image as img

import scripts.centerface_utils as centerface_utils


def get_image(path, target_width, target_height):
    target_width = 640
    target_height = 640

    test_image = img.open(path)
    width, height = test_image.size  # Get dimensions

    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2

    # Crop the center of the image
    test_image = test_image.crop((left, top, right, bottom))
    test_image = test_image.resize((target_width, target_height))

    return test_image


def get_onnxrt_detection(path, target_width, target_height, threshold=0.4):
    target_width = 640
    target_height = 640
    test_image = get_image(path, target_width, target_height)

    # Default array ordering is HWC, we want CHW, expand_dims to add back batch dim
    np_image = np.expand_dims(np.array(test_image).transpose(2, 0, 1), 0)
    np_image = np_image.astype("float32")

    onnx_runner = centerface_utils.CenterFaceOnnx("models/centerface-optimized.onnx")

    start = time.time()
    detections, landmarks = onnx_runner(
        np_image, target_height, target_width, threshold=threshold
    )
    print(time.time() - start)

    centerface_utils.draw_detection(test_image, detections, landmarks)
    test_image.show()


def get_tvm_detection(path, target_width, target_height, use_fp16=False, threshold=0.4):
    target_width = 640
    target_height = 640
    test_image = get_image(path, target_width, target_height)

    # Default array ordering is HWC, we want CHW, expand_dims to add back batch dim
    np_image = np.expand_dims(np.array(test_image).transpose(2, 0, 1), 0)
    np_image = np_image.astype("float32")
    tvm_runner = centerface_utils.CenterFaceTVM(
        package_path="compiled_packages/centerface_autoscheduler_30000kt_fp16_llvm.tar"
        if use_fp16
        else "compiled_packages/centerface_autoscheduler_30000kt_fp32_llvm.tar"
    )

    start = time.time()
    detections, landmarks = tvm_runner(
        np_image, target_height, target_width, threshold=threshold
    )
    print(time.time() - start)

    centerface_utils.draw_detection(test_image, detections, landmarks)
    test_image.show()


if __name__ == "__main__":
    get_tvm_detection("crowd_of_people.jpeg", 640, 640, use_fp16=False)
    get_tvm_detection("crowd_of_people.jpeg", 640, 640, use_fp16=True)
    get_onnxrt_detection("crowd_of_people.jpeg", 640, 640)
