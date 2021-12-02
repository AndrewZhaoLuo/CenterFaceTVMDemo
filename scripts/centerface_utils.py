from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from tvm import rpc
from tvm.contrib import graph_executor as runtime
from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCPackage

"""
Centerface processing utilities, copied mostly from 

https://github.com/Star-Clouds/CenterFace/blob/master/prj-python/centerface.py
"""


def draw_detection(img: Image, detections: list, landmarks: list):
    drawer = ImageDraw.Draw(img)
    for detection in detections:
        boxes, _ = detection[:4], detection[4]
        drawer.rectangle([int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])])

    tic_size = 2
    for landmark in landmarks:
        for i in range(0, 5):
            center_x = int(landmark[i * 2])
            center_y = int(landmark[i * 2 + 1])
            drawer.line(
                [
                    center_x - tic_size,
                    center_y - tic_size,
                    center_x + tic_size,
                    center_y + tic_size,
                ]
            )
            drawer.line(
                [
                    center_x - tic_size,
                    center_y + tic_size,
                    center_x + tic_size,
                    center_y - tic_size,
                ]
            )


class CenterFaceBaseObject(object):
    def __init__(self):
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0

    def __call__(self, img: np.array, height: int, width: int, threshold: float = 0.5):
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self.transform(
            height, width
        )
        return self.inference(img, threshold)

    def inference(self, img: np.array, threshold: float) -> list:
        raise NotImplementedError()

    def transform(self, h: int, w: int) -> Tuple[int, int, float, float]:
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def postprocess(
        self,
        heatmap: np.array,
        lms: np.array,
        offset: np.array,
        scale: np.array,
        threshold: float,
    ) -> Tuple[np.array, np.array]:
        dets, lms = self.decode(
            heatmap,
            scale,
            offset,
            lms,
            (self.img_h_new, self.img_w_new),
            threshold=threshold,
        )

        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = (
                dets[:, 0:4:2] / self.scale_w,
                dets[:, 1:4:2] / self.scale_h,
            )
            lms[:, 0:10:2], lms[:, 1:10:2] = (
                lms[:, 0:10:2] / self.scale_w,
                lms[:, 1:10:2] / self.scale_h,
            )
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            lms = np.empty(shape=[0, 10], dtype=np.float32)

        return dets, lms

    def decode(
        self,
        heatmap: np.array,
        scale: np.array,
        offset: np.array,
        landmark: np.array,
        size: Tuple[int, int],
        threshold: float = 0.1,
    ) -> Tuple[list, list]:
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        boxes, lms = [], []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = (
                    np.exp(scale0[c0[i], c1[i]]) * 4,
                    np.exp(scale1[c0[i], c1[i]]) * 4,
                )
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(
                    0, (c0[i] + o0 + 0.5) * 4 - s0 / 2
                )
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                lm = []
                for j in range(5):
                    lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                    lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            lms = np.asarray(lms, dtype=np.float32)
            lms = lms[keep, :]
        return boxes, lms

    def nms(self, boxes: list, scores: list, nms_thresh: float) -> list:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.bool)

        keep = []
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)

            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True

        return keep


class CenterFaceOnnx(CenterFaceBaseObject):
    def __init__(self, model_path: str):
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = 0, 0, 0, 0

        # Turn off logging
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, sess_options=so)

    def inference(self, img: np.array, threshold: float) -> list:
        result = self.session.run(None, {"input.1": img})
        heatmap, scale, offset, lms = result
        return self.postprocess(heatmap, lms, offset, scale, threshold)


class CenterFaceTVM(CenterFaceBaseObject):
    def __init__(
        self,
        package_path="compiled_packages/centerface_autoscheduler_30000kt_fp32_llvm.tar",
    ):
        session = rpc.LocalSession()
        package = TVMCPackage(package_path)
        session.upload(package.lib_path)
        lib = session.load_module(package.lib_name)
        dev = session.cpu()

        self.module = runtime.create(package.graph, lib, dev)
        self.module.load_params(package.params)

    def inference(self, img: np.array, threshold: float) -> list:
        input_dict = {"input.1": img}
        self.module.set_input(**input_dict)
        self.module.run()
        heatmap, scale, offset, lms = [
            self.module.get_output(i).numpy()
            for i in range(self.module.get_num_outputs())
        ]
        return self.postprocess(heatmap, lms, offset, scale, threshold)


class CenterFaceNoDetection(CenterFaceBaseObject):
    def inference(self, img: np.array, threshold: float) -> list:
        dets = np.empty(shape=[0, 5], dtype=np.float32)
        lms = np.empty(shape=[0, 10], dtype=np.float32)
        return dets, lms
