import queue
import time
from threading import Thread

import cv2

from scripts import centerface_utils

TARGET_WIDTH = 640
TARGET_HEIGHT = 640
TARGET_FPS = 30


class CameraDemo:
    """Multi-threaded python centerface detection demo."""

    def __init__(self, runner: centerface_utils.CenterFaceNoDetection) -> None:
        self.keep_going = True
        self.runner = runner

    def capture_frame(self, cap, queue):
        """Thread function which captures data from webcam and places into queue"""
        prev = 0
        cur = 0
        while self.keep_going:
            cur = time.time()
            _, img = cap.read()
            if (cur - prev) >= 1.0 / TARGET_FPS:
                prev = cur
                queue.put(img)

    def process_frame(
        self, runner, processing_func, input_queue, output_queue, threshold
    ):
        """Thread function which detects and overlays results, add it to queue for rendering"""
        while self.keep_going:
            if input_queue.empty():
                continue
            frame = input_queue.get()
            frame = processing_func(frame)

            np_array = cv2.dnn.blobFromImage(
                frame,
                scalefactor=1.0,
                size=(TARGET_WIDTH, TARGET_HEIGHT),
                mean=(0, 0, 0),
                swapRB=True,
                crop=True,
            )
            start = time.time()
            detections, landmarks = runner(
                np_array, TARGET_HEIGHT, TARGET_WIDTH, threshold=threshold
            )
            end = time.time()
            print(f"Processing frame too {(end - start) * 1000} ms")

            # Draw predictions and show frame
            for det in detections:
                boxes, _ = det[:4], det[4]
                cv2.rectangle(
                    frame,
                    (int(boxes[0]), int(boxes[1])),
                    (int(boxes[2]), int(boxes[3])),
                    (2, 255, 0),
                    3,
                )
            for lm in landmarks:
                for i in range(0, 5):
                    cv2.circle(
                        frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 4, (0, 0, 255), -1
                    )

            output_queue.put(frame)

    def run(self, threshold=0.5):
        cap = cv2.VideoCapture(0)

        cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Doesn't seem to do anything :/
        # cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        print("* Capture width:", cap_width)
        print("* Capture height:", cap_height)
        print("* Capture FPS:", cap_fps)

        _, frame = cap.read()

        # assume w > h
        h, w = frame.shape[:2]
        scale = TARGET_WIDTH / h
        new_width = int(scale * w)
        new_height = int(scale * h)

        # For centercrop
        left = (new_width - TARGET_WIDTH) // 2
        top = (new_height - TARGET_HEIGHT) // 2
        right = (new_width + TARGET_WIDTH) // 2
        bottom = (new_height + TARGET_HEIGHT) // 2

        # initial queue for webcam data
        frames_queue = queue.Queue(maxsize=0)

        # queue after we've streamed it to real-time feed
        ready_for_processing_queue = queue.Queue(maxsize=0)

        # queue for processed frames with prediction overlays
        processed_frames_queue = queue.Queue(maxsize=0)

        # start thread to capture data from webcam
        capture_thread = Thread(
            target=self.capture_frame,
            args=(
                cap,
                frames_queue,
            ),
            daemon=True,
        )
        capture_thread.start()

        def processing_func(cv2_frame):
            # Resize and center crop frame
            frame = cv2.resize(cv2_frame, (new_width, new_height))
            frame = frame[top:bottom, left:right]
            return frame

        # start thread to process images with model
        processing_thread = Thread(
            target=self.process_frame,
            args=(
                self.runner,
                processing_func,
                ready_for_processing_queue,
                processed_frames_queue,
                threshold,
            ),
            daemon=True,
        )
        processing_thread.start()

        while self.keep_going:
            if not frames_queue.empty():
                img_real_time = frames_queue.get()
                if img_real_time is not None:
                    cv2.imshow("realtime", img_real_time)
                    ready_for_processing_queue.put(img_real_time)

            if not processed_frames_queue.empty():
                img_processed = processed_frames_queue.get()
                if img_processed is not None:
                    cv2.imshow("predicted", img_processed)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.keep_going = False
                break

        cap.release()
        capture_thread.join()
        processing_thread.join()


if __name__ == "__main__":
    onnx_runner = centerface_utils.CenterFaceOnnx("models/centerface-optimized.onnx")
    tvm_runner_fp32 = centerface_utils.CenterFaceTVM(
        "compiled_packages/centerface_autoscheduler_30000kt_fp32_llvm.tar"
    )
    tvm_runner_fp16 = centerface_utils.CenterFaceTVM(
        "compiled_packages/centerface_autoscheduler_30000kt_fp16_llvm.tar"
    )
    dummy_runner = centerface_utils.CenterFaceNoDetection()

    # Change runners at will
    demo = CameraDemo(tvm_runner_fp16)

    demo.run()
