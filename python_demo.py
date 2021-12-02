import time

import cv2

from scripts import centerface_utils

TARGET_WIDTH = 640
TARGET_HEIGHT = 640


def camera(runner, threshold=0.5):
    cap = cv2.VideoCapture(0)
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

    # For displaying FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    prev_frame_time = 0
    cur_frame_time = 0

    avg_fps = [0] * 10
    cur_frame = 0

    while True:
        _, frame = cap.read()

        # Resize and center crop frame
        frame = cv2.resize(frame, (new_width, new_height))
        frame = frame[top:bottom, left:right]

        # Run model
        np_array = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=(TARGET_WIDTH, TARGET_HEIGHT),
            mean=(0, 0, 0),
            swapRB=True,
            crop=True,
        )
        detections, landmarks = runner(
            np_array, TARGET_HEIGHT, TARGET_WIDTH, threshold=threshold
        )

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

        # calculate fps
        cur_frame_time = time.time()
        point_fps = 1 / (cur_frame_time - prev_frame_time)
        prev_frame_time = cur_frame_time
        avg_fps[cur_frame % len(avg_fps)] = point_fps
        fps = int(sum(avg_fps) / min(len(avg_fps), cur_frame + 1))

        cv2.putText(frame, str(fps), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("out", frame)

        cur_frame += 1
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


if __name__ == "__main__":
    onnx_runner = centerface_utils.CenterFaceOnnx("models/centerface-optimized.onnx")
    tvm_runner_fp32 = centerface_utils.CenterFaceTVM(
        "compiled_packages/centerface_autoscheduler_30000kt_fp32_llvm.tar"
    )
    tvm_runner_fp16 = centerface_utils.CenterFaceTVM(
        "compiled_packages/centerface_autoscheduler_30000kt_fp16_llvm.tar"
    )
    dummy_runner = centerface_utils.CenterFaceNoDetection()

    camera(onnx_runner)
