# CenterFaceTVMDemo
A demo for using CenterFace in TVM.

TVM commit used: 18c88789a70489d6615234f25bcc4f29e9dd83dc

Guide:

Run `example_tvm_tune.py` for tuning the model and applying fp16 optimizations.

Run `example_run_centerface.py` for an example of running models on different execution providers (onnxruntime, TVM w/ FP32 models, and TVM w/ FP16 models)

Run `python_demo.py` to run a model real time off the webcam.

Note, you need an M1 macbook pro (mine is a 2020 model) to run this properly, but even so this is a valuable example on using FP16 quantization in TVM.