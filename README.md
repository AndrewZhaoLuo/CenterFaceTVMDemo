![DemoClip](https://user-images.githubusercontent.com/13855451/145470544-cf91c00e-0f6c-4c76-bddd-f8493fdc8fb6.gif)

# CenterFaceTVMDemo
A demo for using and deploying CenterFace model using TVM.

TVM commit used: 18c88789a70489d6615234f25bcc4f29e9dd83dc

Guide:

`compiled_packages` contains compiled TVM packages for m1 macbooks. These are ready to be run if you have the relevant hardware.

`models` contains the original `centerface` onnx models. The `centerface-optimized.onnx` is the model run through the `onnxoptimizer` tool.

`scripts` contains scripts you can run:

Run `example_tvm_tune.py` for tuning the model and applying fp16 optimizations.

Run `example_run_centerface.py` for an example of running models on different execution providers (onnxruntime, TVM w/ FP32 models, and TVM w/ FP16 models)

Run `python_demo.py` to run a model real time off the webcam.

`python_demo.py` runs the python demo if you have the proper hardware.

Note, you need an M1 macbook pro (mine is a 2020 model) to run everything properly, but even so this is a valuable example on using FP16 quantization in TVM.
