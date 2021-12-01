import multiprocessing as mp
from os import path
from shutil import copyfile

import numpy as np
import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCModel
from tvm.relay.transform import InferType, ToMixedPrecision

"""Copy pasted mostly from:

https://github.com/AndrewZhaoLuo/TVM-Sandbox/blob/bb209e8845440ed9f40af1b2580618196c939745/fp16_pass/benchmark_fp16.py#L1

Creates centerface autoscheduler log files, which are included in this repo so you 
don't have to spend 24 hrs running the tuning script!
"""


def load_model(name, **kwargs):
    return tvmc.load(path.join("./", name), **kwargs)


def graph_optimize(
    tvmc_model, run_fp16_pass, run_other_opts, try_nhwc_layout=False, fast_math=True
):
    mod, params = tvmc_model.mod, tvmc_model.params
    mod = tvm.IRModule.from_expr(mod["main"])

    # nhwc is typically better for autoscheduler -- more schedules available
    # also winograd is only available for nhwc
    if try_nhwc_layout:
        desired_layouts = {
            "nn.conv2d": ["NHWC", "default"],
            "nn.conv2d_transpose": ["NHWC", "default"],
            "nn.upsampling": ["NHWC", "default"],
            "image.resize2d": ["NHWC", "default"],
            "vision.roi_align": ["NHWC", "default"],
        }
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            mod = relay.transform.InferType()(mod)
            mod = relay.transform.ConvertLayout(desired_layouts)(mod)
            mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
            mod = tvm.relay.transform.FoldConstant()(mod)

    if run_other_opts:
        mod = tvm.relay.transform.FastMath()(mod) if fast_math else mod
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        BindPass = tvm.relay.transform.function_pass(
            lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(
                fn, params
            ),
            opt_level=1,
        )
        mod = BindPass(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)

    if run_fp16_pass:
        mod = InferType()(mod)
        mod = ToMixedPrecision()(mod)

    if run_other_opts and run_fp16_pass:
        # run one more pass to clean up new subgraph
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.FastMath()(mod) if fast_math else mod

    print("Final relay mod:")
    print(mod)
    return TVMCModel(mod, params)


def benchmark_model(
    model_func,
    name,
    run_fp16_pass=True,
    run_other_opts=True,
    enable_autoscheduler=False,
    try_nhwc_layout=False,
    target="llvm",
    target_host="llvm",
    tuning_trials=10000,
    tuning_repeat_trials=5,
    measure_number=100,
    measure_repeats=10,
):
    print("*" * 30, name, "*" * 30)
    print("FP16 pass" if run_fp16_pass else "FP32 pass")
    """Get Module"""
    tvmc_model = model_func(
        run_pass=run_fp16_pass, run_opts=run_other_opts, try_nhwc_layout=try_nhwc_layout
    )
    tuning_records = tvmc.tune(
        tvmc_model,
        target=target,
        enable_autoscheduler=enable_autoscheduler,
        trials=tuning_trials,
        repeat=tuning_repeat_trials,
        tuner="xgb_knob",
        target_host=target_host,
    )

    copyfile(tuning_records, f"./{name}")

    # Create package artifacts
    package = tvmc.compile(tvmc_model, target=target, tuning_records=tuning_records)
    result = tvmc.run(
        package,
        device="cpu" if "llvm" in target else target,
        repeat=measure_number,
        number=measure_repeats,
    )
    print(result)
    print()


def get_centerface(run_pass=True, run_opts=True, try_nhwc_layout=False):
    tvmc_model = load_model("centerface.onnx")
    return graph_optimize(
        tvmc_model, run_pass, run_opts, try_nhwc_layout=try_nhwc_layout
    )


if __name__ == "__main__":
    benchmark_model(
        get_centerface,
        "centerface_autoscheduler_10000kt_fp16_llvm",
        run_fp16_pass=True,
        run_other_opts=True,
        enable_autoscheduler=True,
        try_nhwc_layout=True,
        tuning_trials=10000,
        target="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos",
        target_host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos",
    )
    benchmark_model(
        get_centerface,
        "centerface_autoscheduler_30000kt_fp32_llvm",
        run_fp16_pass=False,
        run_other_opts=True,
        enable_autoscheduler=True,
        try_nhwc_layout=True,
        tuning_trials=30000,
        target="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos",
        target_host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos",
    )
