from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "KernelBench" / "KernelBench"
RESULT_ROOT = ROOT.parent / "benchmarks" / "ml" / "KernelBench"


DEFAULT_OVERRIDES = {
    "batch_size": 2,
    "N": 8,
    "M": 7,
    "K": 6,
    "L": 5,
    "b": 2,
    "i": 3,
    "j": 4,
    "k": 5,
    "l": 6,
    "m": 4,
    "n": 5,
    "dim": 1,
    "dim1": 4,
    "dim2": 5,
    "features": 4,
    "num_features": 4,
    "input_shape": (8,),
    "sequence_length": 8,
    "length": 8,
    "input_length": 8,
    "height": 16,
    "width": 16,
    "height_in": 16,
    "width_in": 16,
    "depth": 6,
    "depth_in": 6,
    "D": 5,
    "H": 8,
    "W": 8,
    "in_channels": 2,
    "out_channels": 4,
    "channels": 2,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "dilation": 1,
    "output_padding": 0,
    "in_features": 8,
    "out_features": 6,
    "num_classes": 8,
    "input_size": 8,
    "hidden_size": 8,
    "output_size": 6,
    "num_layers": 2,
    "num_input_features": 4,
    "growth_rate": 4,
}


PER_CASE_OVERRIDES = {
    ("level1", "1_Square_matrix_multiplication_.py"): {"N": 8},
    ("level1", "3_Batched_matrix_multiplication.py"): {"batch_size": 2, "m": 4, "k": 5, "n": 6},
    ("level1", "10_3D_tensor_matrix_multiplication.py"): {"N": 2, "M": 3, "K": 4, "L": 5},
    ("level1", "11_4D_tensor_matrix_multiplication.py"): {"b": 2, "i": 3, "j": 4, "l": 5, "k": 6},
    ("level1", "12_Matmul_with_diagonal_matrices_.py"): {"N": 6, "M": 5},
    ("level1", "16_Matmul_with_transposed_A.py"): {"M": 4, "K": 5, "N": 4},
    ("level1", "17_Matmul_with_transposed_B.py"): {"M": 4, "K": 5, "N": 4},
    ("level1", "18_Matmul_with_transposed_both.py"): {"M": 4, "K": 5, "N": 4},
    ("level1", "33_BatchNorm.py"): {"features": 4, "dim1": 5, "dim2": 6},
    ("level1", "35_GroupNorm_.py"): {"features": 4, "num_groups": 2, "dim1": 5, "dim2": 6},
    ("level1", "40_LayerNorm.py"): {"normalized_shape": (4, 5, 6), "features": 4, "dim1": 5, "dim2": 6},
    ("level1", "72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_.py"): {
        "in_channels": 4,
        "out_channels": 4,
        "groups": 4,
        "depth": 4,
        "height": 5,
        "width": 6,
    },
    ("level1", "75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__.py"): {
        "in_channels": 4,
        "out_channels": 8,
        "groups": 4,
        "height": 6,
        "width": 7,
    },
    ("level1", "82_conv_depthwise_2D_square_input_square_kernel.py"): {"in_channels": 4, "out_channels": 4, "groups": 4},
    ("level1", "83_conv_depthwise_2D_square_input_asymmetric_kernel.py"): {"in_channels": 4, "out_channels": 4, "groups": 4},
    ("level1", "84_conv_depthwise_2D_asymmetric_input_square_kernel.py"): {"in_channels": 4, "out_channels": 4, "groups": 4},
    ("level1", "85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel.py"): {"in_channels": 4, "out_channels": 4, "groups": 4},
    ("level1", "97_ScaledDotProductAttention.py"): {
        "batch_size": 2,
        "num_heads": 2,
        "sequence_length": 8,
        "embedding_dimension": 8,
    },
    ("level1", "100_HingeLoss.py"): {"batch_size": 2, "input_shape": (2,)},
    ("level2", "21_Conv2d_Add_Scale_Sigmoid_GroupNorm.py"): {"in_channels": 2, "out_channels": 8, "num_groups": 4},
    ("level2", "3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU.py"): {"in_channels": 2, "out_channels": 12, "norm_shape": (8,)},
    ("level2", "27_Conv3d_HardSwish_GroupNorm_Mean.py"): {"in_channels": 2, "out_channels": 4},
    ("level2", "23_Conv3d_GroupNorm_Mean.py"): {"in_channels": 2, "out_channels": 8, "num_groups": 4},
    ("level2", "34_ConvTranspose3d_LayerNorm_GELU_Scaling.py"): {"in_channels": 2, "out_channels": 6},
    ("level2", "60_ConvTranspose3d_Swish_GroupNorm_HardSwish.py"): {"in_channels": 2, "out_channels": 4, "groups": 4},
    ("level2", "92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp.py"): {
        "in_channels": 2,
        "out_channels": 16,
        "groups": 8,
    },
    ("level2", "98_Matmul_AvgPool_GELU_Scale_Max.py"): {"pool_kernel_size": 2},
}


PER_CASE_INPUTS = {
    ("level1", "50_conv_standard_2D__square_input__square_kernel.py"): lambda: [torch.rand(1, 3, 16, 16)],
}


RUNTIME_VALUE_OVERRIDES = {
    ("level1", "63_conv_standard_2D__square_input__square_kernel.py"): {
        "x": np.zeros((1, 3, 16, 16), dtype=np.float32),
        "conv1_weight": np.zeros((96, 3, 11, 11), dtype=np.float32),
        "conv1_bias": np.zeros((96,), dtype=np.float32),
        "out": np.zeros((1, 96, 3, 3), dtype=np.float32),
        "batch_size": 1,
    },
}


class Timeout(Exception):
    pass


def alarm_handler(_signum, _frame):
    raise Timeout()


def numeric_key(path: Path) -> tuple[int, str]:
    match = re.match(r"(\d+)_", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def result_stem(source: Path | str) -> str:
    stem = source.stem if isinstance(source, Path) else source
    match = re.match(r"(\d+)_(.*)", stem)
    if not match:
        return re.sub(r"_+", "_", stem).strip("_")
    number, name = match.groups()
    clean = re.sub(r"_+", "_", name).strip("_")
    return f"{number}_{clean}"


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def patch_sizes(module, level: str, filename: str):
    for key, value in DEFAULT_OVERRIDES.items():
        if hasattr(module, key):
            setattr(module, key, value)
    for key, value in PER_CASE_OVERRIDES.get((level, filename), {}).items():
        setattr(module, key, value)
    if hasattr(module, "num_groups") and hasattr(module, "out_channels"):
        num_groups = getattr(module, "num_groups")
        if isinstance(num_groups, int) and getattr(module, "out_channels") % num_groups != 0:
            setattr(module, "out_channels", num_groups)
    if hasattr(module, "num_groups") and hasattr(module, "out_features"):
        num_groups = getattr(module, "num_groups")
        if isinstance(num_groups, int) and getattr(module, "out_features") % num_groups != 0:
            setattr(module, "out_features", num_groups)
    if hasattr(module, "groups"):
        groups = getattr(module, "groups")
        if isinstance(groups, int) and groups > 1:
            if hasattr(module, "in_channels") and getattr(module, "in_channels") % groups != 0:
                setattr(module, "in_channels", groups)
            if hasattr(module, "out_channels") and getattr(module, "out_channels") % groups != 0:
                setattr(module, "out_channels", groups)


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return type(value)(to_numpy(v) for v in value)
    return value


def compare_outputs(a, b, rtol: float, atol: float):
    a_np = to_numpy(a)
    b_np = to_numpy(b)
    if isinstance(a_np, (list, tuple)):
        if len(a_np) != len(b_np):
            return False, f"tuple length mismatch {len(a_np)} != {len(b_np)}"
        for left, right in zip(a_np, b_np):
            ok, detail = compare_outputs(left, right, rtol, atol)
            if not ok:
                return ok, detail
        return True, ""
    a_np = np.asarray(a_np)
    b_np = np.asarray(b_np)
    if a_np.shape != b_np.shape:
        return False, f"shape mismatch torch={a_np.shape} numpy={b_np.shape}"
    if not np.allclose(a_np, b_np, rtol=rtol, atol=atol, equal_nan=True):
        diff = np.nanmax(np.abs(a_np - b_np))
        return False, f"max_abs_diff={float(diff)}"
    return True, ""


def public_func_name(module, stem: str, manifest_path: Path) -> str:
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.startswith("func_name:"):
                candidate = line.split(":", 1)[1].strip()
                if hasattr(module, candidate):
                    return candidate
    if hasattr(module, "forward"):
        return "forward"
    callables = [
        name for name, value in vars(module).items()
        if callable(value) and not name.startswith("_") and name not in {"np", "numpy"}
    ]
    if stem in callables:
        return stem
    if len(callables) == 1:
        return callables[0]
    raise RuntimeError(f"cannot identify numpy function in {module.__name__}")


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "func_name": None,
        "parameters": {},
        "arrays": {},
        "scalars": {},
        "output_args": [],
    }
    if not manifest_path.exists():
        return manifest

    section = None
    size = None
    init_section = None
    pending_array = None
    logical_lines = []
    pending = ""
    balance = 0
    for raw in manifest_path.read_text().splitlines():
        text = raw.split("#", 1)[0].rstrip()
        if pending:
            pending += " " + text.strip()
        else:
            pending = text
        balance += text.count("(") - text.count(")")
        if balance <= 0:
            logical_lines.append(pending)
            pending = ""
            balance = 0
    if pending:
        logical_lines.append(pending)

    for raw in logical_lines:
        line = raw.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if section == "output_args" and stripped.startswith("- "):
            manifest["output_args"].append(stripped[2:].strip())
            continue
        if indent == 0:
            section = None
            size = None
            init_section = None
            pending_array = None
            if stripped.startswith("func_name:"):
                manifest["func_name"] = stripped.split(":", 1)[1].strip()
            elif stripped == "parameters:":
                section = "parameters"
            elif stripped == "init:":
                section = "init"
            elif stripped == "output_args:":
                section = "output_args"
            continue
        if section == "parameters":
            if indent == 2 and stripped.endswith(":"):
                size = stripped[:-1]
                manifest["parameters"][size] = {}
            elif indent >= 4 and size and ":" in stripped:
                key, value = stripped.split(":", 1)
                manifest["parameters"][size][key.strip()] = _parse_scalar(value)
        elif section == "init":
            if indent == 2 and stripped.endswith(":"):
                init_section = stripped[:-1]
                pending_array = None
            elif indent >= 6 and init_section == "arrays" and pending_array and ":" in stripped:
                key, value = stripped.split(":", 1)
                if key.strip() == "shape":
                    manifest["arrays"][pending_array] = value.strip()
            elif indent >= 4 and init_section in {"arrays", "scalars"} and ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                if init_section == "arrays":
                    if value.strip():
                        manifest["arrays"][key] = value.strip()
                        pending_array = key
                    else:
                        pending_array = key
                else:
                    manifest["scalars"][key] = _parse_scalar(value)
    return manifest


def output_arg_names(manifest_path: Path) -> list[str]:
    return list(read_manifest(manifest_path)["output_args"])


def _shape_from_expr(expr: str, values: dict[str, Any]) -> tuple[int, ...]:
    text = expr.strip()
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        text = ast.literal_eval(text)
    scope = dict(values)
    result = eval(text, {"__builtins__": {}}, scope)
    if isinstance(result, int):
        return (result,)
    return tuple(int(v) for v in result)


def manifest_values(manifest_path: Path, size: str = "S") -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    values: dict[str, Any] = {}
    values.update(manifest["parameters"].get(size, {}))
    values.update(manifest["scalars"])
    for name, shape_expr in manifest["arrays"].items():
        try:
            shape = _shape_from_expr(shape_expr, values)
        except Exception:
            continue
        values[name] = np.zeros(shape, dtype=np.float32)
    return values


def source_arg_names(source_module) -> tuple[list[str], list[str]]:
    init = inspect.signature(source_module.Model.__init__)
    forward = inspect.signature(source_module.Model.forward)
    init_names = [name for name in init.parameters if name != "self"]
    forward_names = [name for name in forward.parameters if name != "self"]
    return init_names, forward_names


def state_values(torch_model) -> dict[str, Any]:
    values = {}
    for key, tensor in torch_model.state_dict().items():
        values[key.replace(".", "_")] = tensor.detach().cpu().numpy()
    return values


def run_buffer_function(func, values: dict[str, Any], torch_out, manifest_path: Path):
    output_names = output_arg_names(manifest_path)
    if output_names:
        if torch_out is not None:
            outputs = torch_out if isinstance(torch_out, (tuple, list)) else (torch_out,)
            for name, value in zip(output_names, outputs):
                values[name] = np.empty_like(to_numpy(value))
    kwargs = {}
    missing = []
    for name, param in inspect.signature(func).parameters.items():
        if name in values:
            kwargs[name] = values[name]
        elif param.default is inspect.Signature.empty:
            missing.append(name)
    if missing:
        raise RuntimeError(f"missing numpy args: {', '.join(missing)}")
    returned = func(**kwargs)
    if output_names:
        outs = tuple(values[name] for name in output_names)
        return outs[0] if len(outs) == 1 else outs
    return returned


def copy_state_to_module(torch_model, numpy_module):
    for name, value in state_values(torch_model).items():
        if hasattr(numpy_module, f"{name}_value"):
            setattr(numpy_module, f"{name}_value", value)
        elif hasattr(numpy_module, name):
            setattr(numpy_module, name, value)


def copy_source_globals(source_module, numpy_module):
    allowed = (int, float, bool, str, tuple)
    for name, value in vars(source_module).items():
        if name.startswith("_") or inspect.ismodule(value) or inspect.isclass(value) or callable(value):
            continue
        if isinstance(value, allowed):
            setattr(numpy_module, name, value)


def run_runtime_case(level: str, filename: str, timeout: int):
    source_path = SOURCE_ROOT / level / filename
    stem = result_stem(source_path)
    result_path = RESULT_ROOT / level / stem / f"{stem}_numpy.py"
    manifest_path = RESULT_ROOT / level / stem / f"{stem}.yaml"
    result = import_module(result_path, f"runtime_numpy_{level}_{stem}")
    if hasattr(result, "TRANSLATION_ERROR"):
        return {"status": "translate_error", "detail": repr(result.TRANSLATION_ERROR)}

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout)
    started = time.perf_counter()
    try:
        manifest = read_manifest(manifest_path)
        values = manifest_values(manifest_path)
        values.update(RUNTIME_VALUE_OVERRIDES.get((level, filename), {}))
        arrays = manifest["arrays"]
        params = manifest["parameters"].get("S", {})
        if hasattr(result, "forward"):
            if hasattr(result, "init"):
                init_args = [params[name] for name in inspect.signature(result.init).parameters if name in params]
                result.init(*init_args)
            forward_args = []
            for name in inspect.signature(result.forward).parameters:
                if name in values:
                    forward_args.append(values[name])
                elif name in params:
                    forward_args.append(params[name])
                elif name in arrays:
                    forward_args.append(values[name])
                else:
                    raise RuntimeError(f"missing runtime arg: {name}")
            result.forward(*forward_args)
        else:
            func_name = public_func_name(result, stem, manifest_path)
            run_buffer_function(getattr(result, func_name), values, None, manifest_path)
        return {"status": "pass", "detail": "", "seconds": round(time.perf_counter() - started, 3)}
    except Timeout:
        return {"status": "timeout", "detail": f">{timeout}s"}
    except Exception as exc:
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        signal.alarm(0)


def run_case(level: str, filename: str, timeout: int, rtol: float, atol: float):
    source_path = SOURCE_ROOT / level / filename
    stem = result_stem(source_path)
    result_path = RESULT_ROOT / level / stem / f"{stem}_numpy.py"
    manifest_path = RESULT_ROOT / level / stem / f"{stem}.yaml"
    source = import_module(source_path, f"torch_{level}_{stem}")
    result = import_module(result_path, f"numpy_{level}_{stem}")
    if hasattr(result, "TRANSLATION_ERROR"):
        return {"status": "translate_error", "detail": repr(result.TRANSLATION_ERROR)}
    patch_sizes(source, level, filename)

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout)
    started = time.perf_counter()
    try:
        torch.manual_seed(0)
        init_inputs = source.get_init_inputs()
        torch_model = source.Model(*init_inputs)
        torch_model.eval()
        make_inputs = PER_CASE_INPUTS.get((level, filename))
        inputs = make_inputs() if make_inputs is not None else source.get_inputs()
        with torch.no_grad():
            torch_out = torch_model(*inputs)

        numpy_inputs = to_numpy(inputs)
        numpy_init_inputs = to_numpy(init_inputs)
        copy_source_globals(source, result)
        copy_state_to_module(torch_model, result)

        if hasattr(result, "Model"):
            numpy_model = result.Model(*numpy_init_inputs)
            copy_state_to_module(torch_model, numpy_model)
            numpy_out = numpy_model.forward(*numpy_inputs)
        elif hasattr(result, "forward"):
            if hasattr(result, "init"):
                result.init(*numpy_init_inputs)
                copy_state_to_module(torch_model, result)
            numpy_out = result.forward(*numpy_inputs, *numpy_init_inputs)
        else:
            init_names, forward_names = source_arg_names(source)
            values = manifest_values(manifest_path)
            values.update(vars(source))
            values.update(dict(zip(init_names, numpy_init_inputs)))
            values.update(dict(zip(forward_names, numpy_inputs)))
            values.update(state_values(torch_model))
            func_name = public_func_name(result, stem, manifest_path)
            numpy_out = run_buffer_function(getattr(result, func_name), values, torch_out, manifest_path)

        ok, detail = compare_outputs(torch_out, numpy_out, rtol=rtol, atol=atol)
        return {"status": "pass" if ok else "fail", "detail": detail, "seconds": round(time.perf_counter() - started, 3)}
    except Timeout:
        return {"status": "timeout", "detail": f">{timeout}s"}
    except Exception as exc:
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        signal.alarm(0)


def iter_cases(levels: list[str], ids: list[str] | None):
    wanted = set(ids or [])
    for level in levels:
        for path in sorted((SOURCE_ROOT / level).glob("*.py"), key=numeric_key):
            names = {path.stem, path.name, f"{level}/{path.stem}"}
            number = path.stem.split("_", 1)[0]
            names.update({number, f"{level}/{number}"})
            if wanted and not (wanted & names):
                continue
            yield level, path.name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", default=["level1", "level2", "level3"])
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=5e-5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    args = parser.parse_args()

    results = []
    for level, filename in iter_cases(args.levels, args.ids):
        if args.runtime_only:
            outcome = run_runtime_case(level, filename, args.timeout)
        else:
            outcome = run_case(level, filename, args.timeout, args.rtol, args.atol)
        row = {"case": f"{level}/{result_stem(Path(filename))}.py", **outcome}
        results.append(row)
        if not args.json:
            detail = f" {row['detail']}" if row.get("detail") else ""
            print(f"{row['case']}: {row['status']}{detail}")

    if args.json:
        print(json.dumps(results, indent=2))
    return 0 if results and all(r["status"] == "pass" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
