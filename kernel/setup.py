import os
import re
import glob
import shutil
import pathlib
import subprocess
import setuptools
from typing import Tuple, List
import torch.utils.cpp_extension

def get_cuda_version() -> Tuple[int, ...]:
    # Try to locate NVCC
    nvcc_bin = None

    # Check in CUDA_HOME environment variable
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home:
        nvcc_bin = pathlib.Path(cuda_home) / "bin" / "nvcc"

    # Check in PATH if not found in CUDA_HOME
    if nvcc_bin is None or not nvcc_bin.is_file():
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            nvcc_bin = pathlib.Path(nvcc_path)

    # Check in default directory
    if nvcc_bin is None or not nvcc_bin.is_file():
        nvcc_bin = pathlib.Path("/usr/local/cuda/bin/nvcc")

    if not nvcc_bin.is_file():
        raise FileNotFoundError(
            "Could not locate NVCC. Please ensure CUDA is installed and accessible."
        )

    # Run nvcc to get the version
    try:
        output = subprocess.run(
            [nvcc_bin, "-V"],
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to run NVCC: {e}")

    # Parse the version string
    match = re.search(r"release\s*([\d.]+)", output.stdout)
    if not match:
        raise ValueError("Could not parse CUDA version from NVCC output.")

    version = tuple(map(int, match.group(1).split(".")))
    return version


def assert_cuda_version(version: Tuple[int, ...]) -> None:
    """
    Validates that the provided CUDA version meets Comfy's minimum requirements.
    Defaults to 12.8
    """
    lowest_cuda_version = (12, 8)
    if version < lowest_cuda_version:
        raise RuntimeError(
            f"Comfy requires CUDA {lowest_cuda_version} or newer. Got {version}"
        )


def get_cxx_flags(debug_mode: bool) -> list[str]:
    base_cxx_flags = [
        "-fvisibility=hidden",
        "-fdiagnostics-color=always",
        "-std=c++20",
    ]
    if debug_mode:
        return base_cxx_flags + ["-g", "-lineinfo", "-O0"]
    else:
        return base_cxx_flags + ["-O3"]


def get_nvcc_flags(debug_mode: bool, for_cutlass_gemm: bool) -> list[str]:
    base_nvcc_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-std=c++20",

        # suppress warning from libtorch headers
        # https://github.com/pytorch/pytorch/issues/98917
        "-diag-suppress=3189",
        "-diag-suppress=2908",  # suppress warning from cutlass sm100_blockscaled_mma_warpspecialized.hpp
    ]
    if debug_mode:
        if for_cutlass_gemm:
            # -G hangs the compilation process for CUTLASS due to template heavy
            # nature of the code.
            # CUTLASS is different since it has it's own way to debug such as
            # trace and compute-sanitizer etc.
            return base_nvcc_flags + ["-g", "-lineinfo", "-O0"]
        else:
            return base_nvcc_flags + ["-g", "-lineinfo", "-O0", "-G"]
    else:
        if for_cutlass_gemm:
            return base_nvcc_flags + ["-O3", "-DNDEBUG"]
        else:
            return base_nvcc_flags + ["-O3"]


def get_nvcc_gencode_flags(cuda_version: Tuple[int, ...]) -> list[str]:
    assert_cuda_version(cuda_version)
    # add nvcc flags for specific architectures
    if os.name == "nt": # Only compile for Consumer on Windows
        cuda_archs = os.getenv("COMFY_CUDA_ARCHS", "89;120")
    else:
        cuda_archs = os.getenv("COMFY_CUDA_ARCHS", "89;90a;100a;120")
    gencode_flags = []

    for arch in cuda_archs.split(";"):
        gencode_flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return gencode_flags


def setup_extensions() -> List[setuptools.Extension]:
    debug_mode = os.getenv("DEBUG", "0") == "1"

    # compiler flags
    cxx_flags = get_cxx_flags(debug_mode)
    ext_nvcc_flags = get_nvcc_flags(debug_mode, False)
    # Necessary linker flags. We need those for our extension regardless of
    # what PyTorch depends on.
    extra_link_args = ["-lcuda", "-lcudart"]

    # version-dependent CUDA options
    try:
        cuda_version = get_cuda_version()
    except FileNotFoundError:
        raise FileNotFoundError("Could not determine CUDA Toolkit version")
    else:
        gencode_flags = get_nvcc_gencode_flags(cuda_version)
        ext_nvcc_flags.extend(gencode_flags)

    # define sources and include directories
    root_dir = pathlib.Path(__file__).resolve().parent
    extensions_dir = root_dir / "comfy_quant/csrc"


    ext_sources = set(
        glob.glob(os.path.join(extensions_dir, "**/*.cpp"), recursive=True)
    )
    ext_sources |= set(
        glob.glob(os.path.join(extensions_dir, "**/*.cu"), recursive=True)
    )

    ext_include_dirs = [str(extensions_dir)]

    # construct the extension
    ext_module = torch.utils.cpp_extension.CUDAExtension(
        name="comfy_quant.ext",
        sources=ext_sources,
        include_dirs=ext_include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": ext_nvcc_flags,
        },
        extra_link_args=extra_link_args,
    )

    return [ext_module]


setuptools.setup(
    name="nvidia-comfy",
    version="0.0.1",
    packages=setuptools.find_packages(),
    include_package_data=False,
    description="NVIDIA Comfy Extension",
    ext_modules=setup_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    setup_requires=["cmake", "ninja"],
    install_requires=["torch"],
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
)
