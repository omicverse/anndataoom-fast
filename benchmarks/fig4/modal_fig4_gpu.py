"""Fig. 4 omicverse GPU (RAPIDS) mode on SEA-AD, run on Modal.

SEA-AD is 1.38M cells x 36,601 genes with 7.6B stored values (float32 data +
int64 indices = 91 GB), so it needs a GPU larger than 80 GB and > 128 GB of
host RAM.

    # once: upload the benchmark input (same file as the Sherlock runs)
    modal volume create ov-fig4
    modal volume put ov-fig4 SEAAD_raw.h5ad /SEAAD_raw.h5ad
    # run (GPU type via FIG4_GPU, default B200)
    modal run modal_fig4_gpu.py
    modal volume get ov-fig4 /out ./modal_out
"""
import os
import subprocess

import modal

GPU = os.environ.get("FIG4_GPU", "B200")
HERE = os.path.dirname(os.path.abspath(__file__))
BRISC_BENCH = os.environ.get(
    "BRISC_BENCH", "/scratch/users/steorra/analysis/brisc_probe/brisc-benchmarking")

# rapids-singlecell >= 0.16 is an sdist with CUDA extensions built for
# CMAKE_CUDA_ARCHITECTURES=native, so it is compiled with the GPU attached.
image = (
    modal.Image.from_registry("rapidsai/base:26.10-cuda12-py3.12")
    .run_commands(
        "mamba create -y -q -p /opt/bt -c conda-forge -c nvidia gxx_linux-64=13 "
        "cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-cccl=12.9 libcublas-dev libcusolver-dev "
        "cuda-version=12.9 cmake ninja")
    .pip_install("omicverse==2.4.0", "anndataoom-fast>=0.1.2", "anndataoom", "polars",
                 "tabulate", "seaborn", "docrep", "natsort", "scikit-misc")
    .env({"CC": "/opt/bt/bin/x86_64-conda-linux-gnu-gcc",
          "CXX": "/opt/bt/bin/x86_64-conda-linux-gnu-g++",
          "CUDACXX": "/opt/bt/bin/nvcc",
          "CUDAHOSTCXX": "/opt/bt/bin/x86_64-conda-linux-gnu-g++",
          "CUDAToolkit_ROOT": "/opt/bt", "CMAKE_BUILD_PARALLEL_LEVEL": "16"})
    .run_commands("PATH=/opt/bt/bin:$PATH pip install --no-deps rapids-singlecell==0.18.1",
                  gpu=GPU)
    .add_local_file(os.path.join(HERE, "run_omicverse.py"), "/bench/run_omicverse.py")
    .add_local_file(os.path.join(BRISC_BENCH, "utils_local.py"), "/bench/utils_local.py")
)
vol = modal.Volume.from_name("ov-fig4")
app = modal.App("ov-fig4-gpu", image=image)


@app.function(gpu=GPU, cpu=16, memory=320 * 1024, timeout=3 * 3600, volumes={"/data": vol})
def run(threads: int = 16):
    out = "/data/out"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/hardware.txt", "w") as f:
        for cmd in (["lscpu"], ["nvidia-smi"], ["free", "-g"],
                    ["python", "-c", "import rapids_singlecell, cupy, omicverse;"
                     "print(rapids_singlecell.__version__, cupy.__version__, omicverse.__version__)"]):
            f.write(subprocess.run(cmd, capture_output=True, text=True).stdout)
    tag = f"SEAAD_ovgpu_t{threads}"
    with open(f"{out}/{tag}.log", "w") as log:
        rc = subprocess.run(
            ["python", "/bench/run_omicverse.py", "SEAAD", "/data/SEAAD_raw.h5ad", str(threads),
             f"{out}/{tag}.csv", "1", "--backend", "cpp", "--mode", "gpu"],
            stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "BRISC_BENCH": "/bench"}).returncode
    vol.commit()
    return rc


@app.local_entrypoint()
def main():
    print("exit code", run.remote())
