# CUDA Muon Sampling Simulation

Lightweight tools to collect single-step Geant4 samples, build CUDA-friendly histograms, and run a GPU-accelerated muon sampler.

## Quick start

Prerequisites
- Python 3.8+ and pip
- CUDA (if you plan to run the CUDA simulation)
- Project environment: ensure PROJECTS_DIR is set or run from the project root so relative paths resolve.

1. Collect single-step sampling data from Geant4 outputs:
   - Run the extractor script:
     ```
     python3 collect_single_step_data.py
     ```
   - See script: [MuonsAndMatter/cuda_muons/collect_single_step_data.py](MuonsAndMatter/cuda_muons/collect_single_step_data.py)

2. Build histograms used by the CUDA sampler:
   - Run:
     ```
     python3 build_histograms.py --alias
     ```
   - See script: [MuonsAndMatter/cuda_muons/build_histograms.py](MuonsAndMatter/cuda_muons/build_histograms.py)

3. Run a full CUDA-accelerated simulation:
   - Launch the main simulator:
     ```
     python3 faster_muons/run_cuda_muons.py
     ```
   - See entrypoint: [MuonsAndMatter/cuda_muons/faster_muons/run_cuda_muons.py](MuonsAndMatter/cuda_muons/faster_muons/run_cuda_muons.py)
   - If present, consult the build helper: [MuonsAndMatter/cuda_muons/faster_muons/setup.py](MuonsAndMatter/cuda_muons/faster_muons/setup.py)

## Typical workflow

- Collect raw single-step samples with the extractor, then convert them into histogram files that the CUDA sampler consumes.
- Build/install any CUDA/PyTorch extensions before running the simulator if required by the `faster_muons` code.


## Troubleshooting

- If the CUDA run fails: confirm CUDA drivers/toolkit are installed and compatible with your PyTorch/CUDA bindings.
- If Python imports fail: ensure your PYTHONPATH includes the project root or activate the virtualenv used to build the extension.
- For histogram or data-file problems, inspect the `data/outputs` files produced by the collector and builder.

