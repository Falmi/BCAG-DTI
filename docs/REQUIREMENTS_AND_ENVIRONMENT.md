# Requirements and environment notes

The Python dependencies required by the MolTrans/BCAG-DTI code are listed in the repository-level `requirements.txt`.

Core runtime components include Python, PyTorch with CUDA support, NumPy, pandas, scikit-learn, subword-nmt, tqdm, and matplotlib.

CPI-GGS additionally requires RDKit and PyTorch Geometric. The completed external reproduction recorded Python 3.12, PyTorch `2.13.0+cu126`, PyTorch Geometric `2.8.0.post1`, and CUDA execution on an NVIDIA GeForce RTX 3060 Laptop GPU (6 GB).

For exact values used in each final MolTrans/BCAG-DTI run, use `configs/all_run_configurations.csv`; there is no single optimisation configuration shared across all A1–A8 variants.

The four-GPU suite runners import the POSIX file-locking module `fcntl` and should be launched on Linux. Single-run model and data utilities can be inspected on Windows, but the complete scheduler test suite is Linux-oriented.
