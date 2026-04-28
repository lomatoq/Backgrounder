"""
Run this on Windows to install the correct CUDA-enabled torch version.
  python fix_torch_cuda.py
"""
import subprocess
import sys
import platform


def run(*args):
    print(f">>> {' '.join(args)}")
    subprocess.check_call(list(args))


def main():
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    # Check current state
    try:
        import torch
        current_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda
        print(f"\nCurrent torch: {current_ver}  CUDA available: {cuda_avail}  CUDA version: {cuda_ver}")
        if cuda_avail:
            print("CUDA is already working!")
            return
    except ImportError:
        current_ver = None
        print("torch not installed")

    # Detect CUDA driver version
    cuda_driver = None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True
        ).strip().split("\n")[0]
        print(f"nvidia-smi driver: {out}")
        # Map driver version to max CUDA version supported
        driver_ver = float(out.split(".")[0])
        if driver_ver >= 525:
            cuda_driver = "cu124"
        elif driver_ver >= 520:
            cuda_driver = "cu118"
        else:
            cuda_driver = "cu118"
    except Exception as e:
        print(f"Could not read nvidia-smi: {e}")
        cuda_driver = "cu124"  # default to latest

    print(f"\nSelected CUDA build: {cuda_driver}")
    index_url = f"https://download.pytorch.org/whl/{cuda_driver}"

    # Uninstall CPU torch
    if current_ver:
        print("\nUninstalling CPU torch...")
        run(sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "-y")

    # Install CUDA torch
    print(f"\nInstalling torch+CUDA from {index_url} ...")
    run(
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision",
        "--index-url", index_url,
        "--force-reinstall",
    )

    # Verify
    print("\nVerifying...")
    out = subprocess.check_output(
        [sys.executable, "-c",
         "import torch; print('torch', torch.__version__); "
         "print('CUDA available:', torch.cuda.is_available()); "
         "print('CUDA version:', torch.version.cuda); "
         "[print('GPU:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"],
        text=True
    )
    print(out)


if __name__ == "__main__":
    main()
