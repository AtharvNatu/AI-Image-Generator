import sys
import subprocess
import os
import venv
import platform

MIN_PYTHON_VERSION = (3, 8)
VENV_DIR = "venv"

def check_python_version():
    if sys.version_info < MIN_PYTHON_VERSION:
        sys.exit(f"Error : Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or higher is required ... Exiting !!!")


def create_virtual_environment():
    if not os.path.exists(VENV_DIR):
        print(f"Creating Virtual Environment In '{VENV_DIR}' ...")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(f"Virtual Environment already exists in '{VENV_DIR}' !!")


def get_pip():
    if os.name == "nt":
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
       return os.path.join(VENV_DIR, "bin", "pip")

def install_requirements(pip):
    print("Installing Dependencies From requirements.txt ...")
    subprocess.check_call([
        pip, 
        "install", 
        "-r", 
        "requirements.txt"
    ])


def check_nvidia_gpu():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    

def check_linux_rocm():
    try:
        output = subprocess.check_output(["rocminfo"], stderr=subprocess.DEVNULL)
        return b"Agent" in output
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    
def install_torch(pip):
    system = platform.system().lower()

    if check_nvidia_gpu():
            print("✅ NVIDIA GPU detected. Installing PyTorch with CUDA 12.8 support...")
            subprocess.check_call([
                pip, "install", "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu128"
            ])

    elif system == "linux" and check_linux_rocm():
        print("✅ ROCm detected on Linux. Installing PyTorch with ROCm 6.3 support...")
        subprocess.check_call([
            pip, "install", "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/rocm6.3"
        ])

    else:
        print("⚠️ No compatible GPU detected. Installing CPU-only PyTorch.")
    subprocess.check_call([
        pip, "install", "torch", "torchvision", "torchaudio"
    ])

if __name__ == "__main__":
    
    # Check Python Version
    check_python_version()

    # Create Virtual Environment
    create_virtual_environment()

    # Get Pip Executable
    pip_executable = get_pip()

    # Install Requirements
    install_requirements(pip_executable)

    # Install Torch
    install_torch(pip_executable)

    print("\n✅ Environment Setup Done ...")
