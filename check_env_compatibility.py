import sys
import importlib.metadata

print(f"Python Executable: {sys.executable}")

def check_package(name):
    try:
        version = importlib.metadata.version(name)
        print(f"{name}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{name}: Not installed")

check_package("numpy")
check_package("torch")
check_package("tensorflow")
check_package("cellpose")

try:
    import numpy
    print(f"Actual Numpy version imported: {numpy.__version__}")
except ImportError:
    print("Could not import numpy")
