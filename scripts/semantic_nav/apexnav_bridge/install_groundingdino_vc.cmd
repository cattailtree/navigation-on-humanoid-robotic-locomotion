@echo off
chcp 65001 >NUL
call D:\vs\hello\VC\Auxiliary\Build\vcvars64.bat

set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
set "PYTHONIOENCODING=utf-8"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"
set "MAX_JOBS=1"
set "GROUNDINGDINO_DISABLE_EXT=1"

cd /d D:\ApexNav\GroundingDINO
D:\anaconda\envs\apexnav-vlm\python.exe -m pip install -e . --no-build-isolation
