param(
    [string]$EnvName = "apexnav-vlm",
    [string]$ApexNavRoot = "D:\ApexNav",
    [string]$CondaExe = "D:\anaconda\Scripts\conda.exe",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu121"
)

if (-not (Test-Path $CondaExe)) {
    throw "Cannot find conda at $CondaExe. Pass -CondaExe with the full path to conda.exe."
}

& $CondaExe create -n $EnvName python=3.10 -y
& $CondaExe run -n $EnvName python -m pip install --upgrade pip setuptools wheel
& $CondaExe run -n $EnvName python -m pip install torch torchvision torchaudio --index-url $TorchIndexUrl
& $CondaExe run -n $EnvName python -m pip install flask numpy opencv-python pillow requests supervision
& $CondaExe run -n $EnvName python -m pip install -e "$ApexNavRoot\GroundingDINO"
& $CondaExe run -n $EnvName python -m pip install -r "$ApexNavRoot\yolov7\requirements.txt"
& $CondaExe run -n $EnvName python -m pip install -e $ApexNavRoot --no-deps
