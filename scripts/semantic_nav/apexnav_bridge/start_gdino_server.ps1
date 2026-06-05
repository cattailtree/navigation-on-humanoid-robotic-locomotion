param(
    [string]$ApexNavRoot = "D:\ApexNav",
    [int]$Port = 12181,
    [string]$PythonExe = "D:\anaconda\envs\apexnav-vlm\python.exe"
)

Set-Location $ApexNavRoot
New-Item -ItemType Directory -Force "$ApexNavRoot\.cache\huggingface" | Out-Null
New-Item -ItemType Directory -Force "$ApexNavRoot\.cache\huggingface\transformers" | Out-Null
$env:PYTHONPATH = "$ApexNavRoot;$ApexNavRoot\GroundingDINO;$env:PYTHONPATH"
$env:HF_HOME = "$ApexNavRoot\.cache\huggingface"
$env:TRANSFORMERS_CACHE = "$ApexNavRoot\.cache\huggingface\transformers"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:GROUNDINGDINO_DISABLE_EXT = "1"
& $PythonExe -m vlm.detector.grounding_dino --port $Port
