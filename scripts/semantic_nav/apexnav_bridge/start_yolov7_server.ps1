param(
    [string]$ApexNavRoot = "D:\ApexNav",
    [int]$Port = 12184,
    [string]$PythonExe = "D:\anaconda\envs\apexnav-vlm\python.exe"
)

Set-Location $ApexNavRoot
$env:PYTHONPATH = "$ApexNavRoot;$ApexNavRoot\yolov7;$env:PYTHONPATH"
& $PythonExe -m vlm.detector.yolov7 --port $Port
