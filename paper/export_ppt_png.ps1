$ErrorActionPreference = "Stop"
$pptx = (Get-ChildItem -Path $PSScriptRoot -Filter *.pptx | Select-Object -First 1).FullName
$out = Join-Path $PSScriptRoot "ppt_preview"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open($pptx, $true, $false, $false)
    $pres.Export($out, "PNG", 1600, 900)
    $pres.Close()
    Write-Output "OK $out"
} finally {
    $app.Quit()
}
