$ErrorActionPreference = "Stop"
$docx = Join-Path $PSScriptRoot "manuscript_zh.docx"
$pdf = Join-Path $PSScriptRoot "manuscript_zh.pdf"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.AutomationSecurity = 3
try {
    $doc = $word.Documents.Open($docx, $false, $true, $false)
    $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 0, 0, 0, $true, $true, 0, $true, $true, $false)
    $doc.Close($false)
    Write-Output "PDF generated: $pdf"
} finally {
    $word.Quit()
}
