param(
  [Parameter(Mandatory = $true)]
  [string]$InputIfc,

  [Parameter(Mandatory = $true)]
  [string]$OutputGlb
)

$converter = Get-Command IfcConvert -ErrorAction SilentlyContinue
if (-not $converter) {
  Write-Error "IfcConvert was not found on PATH. Install IfcOpenShell and expose IfcConvert before running this script."
  exit 1
}

if (-not (Test-Path $InputIfc)) {
  Write-Error "Input IFC not found: $InputIfc"
  exit 1
}

$outputDirectory = Split-Path -Path $OutputGlb -Parent
if ($outputDirectory) {
  New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}

& $converter.Source $InputIfc $OutputGlb
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host "Wrote GLB asset to $OutputGlb"
