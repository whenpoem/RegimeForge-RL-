$Lang = if ($args.Count -ge 1) { [string]$args[0] } else { 'auto' }
$Charset = if ($args.Count -ge 2) { [string]$args[1] } else { 'unicode' }

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location (Join-Path $repoRoot 'backend')

$pythonExe = 'D:\miniconda\envs\statshell\python.exe'

if (-not (Test-Path $pythonExe)) {
  throw "statshell python not found at $pythonExe"
}

try { chcp 65001 > $null } catch {}
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

& $pythonExe -m regime_lens.tui --fresh --lang $Lang --charset $Charset
