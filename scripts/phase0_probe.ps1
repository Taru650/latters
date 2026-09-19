# Phase 0 hardware probe -- run in PowerShell on the target laptop.
#   powershell -ExecutionPolicy Bypass -File scripts\phase0_probe.ps1
#
# Answers the four questions that decide everything downstream:
#   1. Is RAM single- or dual-channel? (single-channel halves LLM decode speed)
#   2. Is C: a spinning disk? (decides whether model load is 5s or 90s)
#   3. Which GPUs exist, and do they expose DirectX 12? (decides the DirectML path)
#   4. How much RAM is actually free with normal apps open?

$ErrorActionPreference = "Continue"
Write-Output "=== Phase 0 hardware probe -- $(Get-Date -Format s) ==="

Write-Output "`n--- CPU ---"
Get-CimInstance Win32_Processor |
  Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed |
  Format-List

Write-Output "--- Memory modules (THE IMPORTANT ONE) ---"
$modules = Get-CimInstance Win32_PhysicalMemory
$modules | Select-Object BankLabel, DeviceLocator,
  @{n='CapacityGB';e={[math]::Round($_.Capacity/1GB,1)}},
  @{n='SpeedMHz';e={$_.Speed}}, ConfiguredClockSpeed, Manufacturer, PartNumber |
  Format-Table -AutoSize

$n = @($modules).Count
if ($n -le 1) {
  Write-Output "!! SINGLE MODULE DETECTED -- the memory controller is running in"
  Write-Output "!! single-channel mode (~19 GB/s). LLM token generation is"
  Write-Output "!! memory-bandwidth-bound, so adding a second matched SODIMM is"
  Write-Output "!! likely the single largest speedup available, ahead of any"
  Write-Output "!! software change. Verify the second slot is physically present."
} else {
  Write-Output "OK: $n modules -> dual-channel capable (~38 GB/s)."
}

Write-Output "`n--- Physical disks (HDD vs SSD) ---"
Get-PhysicalDisk | Select-Object DeviceId, FriendlyName, MediaType, BusType,
  @{n='SizeGB';e={[math]::Round($_.Size/1GB,0)}}, HealthStatus | Format-Table -AutoSize
if ((Get-PhysicalDisk | Where-Object MediaType -eq 'HDD')) {
  Write-Output "!! A spinning disk is present. If the model lives on it, expect"
  Write-Output "!! 40-90s cold load and severe stalls under memory pressure."
  Write-Output "!! Mitigations, in order: (a) fit an SSD; (b) set keep_alive=-1"
  Write-Output "!! so Ollama loads the model once per boot and never re-reads it."
}

Write-Output "`n--- Free space on C: ---"
Get-PSDrive C | Select-Object Used, Free,
  @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}} | Format-Table -AutoSize

Write-Output "--- GPUs ---"
Get-CimInstance Win32_VideoController |
  Select-Object Name, DriverVersion, DriverDate,
    @{n='VRAM_GB';e={[math]::Round($_.AdapterRAM/1GB,2)}},
    VideoProcessor, Status | Format-List

Write-Output "--- DirectX feature levels (dxdiag) ---"
$dx = Join-Path $env:TEMP "dxdiag_phase0.txt"
Start-Process dxdiag -ArgumentList "/whql:off","/t",$dx -Wait -NoNewWindow
if (Test-Path $dx) {
  Select-String -Path $dx -Pattern "Card name|Feature Levels|Driver Version|Display Memory|Dedicated Memory" |
    ForEach-Object { $_.Line.Trim() }
  Write-Output "(full dxdiag: $dx)"
  Write-Output "Need 'Feature Levels' to include 12_0 or 11_1 for ONNX Runtime DirectML."
}

Write-Output "`n--- Memory pressure right now ---"
$os = Get-CimInstance Win32_OperatingSystem
$totalGB = [math]::Round($os.TotalVisibleMemorySize/1MB,2)
$freeGB  = [math]::Round($os.FreePhysicalMemory/1MB,2)
Write-Output "Total: $totalGB GB   Free: $freeGB GB   In use: $([math]::Round($totalGB-$freeGB,2)) GB"
Write-Output "Target for this project: >= 5.0 GB free before starting the assistant."

Write-Output "`n--- Top 10 memory consumers ---"
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 10 `
  Name, @{n='MB';e={[math]::Round($_.WorkingSet64/1MB,0)}} | Format-Table -AutoSize

Write-Output "`n--- Ollama ---"
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) { ollama --version; ollama list } else { Write-Output "ollama not on PATH" }

Write-Output "`n=== probe complete -- paste this whole output into docs/BASELINE.md ==="
