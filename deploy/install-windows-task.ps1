<#
Run ibif-napari-server at start-up on a Windows facility GPU workstation.

Run in an *administrator* PowerShell:
    .\install-windows-task.ps1 -EnvPath C:\Users\ibif\miniconda3\envs\ibif-napari -Allow 10.0.0.0/8

It registers a scheduled task that starts the server at boot (and restarts it if it stops)
under the Windows account you enter, and opens the port in Windows Firewall for the
allowed networks only. The account's home folder holds the users file
(.ibif-napari-server) and the Cellpose models (.cellpose): add users while logged in as
that account. Remove everything with:  .\install-windows-task.ps1 -Uninstall
#>
param(
    [string]$EnvPath = "$env:USERPROFILE\miniconda3\envs\ibif-napari",
    [int]$Port = 8765,
    [string[]]$Allow = @("LocalSubnet"),
    [string]$ExtraArgs = "",
    [switch]$Uninstall
)
$ErrorActionPreference = "Stop"
$name = "IBIF napari Cellpose server"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction SilentlyContinue
    Remove-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
    Write-Host "Removed the task and the firewall rule."
    return
}

$exe = Join-Path $EnvPath "Scripts\ibif-napari-server.exe"
if (-not (Test-Path $exe)) { throw "Not found: $exe. Pass -EnvPath <the conda environment folder>." }
$log = Join-Path $env:ProgramData "ibif-napari-server.log"
$allowArgs = ($Allow | Where-Object { $_ -match '^[0-9a-fA-F:.]+/\d+$' } | ForEach-Object { "--allow $_" }) -join " "
$arguments = "run --port $Port --log-file `"$log`" $allowArgs $ExtraArgs"

$action = New-ScheduledTaskAction -Execute $exe -Argument $arguments
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$cred = Get-Credential -Message "Windows account the Cellpose server runs as"
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings `
    -User $cred.UserName -Password $cred.GetNetworkCredential().Password -RunLevel Limited -Force | Out-Null

Remove-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName $name -Direction Inbound -Protocol TCP -LocalPort $Port `
    -RemoteAddress $Allow -Action Allow | Out-Null

Start-ScheduledTask -TaskName $name
Write-Host "Started. Log: $log"
Write-Host "Command: $exe $arguments"
