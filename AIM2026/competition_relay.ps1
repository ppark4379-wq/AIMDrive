#Requires -Version 5.1
<#
대회장 랜투랜 준비: MORAI Client PC(192.168.0.1) -> 이 PC(192.168.0.10) -> WSL의 main.py.

WSL2는 NAT라서 랜으로 들어온 UDP가 Windows에서 멈춘다. 이 스크립트가 Windows 쪽에서 그것을 WSL로 넘기는
분배기(AIM2026/UDP/udp_relay.py)를 띄우고 (센서별 포트 세트와 9090 공유 포트 세트 둘 다 받음), 그 전후를 점검한다:
  1 유선 어댑터와 링크 속도   2 IP(-SetIp면 192.168.0.10/24 고정)   3 대회 PC ping
  4 방화벽(python 수신 허용, 차단 규칙, 네트워크 프로필)   5 전원(절전 끄기)
  6 예전 분배기 정리   7 분배기 실행(빠른 편집 끔, 높은 우선순위)   8 WSL에서 실제 수신 확인
제어값은 WSL에서 대회 PC로 바로 나간다(NAT가 송신 포트를 바꿔도 MORAI는 받는다, 10-29 확인).

관리자 PowerShell에서:
  powershell -NoProfile -ExecutionPolicy Bypass -File \\wsl.localhost\Ubuntu-20.04\home\dydwns\AIMDrive\AIM2026\competition_relay.ps1
  옵션: -SetIp (IP 고정)  -ControlCheck (차가 서 있을 때 brake 0.37/1.0을 보내 제어 경로까지 확인)
        -Config <WSL 경로>  -AdapterIndex <번호>  -NoRelay  -ProbeSeconds 5  -SharedPort 9090
끝나면 WSL에서: cd ~/AIMDrive && python3 AIM2026/main.py --config AIM2026/config_competition.yaml --competition
#>
param(
    [string]$SimIp = "192.168.0.1",
    [string]$MyIp = "192.168.0.10",
    [int]$AdapterIndex = 0,
    [string]$Config = "",
    [string]$Distro = "",
    [string]$WinPython = "",
    [int]$ProbeSeconds = 5,
    [int]$SharedPort = 9090,
    [switch]$SetIp,
    [switch]$ControlCheck,
    [switch]$NoRelay
)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$OutputEncoding = [Text.Encoding]::UTF8

function Step($t) { Write-Host ""; Write-Host "== $t" -ForegroundColor Cyan }
function Ok($t)   { Write-Host "   [OK]   $t" -ForegroundColor Green }
function Warn($t) { Write-Host "   [주의] $t" -ForegroundColor Yellow }
function Info($t) { Write-Host "          $t" -ForegroundColor DarkGray }
function Fail($t) { Write-Host "   [실패] $t" -ForegroundColor Red; exit 1 }

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)

# ------------------------------------------------------------------ 0. 위치 찾기
# 이 파일은 \\wsl.localhost\<배포판>\...\AIMDrive\AIM2026 에서 실행된다: 배포판과 저장소 경로를 거기서 읽는다.
$m = [regex]::Match($PSScriptRoot, '^\\\\wsl(?:\.localhost|\$)\\(?<distro>[^\\]+)\\(?<path>.+)$', 'IgnoreCase')
if (-not $m.Success) { Fail "WSL 경로(\\wsl.localhost\...)에서 실행하세요: $PSScriptRoot" }
if (-not $Distro) { $Distro = $m.Groups['distro'].Value }
$aimWsl = '/' + ($m.Groups['path'].Value -replace '\\', '/')
$repoWsl = $aimWsl -replace '/AIM2026$', ''
if (-not $Config) { $Config = "$aimWsl/config_competition.yaml" }
$helper = "$aimWsl/UDP/competition_ports.py"
$relayWin = Join-Path $PSScriptRoot "UDP\udp_relay.py"
if (-not (Test-Path -LiteralPath $relayWin)) { Fail "분배기가 없습니다: $relayWin" }
if (-not $WinPython) {
    $c = @((Join-Path $env:USERPROFILE "miniconda3\python.exe"), (Join-Path $env:USERPROFILE "anaconda3\python.exe"))
    $c += @(Get-ChildItem (Join-Path $env:LOCALAPPDATA "Programs\Python\Python*\python.exe") -ErrorAction SilentlyContinue |
            ForEach-Object { $_.FullName })
    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($cmd) { $c += $cmd.Source }
    $WinPython = $c | Where-Object { $_ -and (Test-Path -LiteralPath $_) -and $_ -notlike '*WindowsApps*' } |
                 Select-Object -First 1
    if (-not $WinPython) { Fail "Windows python.exe를 못 찾았습니다. -WinPython 경로를 주세요." }
}
Write-Host "배포판 $Distro / 설정 $Config / Windows python $WinPython"
if (-not $admin) { Warn "관리자 권한이 아닙니다: 방화벽 규칙 추가와 -SetIp는 건너뜁니다." }

# ------------------------------------------------------------ 1. 유선 어댑터, 링크 속도
Step "유선 어댑터와 링크 속도"
if ($AdapterIndex) {
    $adapter = Get-NetAdapter -InterfaceIndex $AdapterIndex -ErrorAction SilentlyContinue
} else {
    $adapter = @(Get-NetAdapter -Physical | Where-Object { $_.Status -eq "Up" -and $_.PhysicalMediaType -notlike "*802.11*" }) |
               Select-Object -First 1
}
if (-not $adapter) {
    Get-NetAdapter -Physical | Format-Table Name, InterfaceDescription, Status, LinkSpeed -AutoSize | Out-Host
    Fail "연결된 유선 어댑터가 없습니다. 케이블을 확인하세요(양쪽 포트 LED가 켜져야 함)."
}
if ($adapter.Status -ne "Up") { Fail "$($adapter.Name) 상태가 $($adapter.Status) 입니다. 케이블을 확인하세요." }
$mbps = [math]::Round($adapter.Speed / 1e6)
Info "$($adapter.Name) ($($adapter.InterfaceDescription)), 번호 $($adapter.InterfaceIndex)"
if ($mbps -lt 1000) {
    Warn "링크 $($adapter.LinkSpeed) 입니다. 센서 전체가 약 50 Mbps를 써서 100 Mbps면 카메라 프레임이 유실될 수 있습니다."
    Info "Cat5e/Cat6 케이블로 바꾸거나 다시 꽂아 1 Gbps 이상으로 잡히는지 보세요(허브/공유기 말고 직결)."
} else { Ok "링크 $($adapter.LinkSpeed), 전이중 $($adapter.FullDuplex)" }
$wifi = @(Get-NetAdapter -Physical | Where-Object { $_.Status -eq "Up" -and $_.PhysicalMediaType -like "*802.11*" })
if ($wifi.Count) { Warn "Wi-Fi도 연결돼 있습니다($($wifi[0].Name)). 192.168.0.x 대역이 겹치면 꺼 두세요." }

# --------------------------------------------------------------------------- 2. IP
Step "IP ($MyIp/24, 대회 PC $SimIp)"
if ($SetIp) {
    if (-not $admin) { Warn "-SetIp는 관리자 권한이 필요합니다." }
    else {
        Set-NetIPInterface -InterfaceIndex $adapter.InterfaceIndex -Dhcp Disabled
        Get-NetIPAddress -InterfaceIndex $adapter.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Remove-NetIPAddress -Confirm:$false
        New-NetIPAddress -InterfaceIndex $adapter.InterfaceIndex -IPAddress $MyIp -PrefixLength 24 | Out-Null
        Ok "$MyIp/24 로 고정했습니다. 되돌리기: Set-NetIPInterface -InterfaceIndex $($adapter.InterfaceIndex) -Dhcp Enabled"
        Start-Sleep -Seconds 2
    }
}
$ips = @(Get-NetIPAddress -InterfaceIndex $adapter.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
         ForEach-Object { $_.IPAddress })
if ($ips -contains $MyIp) { Ok "이 PC: $($ips -join ', ')" }
else { Warn "이 어댑터 IP가 $($ips -join ', ') 입니다. MORAI는 $MyIp 로 보냅니다 (-SetIp로 고정 가능)." }

# ------------------------------------------------------------------------- 3. ping
Step "대회 PC 연결"
if (Test-Connection -ComputerName $SimIp -Count 2 -Quiet) { Ok "$SimIp ping 응답" }
else { Warn "$SimIp ping 응답 없음. 대회 PC가 ping을 막았을 수도 있으니 8단계 수신 결과로 판단하세요." }

# ---------------------------------------------------------------------- 4. 방화벽
Step "방화벽 (python 수신)"
$category = (Get-NetConnectionProfile -InterfaceIndex $adapter.InterfaceIndex -ErrorAction SilentlyContinue).NetworkCategory
Info "이 연결의 네트워크 프로필: $category"
$mine = @(Get-NetFirewallApplicationFilter | Where-Object { $_.Program -and ($_.Program -eq $WinPython) } |
          Get-NetFirewallRule | Where-Object { $_.Enabled -eq 'True' -and $_.Direction -eq 'Inbound' })
$block = @($mine | Where-Object { $_.Action -eq 'Block' })
if ($block.Count) {
    Warn "python.exe 수신 '차단' 규칙이 있습니다(허용보다 우선): $(($block | ForEach-Object { $_.DisplayName }) -join ', ')"
    if ($admin) { $block | Disable-NetFirewallRule; Ok "차단 규칙을 껐습니다." }
    else { Info "관리자 권한으로 다시 실행하면 끕니다." }
}
$allow = @($mine | Where-Object { $_.Action -eq 'Allow' -and ($_.Profile -eq 'Any' -or "$($_.Profile)" -like "*$category*") })
if ($allow.Count) { Ok "python 수신 허용 규칙: $($allow[0].DisplayName) ($($allow[0].Profile))" }
elseif ($admin) {
    New-NetFirewallRule -DisplayName "AIM2026 relay (python UDP in)" -Direction Inbound -Protocol UDP `
        -Program $WinPython -Action Allow -Profile Any | Out-Null
    Ok "python 수신 허용 규칙을 추가했습니다 (모든 프로필)."
} else { Warn "이 프로필($category)에 맞는 python 허용 규칙이 없습니다. 관리자 권한으로 다시 실행하세요." }
# 본선 매뉴얼 4단계: 대회 PC에서 ping 192.168.0.10 으로 연결을 확인한다. Windows는 Public 프로필에서 ping을 막는다.
$pingRule = "AIM2026 ping in (ICMPv4 echo)"
if (Get-NetFirewallRule -DisplayName $pingRule -ErrorAction SilentlyContinue) { Ok "ping 응답 허용 규칙 있음" }
elseif ($admin) {
    New-NetFirewallRule -DisplayName $pingRule -Direction Inbound -Protocol ICMPv4 -IcmpType 8 -Action Allow -Profile Any | Out-Null
    Ok "ping 응답 허용 규칙을 추가했습니다 (대회 PC에서 ping 192.168.0.10 확인용)."
} else { Warn "ping 응답 허용 규칙이 없습니다. 대회 PC의 ping 192.168.0.10 확인이 실패할 수 있습니다(관리자로 실행)." }

# -------------------------------------------------------------------------- 5. 전원
Step "전원"
$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
if ($battery -and $battery.BatteryStatus -eq 1) { Warn "배터리로 돌고 있습니다. 전원 어댑터를 꽂으세요." }
if ($NoRelay) { Info "점검만(-NoRelay): 절전 설정은 바꾸지 않음" }
else {
    & powercfg /change standby-timeout-ac 0 | Out-Null
    & powercfg /change hibernate-timeout-ac 0 | Out-Null
    Ok "전원 연결 시 절전/최대 절전 끔"
}

# ------------------------------------------------------------ 6. 포트와 WSL IP
$ports = ((& wsl.exe -d $Distro -- python3 $helper ports --config $Config) -join '').Trim()
if ($LASTEXITCODE -ne 0 -or $ports -notmatch '^\d+(,\d+)*$') { Fail "설정에서 포트를 못 읽었습니다: $ports" }
# 센서 세트가 센서별 포트(aim_sensor.json)든, 한 포트로 모아 보내는 것(로컬과 같은 세트, 목적지 9090을 보낸 쪽
# 포트로 구분)이든 WSL에는 같은 포트로 넘긴다: 둘 다 열어 둔다.
$demux = ((& wsl.exe -d $Distro -- python3 $helper demux --config $Config --shared-port $SharedPort) -join '').Trim()
if ($LASTEXITCODE -ne 0 -or $demux -notmatch '^\d+:\d+=\d+(,\d+=\d+)*$') { Fail "설정에서 센서 송신 포트를 못 읽었습니다: $demux" }
$wslIp = ((& wsl.exe -d $Distro -- hostname -I) -join ' ').Trim().Split(' ')[0]
if (-not $wslIp) { Fail "WSL IP를 못 읽었습니다." }
$relayArgs = @($relayWin, "--listen", "0.0.0.0", "--to", $wslIp, "--ports", $ports, "--demux", $demux,
               "--log", (Join-Path $env:TEMP "udp_relay_competition.log"))
if ($NoRelay) {
    Step "분배기 (건너뜀: -NoRelay, 점검만)"
    Info "$WinPython $($relayArgs -join ' ')"
    exit 0
}
# ------------------------------------------- 7. 예전 분배기 정리, 분배기 실행
Step "예전 분배기 정리"
$old = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" -ErrorAction SilentlyContinue |
         Where-Object { $_.CommandLine -like "*udp_relay*" })
foreach ($p in $old) { Stop-Process -Id $p.ProcessId -Force; Info "종료: PID $($p.ProcessId)" }
if (-not $old.Count) { Ok "실행 중인 분배기 없음" } else { Start-Sleep -Seconds 1 }
$busy = @(Get-NetUDPEndpoint -ErrorAction SilentlyContinue | Where-Object { "$($_.LocalPort)" -in ($ports -split ',') })
foreach ($b in $busy) {
    $name = (Get-Process -Id $b.OwningProcess -ErrorAction SilentlyContinue).ProcessName
    Warn "Windows에서 UDP $($b.LocalPort)를 $name(PID $($b.OwningProcess))이 잡고 있습니다. 분배기가 못 받을 수 있습니다."
}
Step "분배기 실행 -> WSL $wslIp"
Info "센서별 포트 $ports 그대로 / 공유 포트 $($demux -replace ':.*','')는 보낸 쪽 포트로 나눔 ($($demux -replace '^\d+:',''))"
# 콘솔 창을 마우스로 누르면 '빠른 편집'이 프로세스를 멈춘다(센서 끊김). 이 python으로 여는 창에서는 끈다.
$consoleKey = "HKCU:\Console\" + ($WinPython -replace '\\', '_')
New-Item -Path $consoleKey -Force | Out-Null
Set-ItemProperty -Path $consoleKey -Name QuickEdit -Value 0 -Type DWord
Set-ItemProperty -Path $consoleKey -Name InsertMode -Value 0 -Type DWord
$relay = Start-Process -FilePath $WinPython -ArgumentList $relayArgs -WindowStyle Normal -PassThru
Start-Sleep -Seconds 2
if ($relay.HasExited) { Fail "분배기가 바로 종료됐습니다(코드 $($relay.ExitCode)). 포트 점유를 확인하세요." }
try { $relay.PriorityClass = 'High' } catch { }
Ok "분배기 PID $($relay.Id), 우선순위 높음. 주행이 끝날 때까지 그 창을 닫거나 누르지 마세요."
Info "기록: $(Join-Path $env:TEMP 'udp_relay_competition.log') (STALL 줄이 있으면 분배기가 멈췄던 것)"

# ------------------------------------------------------------------ 8. WSL 수신 확인
Step "WSL 수신 확인 ($ProbeSeconds 초)"
$probeArgs = @('-d', $Distro, '--', 'python3', $helper, 'probe', '--config', $Config, '--seconds', "$ProbeSeconds")
if ($ControlCheck) { $probeArgs += '--control-check' }
& wsl.exe @probeArgs | ForEach-Object { Write-Host "   $_" }
if ($LASTEXITCODE -ne 0) {
    Warn "수신이 비었거나 이상합니다. 확인 순서:"
    Info "1) 대회 PC MORAI의 목적지 IP가 $MyIp 인지   2) 분배기 창의 포트별 pkt/s가 0이 아닌지"
    Info "3) 분배기 창에 숫자가 있는데 WSL이 비면: WSL IP($wslIp)가 바뀌었는지 -> 이 스크립트를 다시 실행"
    exit 1
}
Write-Host ""
Write-Host "준비 완료. WSL 터미널에서:" -ForegroundColor Green
Write-Host "    cd $repoWsl && python3 AIM2026/main.py --config AIM2026/config_competition.yaml --competition" -ForegroundColor White
