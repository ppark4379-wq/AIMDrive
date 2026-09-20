#Requires -Version 5.1
<#
.SYNOPSIS
대회장에서 MORAI -> WSL 추론 경로를 한 번에 세우고 검증한다.

.DESCRIPTION
시뮬레이터는 대회 PC에서 돌고, 추론은 이 PC의 WSL에서 돈다.  WSL2는 NAT라
랜으로 들어온 UDP가 Windows 호스트에서 멈추므로 inference/udp_relay.py가
한 번 더 넘겨줘야 하고, 포트 9090 하나로 섞여 오는 카메라/GPS/IMU 여섯 개를
송신 포트로 갈라주는 일도 같은 relay가 한다.

이 스크립트가 하는 일:

    1. 물리 어댑터를 찾고 IP를 설정한다          - 고정(-LanIp) 또는 자동(-AutoIp)
    2. 인바운드 방화벽 규칙을 보장한다           - 포트별 + python 프로그램별
    3. 협상된 링크 속도를 확인한다               - 카메라 4대 + LiDAR는 약 50 Mbps
    4. 대회 PC에 알려줄 이 PC의 랜 IP를 출력한다
    5. Windows 쪽에서 시뮬레이터 패킷이 오는지 실측한다
    6. relay를 전용 창에 띄운다
    7. WSL 쪽에서 여덟 개 포트가 다 들어오는지 실측한다

어느 단계에서 막혔는지가 바로 원인이 되도록, 앞 단계가 실패하면 그 자리에서
멈춘다.  시뮬레이터 방향으로는 아무것도 보내지 않는다.

대회 PC의 포트 배치가 우리와 다를 수 있다.  -AutoPorts를 주면 포트를 가정하지
않고 inference/autodetect_streams.py가 실제로 도착한 패킷의 바이트로 스트림을
가려내어 relay의 --ports/--demux를 만들어 준다.  MORAI UI에 이 PC의 IP를 넣는
일만은 사람이 해야 한다.

.EXAMPLE
    # 로컬 예행연습 (시뮬레이터가 이 PC에서 도는 경우)
    .\competition_setup.ps1

.EXAMPLE
    # 대회장 - 관리자 PowerShell, 대역도 포트도 모르는 채로
    .\competition_setup.ps1 -Auto

.EXAMPLE
    # 대역을 지정받았을 때
    .\competition_setup.ps1 -LanIp 192.168.100.2 -Peer 192.168.100.1 -AutoPorts

.EXAMPLE
    # 끝나고 원래 학교망 설정으로 되돌리기
    .\competition_setup.ps1 -Restore
#>
[CmdletBinding()]
param(
    # 이 PC에 설정할 고정 IP.  생략하면 현재 설정을 그대로 쓰고 출력만 한다.
    [string]$LanIp,
    [string]$Mask = "255.255.255.0",
    [string]$Gateway,
    # 대회 PC의 IP.  주면 ping으로 케이블/대역을 먼저 확인한다.
    [string]$Peer,
    # 쓸 어댑터 이름.  지역변수 $adapter 와 겹치지 않도록 이름을 달리한다.
    [string]$AdapterName,
    # 같은 용도의 숫자 지정.  한글 어댑터 이름이 콘솔 인코딩에 깨지는 환경에서
    # 안전하다.  Get-NetAdapter 의 InterfaceIndex 값을 그대로 준다.
    [int]$AdapterIndex,
    # 생략하면 이 스크립트의 \\wsl.localhost 경로에서 자동 탐지한다.
    [string]$Distro,
    [string]$RepoPath,
    # 생략하면 사용자별 miniconda/anaconda/일반 Python을 순서대로 찾는다.
    [string]$WinPython,
    [int]$ProbeSeconds = 8,
    # -LanIp이 저장해 둔 직전 설정으로 되돌린다.
    [switch]$Restore,
    # relay를 띄우지 않고 검사만 한다.
    [switch]$NoRelay,
    # -AutoIp과 -AutoPorts를 한꺼번에.
    [switch]$Auto,
    # IP를 DHCP로, 없으면 APIPA로 받는다.  대역을 미리 몰라도 된다.
    [switch]$AutoIp,
    # 포트와 demux를 실제로 도착한 패킷에서 알아낸다.
    [switch]$AutoPorts
)

if ($Auto) { $AutoIp = $true; $AutoPorts = $true }

$ErrorActionPreference = "Stop"

# 시뮬레이터가 쏘는 목적지 포트.  909 Vehicle Status, 2368 LiDAR,
# 9090 카메라 4대 + GPS + IMU 공유.
$InboundPorts = @(909, 2368, 9090)
# relay가 갈라 준 뒤 인코더가 듣는 포트.  configs/vision_encoder_realtime.json
# 의 ports 블록과 1:1로 같아야 한다.  해시테이블이 아니라 목록인 이유는
# OrderedDictionary를 정수로 인덱싱하면 키가 아니라 위치로 해석되기 때문이다.
$EncoderPorts = @(
    @{ Port = 2368;  Name = "lidar" }
    @{ Port = 10501; Name = "rgb_front" }
    @{ Port = 10502; Name = "rgb_left" }
    @{ Port = 10503; Name = "rgb_right" }
    @{ Port = 10504; Name = "rgb_rear" }
    @{ Port = 10506; Name = "gps" }
    @{ Port = 10507; Name = "imu" }
    @{ Port = 10909; Name = "vehicle_status" }
)
# 카메라 4대 약 41 Mbps + LiDAR 약 7 Mbps.  100 Mbps 링크면 절반을 쓰게 되고,
# 65000바이트 datagram이 IP 조각 약 45개로 쪼개지므로 버스트에서 위험하다.
$MeasuredLoadMbps = 50
$FirewallRule = "MORAI UDP in (competition)"
$FirewallProgramRule = "MORAI UDP in (competition, python)"
$StatePath = Join-Path $env:LOCALAPPDATA "morai_competition_setup.json"

function Write-Step { param([string]$Text) Write-Host ""; Write-Host "== $Text" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Text) Write-Host "   [OK]   $Text" -ForegroundColor Green }
function Write-Warn { param([string]$Text) Write-Host "   [주의] $Text" -ForegroundColor Yellow }
function Write-Info { param([string]$Text) Write-Host "          $Text" -ForegroundColor DarkGray }
function Stop-With {
    param([string]$Text, [string[]]$Hints = @())
    Write-Host "   [실패] $Text" -ForegroundColor Red
    foreach ($hint in $Hints) { Write-Host "          $hint" -ForegroundColor DarkGray }
    exit 1
}

function ConvertTo-IPMask {
    param([int]$PrefixLength)
    $bits = ([uint32]::MaxValue) -shl (32 - $PrefixLength)
    return (3, 2, 1, 0 | ForEach-Object { ($bits -shr ($_ * 8)) -band 0xFF }) -join '.'
}

# -Restore가 되돌릴 수 있도록 현재 IP 설정을 남긴다.  DHCP/APIPA로 바꾸든
# 고정 IP를 넣든 바꾸기 전에 한 번만 부른다.
function Save-CurrentIPConfig {
    # 이름이 $Adapter면 스크립트 인자(어댑터 이름 문자열)를 가려서 헷갈린다.
    param($Nic)
    if (Test-Path $StatePath) { return }   # 첫 변경 시점의 설정만 원본이다
    $current = Get-NetIPConfiguration -InterfaceIndex $Nic.InterfaceIndex
    $v4 = $current.IPv4Address | Select-Object -First 1
    @{
        Alias   = $Nic.Name
        Dhcp    = (Get-NetIPInterface -InterfaceIndex $Nic.InterfaceIndex -AddressFamily IPv4).Dhcp -eq "Enabled"
        Address = $v4.IPAddress
        Mask    = (ConvertTo-IPMask $v4.PrefixLength)
        Gateway = ($current.IPv4DefaultGateway | Select-Object -First 1).NextHop
    } | ConvertTo-Json | Set-Content -Path $StatePath -Encoding UTF8
    Write-Info "직전 설정 저장: $StatePath"
}

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    return ([Security.Principal.WindowsPrincipal]$identity).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-PhysicalAdapter {
    # 이름을 받았으면 그것만 쓴다.  대회장에서 유선과 무선이 함께 살아 있으면
    # 자동 선택이 무선을 집을 수 있고, 그러면 엉뚱한 IP를 대회 PC에 알려주게 된다.
    $named = $null
    if ($AdapterIndex) {
        $named = Get-NetAdapter -InterfaceIndex $AdapterIndex -ErrorAction SilentlyContinue |
                 Select-Object -First 1
        if (-not $named) { Stop-With "InterfaceIndex $AdapterIndex 인 어댑터가 없습니다." @(
            "Get-NetAdapter 로 번호를 확인하세요.") }
    }
    elseif ($AdapterName) {
        # -Name 은 와일드카드를 받으므로, 이름이 깨져 들어오면 엉뚱한 것이 잡힌다.
        $named = Get-NetAdapter -Name $AdapterName -ErrorAction SilentlyContinue |
                 Select-Object -First 1
        if (-not $named -or -not $named.Name) {
            Stop-With "그런 어댑터가 없습니다: $AdapterName" @(
                "한글 이름이 깨졌을 수 있습니다. -AdapterIndex 숫자로 지정하세요.")
        }
    }
    if ($named) {
        if ($named.Status -ne "Up") {
            Stop-With "$($named.Name) 이(가) Up이 아닙니다 (현재: $($named.Status))." @(
                "랜 케이블 연결과 반대편 포트를 확인하세요.")
        }
        return $named
    }

    $up = @(Get-NetAdapter -Physical | Where-Object { $_.Status -eq "Up" })
    if ($up.Count -eq 0) {
        Stop-With "연결된 물리 어댑터가 없습니다." @("랜 케이블이 꽂혀 있는지 확인하세요.")
    }
    # 유선을 무선보다 먼저 고른다.  대회장에서 Wi-Fi가 같이 살아 있을 때 무선
    # IP를 대회 PC에 알려주면 패킷은 영영 오지 않는다.
    $wired = @($up | Where-Object { $_.PhysicalMediaType -notlike "*802.11*" })
    $pick = if ($wired.Count -gt 0) { $wired } else { $up }
    if ($pick.Count -gt 1) {
        Write-Warn "어댑터가 여러 개 살아 있습니다. 첫 번째를 씁니다: $($pick[0].Name)"
        foreach ($a in $pick) { Write-Info "$($a.Name)  -  $($a.InterfaceDescription)" }
        Write-Info "못박으려면 -AdapterIndex $($pick[0].InterfaceIndex) 처럼 지정하세요."
    }
    return $pick[0]
}

# discover_udp.py의 표를 (포트 -> 패킷 수)로 읽는다.  머리글과 구분선은 건너뛴다.
function Read-ProbeTable {
    param([string[]]$Lines)
    $seen = @{}
    foreach ($line in $Lines) {
        if ($line -match '^\s*(\d+)\s+(\d+)\s') { $seen[[int]$Matches[1]] = [int]$Matches[2] }
    }
    return $seen
}

# 개인 사용자명이나 배포판 이름을 저장소에 고정하지 않는다. 보통 이 파일은
# \\wsl.localhost\<distro>\... 경로에서 실행되므로 그 실제 위치가 기준이다.
$scriptRootMatch = [regex]::Match(
    $PSScriptRoot,
    '^\\\\wsl(?:\.localhost|\$)\\(?<distro>[^\\]+)\\(?<path>.+)$',
    [Text.RegularExpressions.RegexOptions]::IgnoreCase)
if (-not $Distro -and $scriptRootMatch.Success) {
    $Distro = $scriptRootMatch.Groups['distro'].Value
}
if (-not $Distro) {
    $available = @(& wsl.exe -l -q 2>$null | ForEach-Object { "$($_)".Trim() } |
                   Where-Object { $_ })
    if ($available.Count -eq 1) { $Distro = $available[0] }
    else {
        Stop-With "WSL 배포판을 자동 선택할 수 없습니다." @(
            "-Distro <이름>을 지정하세요. wsl -l -q 로 목록을 볼 수 있습니다.")
    }
}
if (-not $RepoPath) {
    if ($scriptRootMatch.Success) {
        $scriptRootWsl = '/' + ($scriptRootMatch.Groups['path'].Value -replace '\\', '/')
        $RepoPath = $scriptRootWsl -replace '/inference$', ''
    }
    else {
        Stop-With "WSL 저장소 경로를 자동 탐지할 수 없습니다: $PSScriptRoot" @(
            "-RepoPath /home/<사용자>/AIMDrive/VLDrive 를 지정하세요.")
    }
}
if (-not $WinPython) {
    $pythonCandidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\python.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\python.exe")
    )
    $pythonCandidates += @(Get-ChildItem (Join-Path $env:LOCALAPPDATA "Programs\Python\Python*\python.exe") -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
    $commandPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($commandPython) { $pythonCandidates += $commandPython.Source }
    $WinPython = $pythonCandidates | Where-Object { $_ -and (Test-Path $_) } |
                 Select-Object -First 1
    if (-not $WinPython) {
        Stop-With "Windows Python을 자동으로 찾지 못했습니다." @(
            "-WinPython C:\\path\\to\\python.exe 를 지정하세요.")
    }
}

$repoWin = Split-Path -Parent $PSScriptRoot
$discoverWin = Join-Path $repoWin "inference\discover_udp.py"
$autodetectWin = Join-Path $repoWin "inference\autodetect_streams.py"
$relayWin = Join-Path $repoWin "inference\udp_relay.py"
$discoverWsl = "$RepoPath/inference/discover_udp.py"
$configWin = Join-Path $repoWin "configs\vision_encoder_realtime.json"
# -AutoPorts가 채우면 그 값으로, 아니면 udp_relay.py의 기본 배치로 중계한다.
$relayPorts = $null
$relayDemux = $null

Write-Host ""
Write-Host "MORAI -> WSL 실시간 추론 경로 준비" -ForegroundColor White
Write-Host "대회 PC(시뮬) --랜--> 이 PC(Windows, relay) --> WSL($Distro, 추론)" -ForegroundColor DarkGray

$admin = Test-Admin

# --------------------------------------------------------------- 0. 원복 모드
if ($Restore) {
    Write-Step "저장해 둔 IP 설정으로 되돌리기"
    if (-not (Test-Path $StatePath)) {
        Stop-With "저장된 설정이 없습니다: $StatePath" @(
            "-LanIp으로 IP를 바꾼 적이 있어야 되돌릴 수 있습니다.")
    }
    if (-not $admin) { Stop-With "관리자 권한이 필요합니다." @("PowerShell을 관리자로 다시 여세요.") }
    $saved = Get-Content $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($saved.Dhcp) {
        netsh interface ipv4 set address name="$($saved.Alias)" source=dhcp | Out-Null
        Write-Ok "$($saved.Alias) -> DHCP"
    }
    else {
        $restoreArgs = @("interface", "ipv4", "set", "address", "name=$($saved.Alias)",
                         "static", $saved.Address, $saved.Mask)
        if ($saved.Gateway) { $restoreArgs += $saved.Gateway }
        netsh @restoreArgs | Out-Null
        Write-Ok "$($saved.Alias) -> $($saved.Address)/$($saved.Mask) gw=$($saved.Gateway)"
    }
    Remove-Item $StatePath -Force
    exit 0
}

# ------------------------------------------------------------- 1. 어댑터 / IP
Write-Step "물리 어댑터"
$adapter = Get-PhysicalAdapter
Write-Ok "$($adapter.Name)  -  $($adapter.InterfaceDescription)"

if ($LanIp) {
    if (-not $admin) {
        Stop-With "IP를 바꾸려면 관리자 권한이 필요합니다." @(
            "PowerShell을 관리자로 열어 다시 실행하거나, 아래를 직접 치세요:",
            "netsh interface ipv4 set address name=`"$($adapter.Name)`" static $LanIp $Mask")
    }
    Save-CurrentIPConfig $adapter
    $setArgs = @("interface", "ipv4", "set", "address", "name=$($adapter.Name)", "static", $LanIp, $Mask)
    if ($Gateway) { $setArgs += $Gateway }
    netsh @setArgs | Out-Null
    Start-Sleep -Seconds 2
    Write-Ok "$($adapter.Name) -> $LanIp / $Mask"
}
elseif ($AutoIp) {
    # 대회 PC가 대역을 알려주지 않아도 되도록, DHCP를 먼저 시도하고 없으면
    # APIPA(169.254.x.x)를 받는다.  직결 케이블이면 양쪽이 APIPA로 자기 주소를
    # 정하고 그대로 통신되므로 이것으로 충분하다.
    if (-not $admin) {
        Stop-With "-AutoIp으로 IP를 받으려면 관리자 권한이 필요합니다." @(
            "PowerShell을 관리자로 열어 다시 실행하세요.")
    }
    Write-Step "IP 자동 획득 (DHCP -> 실패 시 APIPA)"
    Save-CurrentIPConfig $adapter
    netsh interface ipv4 set address name="$($adapter.Name)" source=dhcp | Out-Null
    # DHCP 서버가 없으면 Windows가 APIPA로 내려오기까지 1분 가까이 걸린다.
    # 그 전에 포기하면 주소가 없는 채로 다음 단계에 들어가므로 넉넉히 기다린다.
    $deadline = (Get-Date).AddSeconds(75)
    $got = $null
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        $got = Get-NetIPAddress -InterfaceIndex $adapter.InterfaceIndex -AddressFamily IPv4 `
                   -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($got -and $got.IPAddress -notlike "169.254.*") { break }
        Write-Info "주소 대기 중... ($([int]($deadline - (Get-Date)).TotalSeconds)초 남음)"
    }
    if (-not $got) { Stop-With "IPv4 주소를 받지 못했습니다." @("-LanIp으로 직접 지정하세요.") }
    if ($got.IPAddress -like "169.254.*") {
        Write-Ok "$($got.IPAddress) (APIPA - DHCP 서버가 없는 직결 구성)"
        Write-Info "대회 PC도 자동 설정이면 169.254.x.x를 받아 서로 통신됩니다."
    }
    else {
        Write-Ok "$($got.IPAddress) (DHCP)"
    }
}

$config = Get-NetIPConfiguration -InterfaceIndex $adapter.InterfaceIndex
$myIp = ($config.IPv4Address | Select-Object -First 1).IPAddress
if (-not $myIp) { Stop-With "어댑터에 IPv4 주소가 없습니다." @("-LanIp으로 고정 IP를 지정하세요.") }

# ----------------------------------------------------------------- 2. 방화벽
Write-Step "인바운드 방화벽 (UDP $($InboundPorts -join ', '))"
$rule = Get-NetFirewallRule -DisplayName $FirewallRule -ErrorAction SilentlyContinue
if ($rule) {
    Write-Ok "규칙이 이미 있습니다."
}
elseif ($admin) {
    New-NetFirewallRule -DisplayName $FirewallRule -Direction Inbound -Protocol UDP `
        -LocalPort $InboundPorts -Action Allow -Profile Any | Out-Null
    Write-Ok "규칙을 추가했습니다."
}
else {
    Write-Warn "규칙이 없고 관리자 권한도 없습니다. 대회 PC에서 온 패킷이 차단될 수 있습니다."
    Write-Info "관리자 PowerShell에서: New-NetFirewallRule -DisplayName `"$FirewallRule`" -Direction Inbound -Protocol UDP -LocalPort $($InboundPorts -join ',') -Action Allow -Profile Any"
}

# 위 규칙은 우리 기본 배치의 세 포트만 덮는다.  대회 PC가 다른 포트를 쓰면 탐지
# 자체가 막히므로, 중계/탐지를 돌리는 python에는 포트와 무관하게 허용해 둔다.
if (Get-NetFirewallRule -DisplayName $FirewallProgramRule -ErrorAction SilentlyContinue) {
    Write-Ok "python 프로그램 규칙이 이미 있습니다."
}
elseif ($admin) {
    New-NetFirewallRule -DisplayName $FirewallProgramRule -Direction Inbound -Protocol UDP `
        -Program $WinPython -Action Allow -Profile Any | Out-Null
    Write-Ok "python 프로그램 규칙을 추가했습니다 (임의 포트 대응)."
}
else {
    Write-Warn "python 프로그램 규칙이 없습니다. 기본 포트가 아니면 차단될 수 있습니다."
}

# --------------------------------------------------------------- 3. 링크 속도
Write-Step "링크 속도"
$speed = $adapter.LinkSpeed
$bps = (Get-NetAdapter -Name $adapter.Name).Speed
$mbps = [math]::Round($bps / 1MB)
if ($mbps -lt $MeasuredLoadMbps * 2) {
    Write-Warn "$speed 로 협상됐습니다. 카메라 4대 + LiDAR가 약 $MeasuredLoadMbps Mbps를 씁니다."
    Write-Info "여유가 부족하면 65000바이트 datagram의 IP 조각이 유실되어 카메라 프레임이 통째로 버려집니다."
    Write-Info "Cat5e 이상 케이블로 바꾸면 기가비트로 협상될 수 있습니다 (이 NIC는 2.5GbE)."
}
else {
    Write-Ok "$speed  (필요량 약 $MeasuredLoadMbps Mbps)"
}

# ------------------------------------------------------- 4. 대회 PC에 알릴 IP
Write-Step "대회 PC MORAI에 넣을 목적지 IP"
Write-Host "          $myIp" -ForegroundColor White
Write-Info "포트는 그대로 두세요: Vehicle Status 909, LiDAR 2368, 카메라/GPS/IMU 9090"
Write-Info "센서 7개는 sensor set 파일을 고쳐 불러오면 손으로 안 넣어도 됩니다:"
Write-Info "  python inference/morai_sensorset_retarget.py <sensorset>.json --dest-ip $myIp --host-ip <대회PC IP>"
Write-Info "Competition Vehicle Status(909)만 Network Settings에서 직접 바꾸세요."

if ($Peer) {
    Write-Step "대회 PC 연결 확인 ($Peer)"
    if (Test-Connection -ComputerName $Peer -Count 2 -Quiet) { Write-Ok "ping 응답 있음" }
    else {
        Stop-With "ping이 닿지 않습니다." @(
            "같은 대역인지, 케이블이 꽂혔는지, 대회 PC 방화벽이 ICMP를 막는지 확인하세요.",
            "(ICMP만 막힌 것일 수도 있으니 -Peer 없이 다시 돌려 5단계 결과를 보세요.)")
    }
}

# ---------------------------------------------------- 5. Windows 쪽 수신 실측
if (-not (Test-Path $WinPython)) {
    Stop-With "Windows python을 찾을 수 없습니다: $WinPython" @("-WinPython 으로 경로를 지정하세요.")
}
if (-not (Test-Path $discoverWin)) {
    Stop-With "WSL 저장소에 접근할 수 없습니다: $discoverWin" @(
        "-Distro / -RepoPath 가 맞는지 확인하세요. (현재: $Distro, $RepoPath)")
}

Write-Step "Windows 쪽 수신 실측 ($ProbeSeconds 초)"
# relay가 이미 돌고 있으면 같은 포트를 잡고 있다.  Windows는 SO_REUSEADDR로
# 두 소켓의 bind를 허용하지만 datagram은 한쪽에만 가므로, 먼저 비우지 않으면
# 시뮬레이터가 멀쩡히 쏘는데도 0 패킷으로 읽힌다.
$running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -like "*udp_relay.py*" })
if ($NoRelay -and $running.Count -gt 0) {
    Write-Warn "relay가 이미 실행 중이라 (PID $($running[0].ProcessId)) 실측을 건너뜁니다."
    Write-Info "-NoRelay 없이 다시 실행하면 relay를 재기동하면서 정확히 측정합니다."
}
else {
    foreach ($proc in $running) {
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Info "포트를 비우려고 이전 relay를 종료했습니다 (PID $($proc.ProcessId))."
    }
    if ($running.Count -gt 0) { Start-Sleep -Seconds 1 }
    Write-Info "시뮬레이터에서 차량이 스폰되고 센서가 켜져 있어야 합니다."
    if ($AutoPorts) {
        # 포트를 가정하지 않고, 실제로 도착한 패킷의 바이트로 스트림을 가려낸다.
        $raw = & $WinPython $autodetectWin --config $configWin --seconds $ProbeSeconds --json 2>&1
        $line = ($raw | ForEach-Object { "$_" } | Where-Object { $_.Trim().StartsWith("{") } | Select-Object -Last 1)
        if (-not $line) { Stop-With "스트림 탐지 출력을 읽지 못했습니다." @(($raw | Select-Object -Last 3)) }
        $detected = $line | ConvertFrom-Json
        if (-not $detected.ok) {
            $why = if ($detected.missing) { "찾지 못한 스트림: $($detected.missing -join ', ')" }
                   else { "패킷이 전혀 오지 않았습니다." }
            Stop-With $why @(
                "대회 PC의 MORAI Network Settings 목적지 IP가 $myIp 인지 확인하세요.",
                "차량이 스폰되고 해당 센서 publisher가 켜져 있는지도 보세요.")
        }
        foreach ($name in $detected.streams.PSObject.Properties.Name | Sort-Object) {
            $info = $detected.streams.$name
            Write-Ok ("{0,-16} {1} <- {2}  {3} 패킷" -f $name, $info.dest, $info.source, $info.packets)
        }
        foreach ($warning in $detected.warnings) { Write-Warn $warning }
        $relayPorts = $detected.relay_ports
        $relayDemux = $detected.relay_demux
        Write-Info "relay 인자: --ports $relayPorts --demux $relayDemux"
    }
    else {
        $out = & $WinPython $discoverWin --bind 0.0.0.0 --ports ($InboundPorts -join ',') --seconds $ProbeSeconds 2>&1
        $seen = Read-ProbeTable ($out | ForEach-Object { "$_" })
        $missing = @($InboundPorts | Where-Object { -not $seen.ContainsKey($_) })
        foreach ($port in $InboundPorts) {
            if ($seen.ContainsKey($port)) { Write-Ok "$port  $($seen[$port]) 패킷" }
        }
        if ($missing.Count -gt 0) {
            Stop-With "도착하지 않은 포트: $($missing -join ', ')" @(
                "대회 PC의 MORAI Network Settings 목적지 IP가 $myIp 인지 확인하세요.",
                "방화벽 규칙과 링크 상태도 위 단계 결과를 다시 보세요.",
                "포트 배치가 다른 것 같으면 -AutoPorts 로 다시 실행하세요.")
        }
    }
}

# -------------------------------------------------------------- 6. relay 기동
$wslIp = (& wsl.exe -d $Distro hostname -I).Trim().Split(" ")[0]
if (-not $wslIp) { Stop-With "WSL IP를 읽지 못했습니다." @("wsl.exe -d $Distro hostname -I 를 직접 실행해 보세요.") }

$relayArgs = @($relayWin, "--listen", "0.0.0.0", "--to", $wslIp)
if ($relayPorts) { $relayArgs += @("--ports", $relayPorts) }
if ($relayDemux) { $relayArgs += @("--demux", $relayDemux) }

if ($NoRelay) {
    Write-Step "relay (건너뜀: -NoRelay)"
    Write-Info "직접 띄우려면: $WinPython $($relayArgs -join ' ')"
    exit 0
}

Write-Step "relay 기동 -> WSL $wslIp"
Start-Process -FilePath $WinPython -ArgumentList $relayArgs -WindowStyle Normal
Start-Sleep -Seconds 3
$alive = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
           Where-Object { $_.CommandLine -like "*udp_relay.py*" })
if ($alive.Count -eq 0) {
    Stop-With "relay가 뜨자마자 죽었습니다." @(
        "새로 열린 창의 메시지를 확인하세요. 포트를 다른 프로세스가 잡고 있을 수 있습니다.")
}
Write-Ok "전용 창에서 실행 중 (PID $($alive[0].ProcessId)). 추론이 끝날 때까지 닫지 마세요."

# -------------------------------------------------------- 7. WSL 쪽 수신 실측
Write-Step "WSL 쪽 수신 실측 ($ProbeSeconds 초)"
$ports = ($EncoderPorts | ForEach-Object { $_.Port }) -join ','
$wslOut = & wsl.exe -d $Distro -- /usr/bin/python3 $discoverWsl --bind 0.0.0.0 --ports $ports --seconds $ProbeSeconds 2>&1
$wslSeen = Read-ProbeTable ($wslOut | ForEach-Object { "$_" })
$wslMissing = @()
foreach ($entry in $EncoderPorts) {
    if ($wslSeen.ContainsKey($entry.Port)) {
        Write-Ok ("{0,-6} {1,-16} {2} 패킷" -f $entry.Port, $entry.Name, $wslSeen[$entry.Port])
    }
    else { $wslMissing += $entry }
}
if ($wslMissing.Count -gt 0) {
    $names = ($wslMissing | ForEach-Object { "$($_.Port)($($_.Name))" }) -join ', '
    $hints = @("relay 창을 Ctrl+C로 멈추면 포트별 통계가 나옵니다.")
    if (@($wslMissing | Where-Object { $_.Port -ge 10501 -and $_.Port -le 10507 }).Count -gt 0) {
        $hints += "'no route for sender NNNNN' 이 보이면 시뮬레이터의 송신 포트가 기본값과 다른 것입니다."
        $hints += "그 번호로 relay에 --demux 9090:NNNNN=10501,... 를 붙여 다시 띄우세요."
    }
    Stop-With "WSL까지 오지 않은 포트: $names" $hints
}

# ------------------------------------------------------------------- 8. 마무리
Write-Host ""
Write-Host "준비 완료. WSL 터미널에서 추론만 실행하세요:" -ForegroundColor Green
Write-Host ""
Write-Host "    cd ~/AIMDrive/VLDrive" -ForegroundColor White
Write-Host "    python inference/realtime_vision_encoder.py --config configs/vision_encoder_realtime.json" -ForegroundColor White
Write-Host ""
if ($LanIp) {
    Write-Info "끝나고 학교망으로 되돌리기:  .\competition_setup.ps1 -Restore"
}
