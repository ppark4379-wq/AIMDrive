$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../internal_sensor_record/tools/publish_sensor_record_nas.ps1')
function Assert($condition, $message) { if (!$condition) { throw $message } }
# The fixture uses a unique mutex so it cannot compete with a live NAS job.
# Production still uses the unchanged global singleton lock.
$invoke=(Get-Item Function:Invoke-NasJob).ScriptBlock.ToString()
$testMutex='Local\MoraiDriveNasTest_'+[Guid]::NewGuid().ToString('N')
Assert ($invoke.Contains("'Global\MoraiDriveNasTransferV1'")) 'Production global mutex retained'
Set-Item Function:Invoke-NasJob ([scriptblock]::Create(
    $invoke.Replace("'Global\MoraiDriveNasTransferV1'", "'"+$testMutex+"'")))
$fixture=Join-Path $env:TEMP ('morai-nas-test-'+[Guid]::NewGuid().ToString('N'))
$null=[IO.Directory]::CreateDirectory($fixture)
# Real production copy/hash paths run here; artificial sleeps are forbidden.
function Start-Sleep { throw 'Unexpected transfer delay' }
function Get-HostSample { throw 'Unexpected CPU/RAM/provider sampling' }
try {
    $source=Join-Path $fixture 'source'
    $state=Join-Path $fixture 'state'
    $dest=Join-Path $fixture 'target\lap_001'
    $null=[IO.Directory]::CreateDirectory($source)
    $null=[IO.Directory]::CreateDirectory($state)
    $rows=@()
    foreach($name in @('a.bin','b.bin')) {
        $path=Join-Path $source $name
        $payload=if($name -eq 'a.bin'){[Text.Encoding]::UTF8.GetBytes('payload-'+$name)}else{
            $data=New-Object byte[] (2MB+31)
            $random=New-Object Random 19
            $random.NextBytes($data)
            ,$data
        }
        [IO.File]::WriteAllBytes($path,$payload)
        $item=Get-Item -LiteralPath $path
        $rows+=@{source=$path;relative=$name;kind='file';size=$item.Length;
            mtime_ns=([decimal]($item.LastWriteTimeUtc.Ticks-621355968000000000L)*100)}
    }
    $empty=Join-Path $source 'empty'
    $null=[IO.Directory]::CreateDirectory($empty)
    $rows+=@{source=$empty;relative='empty';kind='directory';size=0;mtime_ns=0}
    $inventory=Join-Path $fixture 'inventory.jsonl'
    [IO.File]::WriteAllLines($inventory,[string[]]@($rows | ForEach-Object {$_|ConvertTo-Json -Compress}))
    $totalBytes=0L
    foreach($row in $rows){$totalBytes += $row.size}
    $job=[pscustomobject]@{schema='morai_nas_transfer_v1';state=$state;destination=$dest;
        sources=@($source);inventory=$inventory;inventory_sha256=(Get-FileHash $inventory).Hash.ToLower();
        identity='test-snapshot';files=2;bytes=$totalBytes}
    [IO.File]::WriteAllText((Join-Path $state 'parent.lease'),'active')
    $script:copyCalls=0
    $originalCopy=(Get-Item Function:Copy-FileWithHash).ScriptBlock
    function Copy-FileWithHash($source,$target) {
        $script:copyCalls++
        if($script:copyCalls -eq 2){throw 'Injected interruption'}
        & $originalCopy $source $target
    }
    $failed=$false
    try { Invoke-NasJob $job } catch { $failed=$true; Assert ($_.Exception.Message -match 'Injected') $_ }
    Assert $failed 'Interruption must fail'
    Assert (![IO.Directory]::Exists($dest)) 'Partial transfer must not publish'
    Assert ([IO.File]::Exists((Join-Path $source 'a.bin'))) 'Local source retained'
    $script:copyCalls=0
    $script:copiedSources=@()
    function Copy-FileWithHash($source,$target) {
        $script:copyCalls++
        $script:copiedSources += [IO.Path]::GetFileName($source)
        if($script:copyCalls -eq 1){throw [IO.IOException]::new('Injected transient I/O error')}
        & $originalCopy $source $target
    }
    [IO.File]::WriteAllText((Join-Path $state 'parent.lease'),'active')
    Invoke-NasJob $job
    Assert ($script:copyCalls -eq 2) 'Transient I/O failure retried immediately'
    Assert (($script:copiedSources -join ',') -eq 'b.bin,b.bin') 'Resume must skip completed receipt'
    $status=Read-Json (Join-Path $state 'status.json')
    Assert ($status.transfer_mode -eq 'unthrottled' -and !$status.throttling_enabled) 'Unthrottled state'
    Assert ($status.status -eq 'copy_and_readback_verified') 'Verified completion state'
    Assert ($status.processed_files -eq 2 -and $status.resumed_files -eq 1 -and
        $status.copied_this_attempt -eq 1 -and $status.verified_files -eq 2) 'Resume/copy/verification counters reflect completed files'

    Set-Item Function:Copy-FileWithHash $originalCopy
    $complete=Read-Json (Join-Path $dest '.NAS_TRANSFER_COMPLETE.json')
    Assert ($complete.files -eq 2 -and !$complete.remote_durability_verified) 'Honest completion'
    Assert ((Get-FileHash (Join-Path $dest 'a.bin')).Hash -eq
        (Get-FileHash (Join-Path $source 'a.bin')).Hash) 'Copy hash'
    [IO.File]::WriteAllText((Join-Path $state 'parent.lease'),'active')
    Invoke-NasJob $job
    Assert ((Get-FileHash (Join-Path $dest 'b.bin')).Hash -eq
        (Get-FileHash (Join-Path $source 'b.bin')).Hash) 'Multi-buffer file hash'
    $extra=Join-Path $dest 'unexpected.txt'
    [IO.File]::WriteAllText($extra,'extra')
    $failed=$false
    try { Invoke-NasJob $job } catch {
        $failed=$true
        Assert ($_.Exception.Message -match 'count/size mismatch') 'Extra file detection'
    }
    Assert $failed 'Extra files must fail final count verification'
    [IO.File]::Delete($extra)
    [IO.File]::WriteAllText((Join-Path $dest 'a.bin'),'XXXXXXXXXXXXX')
    $failed=$false
    try { Invoke-NasJob $job } catch { $failed=$true }
    Assert $failed 'Published corruption must fail'
    Assert ([IO.File]::ReadAllText((Join-Path $dest 'a.bin')) -eq 'XXXXXXXXXXXXX') 'Published data never overwritten'
    $failed=$false
    try { Join-Safe $source '../escape' } catch { $failed=$true }
    Assert $failed 'Traversal rejected'
    # Session JSONs live outside the run/native directories and publish at lap root.
    $sessionName='scenario_campaign_20261001_123456_lap_003'
    $evidenceRows=@()
    foreach($kind in @('apply','plan')) {
        $name=$sessionName+'_native_session_'+$kind+'.json'
        $path=Join-Path $fixture $name
        [IO.File]::WriteAllText($path,(@{session_name=$sessionName;kind=$kind}|ConvertTo-Json))
        $item=Get-Item -LiteralPath $path
        $evidenceRows+=@{source=$path;relative=$name;kind='file';size=$item.Length;
            mtime_ns=([decimal]($item.LastWriteTimeUtc.Ticks-621355968000000000L)*100)}
    }
    $evidenceInventory=Join-Path $fixture 'session-inventory.jsonl'
    [IO.File]::WriteAllLines($evidenceInventory,[string[]]@(
        $evidenceRows | ForEach-Object {$_|ConvertTo-Json -Compress}))
    $evidenceState=Join-Path $fixture 'session-state'
    $null=[IO.Directory]::CreateDirectory($evidenceState)
    [IO.File]::WriteAllText((Join-Path $evidenceState 'parent.lease'),'active')
    $evidenceDest=Join-Path $fixture ('target\'+$sessionName)
    $evidenceBytes=0L
    foreach($row in $evidenceRows){$evidenceBytes += $row.size}
    $evidenceJob=[pscustomobject]@{schema='morai_nas_transfer_v1';state=$evidenceState;
        destination=$evidenceDest;sources=@($evidenceRows | ForEach-Object {$_.source});
        inventory=$evidenceInventory;inventory_sha256=(Get-FileHash $evidenceInventory).Hash.ToLower();
        identity='session-evidence-snapshot';files=2;bytes=$evidenceBytes}
    Invoke-NasJob $evidenceJob
    foreach($row in $evidenceRows) {
        Assert ((Get-FileHash (Join-Path $evidenceDest $row.relative)).Hash -eq
            (Get-FileHash $row.source).Hash) 'Session evidence copied unchanged at lap root'
    }
    $evidenceComplete=Read-Json (Join-Path $evidenceDest '.NAS_TRANSFER_COMPLETE.json')
    Assert ($evidenceComplete.files -eq 2 -and
        $evidenceComplete.status -eq 'copy_and_readback_verified') 'Session evidence counted and verified'
    Write-Host 'PASS: unthrottled copy, retry, resume, hashes, extra-file detection, preservation, session JSON export'
} finally {
    $resolved=[IO.Path]::GetFullPath($fixture)
    $tempRoot=[IO.Path]::GetFullPath($env:TEMP).TrimEnd('\')+'\'
    if(!$resolved.StartsWith($tempRoot,[StringComparison]::OrdinalIgnoreCase) -or
       [IO.Path]::GetFileName($resolved) -notlike 'morai-nas-test-*'){throw 'Unsafe test cleanup'}
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
