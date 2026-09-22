# Windows-native, unthrottled, resumable publication. Dot-source to test functions.
param([string]$JobFile)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding=New-Object Text.UTF8Encoding($false)

function Initialize-NativeWatchdog {
    if ('MoraiNasWatchdog' -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Threading;
using System.Text;
using System.Diagnostics;
public static class MoraiNasWatchdog {
    static Timer timer;
    static long pulse;
    static string lease, directory;
    static readonly object gate=new object();
    static string operation="idle", operationPath="", operationResult="", phase="starting", currentFile="";
    static long operationStarted,totalFiles,processedFiles,copiedFiles,resumedFiles,verifiedFiles;
    static long lastSnapshot,lastConsole;
    static string lastOperation="",lastOperationPath="",lastOperationError="";
    static long lastOperationElapsed;
    static int busy;
    static string Quote(string value) {
        var b=new StringBuilder("\"");
        foreach(char c in value ?? "") {
            if(c=='"' || c=='\\') b.Append('\\').Append(c);
            else if(c<32) b.Append("\\u").Append(((int)c).ToString("x4"));
            else b.Append(c);
        }
        return b.Append('"').ToString();
    }
    public static void Progress(string stage,string file,long total,long processed,long copied,long resumed,long verified) {
        lock(gate) {
            phase=stage;currentFile=file;totalFiles=total;processedFiles=processed;
            copiedFiles=copied;resumedFiles=resumed;verifiedFiles=verified;
        }
    }
    public static void Begin(string name,string path) {
        lock(gate) { operation=name;operationPath=path;operationResult="";operationStarted=Stopwatch.GetTimestamp(); }
    }
    public static void Result(string result) { lock(gate) { operationResult=result; } }
    public static void End(string error) {
        string record=null;
        lock(gate) {
            double elapsed=operationStarted==0?0:1000.0*(Stopwatch.GetTimestamp()-operationStarted)/Stopwatch.Frequency;
            if(elapsed>=1000 || !String.IsNullOrEmpty(error)) record=Snapshot("operation_end",error);
            lastOperation=operation;lastOperationPath=operationPath;lastOperationError=error ?? "";
            lastOperationElapsed=(long)elapsed;
            operation="idle";operationStarted=0;
        }
        if(record!=null && directory!=null) {
            try { File.AppendAllText(Path.Combine(directory,"io_events.jsonl"),record+Environment.NewLine,new UTF8Encoding(false)); }
            catch { /* Observation failure must not change the transfer result. */ }
        }
    }
    static string Snapshot(string kind,string reason) {
        lock(gate) {
            double elapsed=operationStarted==0?0:1000.0*(Stopwatch.GetTimestamp()-operationStarted)/Stopwatch.Frequency;
            int pid; using(var process=Process.GetCurrentProcess()) pid=process.Id;
            return "{\"utc\":"+Quote(DateTime.UtcNow.ToString("o"))+",\"kind\":"+Quote(kind)+
                ",\"reason\":"+Quote(reason)+",\"pid\":"+pid+
                ",\"phase\":"+Quote(phase)+",\"current_file\":"+Quote(currentFile)+
                ",\"operation\":"+Quote(operation)+",\"path\":"+Quote(operationPath)+
                ",\"result\":"+Quote(operationResult)+",\"operation_elapsed_ms\":"+((long)elapsed)+",\"total_files\":"+totalFiles+
                ",\"processed_files\":"+processedFiles+",\"copied_this_attempt\":"+copiedFiles+
                ",\"resumed_files\":"+resumedFiles+",\"verified_files\":"+verifiedFiles+
                ",\"last_completed_operation\":"+Quote(lastOperation)+",\"last_completed_path\":"+Quote(lastOperationPath)+
                ",\"last_completed_error\":"+Quote(lastOperationError)+",\"last_completed_elapsed_ms\":"+lastOperationElapsed+
                ",\"pulse_age_ms\":"+((DateTime.UtcNow.Ticks-Interlocked.Read(ref pulse))/TimeSpan.TicksPerMillisecond)+"}";
        }
    }
    static void SaveSnapshot(string name,string text) {
        string target=Path.Combine(directory,name),temp=target+".tmp";
        File.WriteAllText(temp,text,new UTF8Encoding(false));
        if(File.Exists(target)) File.Replace(temp,target,null); else File.Move(temp,target);
    }
    static void Fail(string reason) {
        try { SaveSnapshot("watchdog_failure.json",Snapshot("watchdog_exit",reason)); }
        catch(Exception ex) { Console.Error.WriteLine("[NAS] Watchdog diagnostic write failed: "+ex.Message); }
        Console.Error.WriteLine("[NAS] Watchdog exit: "+reason);
        Environment.Exit(4);
    }
    public static void Pulse() { Interlocked.Exchange(ref pulse,DateTime.UtcNow.Ticks); }
    public static void Watch(string path) {
        lease=path;directory=Path.GetDirectoryName(path);lastSnapshot=0;lastConsole=0;Pulse();
        lock(gate) { operation="idle";operationStarted=0;lastOperation="";lastOperationPath="";lastOperationError="";lastOperationElapsed=0; }
        string previous=Path.Combine(directory,"watchdog_failure.json");
        if(File.Exists(previous)) File.Move(previous,Path.Combine(directory,"watchdog_failure.previous-"+Guid.NewGuid().ToString("N")+".json"));
        timer=new Timer(_ => {
            if(Interlocked.Exchange(ref busy,1)!=0) return;
            try {
                if(!File.Exists(lease)) { Fail("parent_lease_missing");return; }
                if((DateTime.UtcNow-File.GetLastWriteTimeUtc(lease)).TotalSeconds>20) {
                    Fail("parent_lease_stale_over_20_seconds");return;
                }
                if((DateTime.UtcNow-new DateTime(Interlocked.Read(ref pulse),DateTimeKind.Utc)).TotalSeconds>120) {
                    Fail("no_progress_over_120_seconds");return;
                }
                long now=DateTime.UtcNow.Ticks;
                if(now-lastSnapshot>=TimeSpan.FromSeconds(5).Ticks) {
                    lastSnapshot=now;
                    try { SaveSnapshot("operation.json",Snapshot("observation","")); }
                    catch(Exception ex) { Console.Error.WriteLine("[NAS] Observation write failed: "+ex.Message); }
                }
                lock(gate) {
                    double seconds=operationStarted==0?0:(double)(Stopwatch.GetTimestamp()-operationStarted)/Stopwatch.Frequency;
                    if(seconds>=10 && now-lastConsole>=TimeSpan.FromSeconds(10).Ticks) {
                        lastConsole=now;
                        Console.WriteLine("[NAS] WAIT {0:F1}s; operation={1}; path={2}; file={3}",seconds,operation,operationPath,currentFile);
                    }
                }
            } catch(Exception ex) { Fail("watchdog_exception: "+ex.Message); }
            finally { Interlocked.Exchange(ref busy,0); }
        },null,1000,1000);
    }
    public static void Stop() { if(timer!=null) timer.Dispose(); }
}
'@
}

function Invoke-NasIo([string]$operation,[string]$path,[scriptblock]$action) {
    $observing=$null -ne ('MoraiNasWatchdog' -as [type])
    if($observing){[MoraiNasWatchdog]::Begin($operation,$path)}
    $failure=''
    try {
        $ioResult=& $action
        if($observing -and ($ioResult -is [bool] -or $ioResult -is [int] -or $ioResult -is [long])){
            [MoraiNasWatchdog]::Result([string]$ioResult)
        }
        $ioResult
    }
    catch { $failure=$_.Exception.ToString(); throw }
    finally { if($observing){[MoraiNasWatchdog]::End($failure)} }
}

function Set-NasProgress {
    [MoraiNasWatchdog]::Progress($script:phase,$script:currentFile,$script:totalFiles,
        $script:processedFiles,$script:copiedFiles,$script:resumedFiles,$script:verifiedFiles)
}

function Write-JsonAtomic([string]$path, $value) {
    $temp = $path + '.tmp'
    Invoke-NasIo 'write_json' $temp { [IO.File]::WriteAllText($temp, ($value | ConvertTo-Json -Depth 8 -Compress),
        (New-Object Text.UTF8Encoding($false))) }
    if ([IO.File]::Exists($path)) { (Invoke-NasIo 'replace_json' $path { [IO.File]::Replace($temp,$path,[NullString]::Value) }) }
    else { (Invoke-NasIo 'publish_json' $path { [IO.File]::Move($temp,$path) }) }
}

function Read-Json([string]$path) {
    return ((Invoke-NasIo 'read_json' $path { [IO.File]::ReadAllText($path) }) | ConvertFrom-Json)
}

function Assert-PlainPath([string]$path) {
    # Check ancestors too: a junction in the destination can escape the target.
    $cursor = [IO.Path]::GetFullPath($path)
    while ($cursor) {
        # One fresh metadata lookup per component. Missing paths are allowed for
        # creation, but access/network failures must propagate, not look absent.
        $attributes = Invoke-NasIo 'get_attributes' $cursor {
            try { [IO.File]::GetAttributes($cursor) }
            catch [IO.FileNotFoundException] { $null }
            catch [IO.DirectoryNotFoundException] { $null }
        }
        if ($null -ne $attributes -and ($attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Linked path refused: $cursor"
        }
        $parent = [IO.Directory]::GetParent($cursor)
        $cursor = if ($null -eq $parent) { $null } else { $parent.FullName }
    }
}

function Join-Safe([string]$root, [string]$relative) {
    if ([string]::IsNullOrEmpty($relative) -or [IO.Path]::IsPathRooted($relative)) {
        throw 'Invalid relative path'
    }
    foreach ($part in ($relative -split '[/\\]')) {
        if ($part -eq '..' -or $part -eq '.' -or $part -eq '' -or $part.Contains(':') -or
            $part.EndsWith('.') -or $part.EndsWith(' ')) { throw 'Unsafe path component' }
    }
    $prefix = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
    $path = [IO.Path]::GetFullPath((Join-Path $root $relative))
    if (!$path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Path escapes root' }
    Assert-PlainPath $path
    return $path
}

function Write-State([string]$status, [string]$reason='') {
    $script:phase=$status
    Set-NasProgress
    Write-JsonAtomic (Join-Path $script:state 'status.json') @{
        status=$status; reason=$reason; utc=[DateTime]::UtcNow.ToString('o');
        transfer_mode='unthrottled'; throttling_enabled=$false;
        verification_scope='client_readback'; remote_durability_verified=$false;
        current_file=$script:currentFile; total_files=$script:totalFiles;
        processed_files=$script:processedFiles; copied_this_attempt=$script:copiedFiles;
        resumed_files=$script:resumedFiles; verified_files=$script:verifiedFiles
    }
}

function Update-Transfer {
    # Liveness and logging only: no host sampling, budgets, pauses or sleeps.
    [MoraiNasWatchdog]::Pulse()
    Set-NasProgress
    $now=$script:clock.Elapsed.TotalSeconds
    if ($now -lt $script:nextLog) { return }
    $script:nextLog=$now+10
    $entry=@{utc=[DateTime]::UtcNow.ToString('o'); status=$script:phase;
        transfer_mode='unthrottled'; throttling_enabled=$false;
        current_file=$script:currentFile; total_files=$script:totalFiles;
        processed_files=$script:processedFiles; copied_this_attempt=$script:copiedFiles;
        resumed_files=$script:resumedFiles; verified_files=$script:verifiedFiles}
    [IO.File]::AppendAllText((Join-Path $script:state 'telemetry.jsonl'),
        ($entry | ConvertTo-Json -Compress) + [Environment]::NewLine)
    Write-Host ('[NAS] Unthrottled; phase={0}; checked={1}/{2}; copied={3}; resumed={4}; verified={5}; file={6}' -f
        $script:phase,$script:processedFiles,$script:totalFiles,$script:copiedFiles,
        $script:resumedFiles,$script:verifiedFiles,$script:currentFile)
}

function Get-RowKey([string]$relative) {
    $h = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($h.ComputeHash([Text.Encoding]::UTF8.GetBytes($relative)))).Replace('-','').ToLowerInvariant() }
    finally { $h.Dispose() }
}

function Get-FileContentHash([string]$path) {
    Assert-PlainPath $path
    $hash = [Security.Cryptography.SHA256]::Create()
    $stream = Invoke-NasIo 'hash_open_read' $path { [IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read) }
    try {
        while (($n=(Invoke-NasIo 'hash_read' $path { $stream.Read($script:buffer,0,$script:buffer.Length) })) -gt 0) {
            Update-Transfer
            $null = $hash.TransformBlock($script:buffer,0,$n,$null,0)
        }
        $null = $hash.TransformFinalBlock([byte[]]@(),0,0)
        return ([BitConverter]::ToString($hash.Hash)).Replace('-','').ToLowerInvariant()
    } finally { Invoke-NasIo 'hash_close' $path { $stream.Dispose() }; $hash.Dispose() }
}

function Copy-FileWithHash([string]$source,[string]$target) {
    Assert-PlainPath $source
    Assert-PlainPath $target
    $inputStream = Invoke-NasIo 'copy_open_source' $source { [IO.File]::Open($source,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read) }
    $outputStream = $null
    $hash = [Security.Cryptography.SHA256]::Create()
    try {
        $outputStream = Invoke-NasIo 'copy_open_target' $target { New-Object IO.FileStream($target,[IO.FileMode]::Create,
            [IO.FileAccess]::Write,[IO.FileShare]::None,$script:buffer.Length,[IO.FileOptions]::None) }
        while (($n=(Invoke-NasIo 'copy_read_source' $source { $inputStream.Read($script:buffer,0,$script:buffer.Length) })) -gt 0) {
            Update-Transfer
            Invoke-NasIo 'copy_write_target' $target { $outputStream.Write($script:buffer,0,$n) }
            $null = $hash.TransformBlock($script:buffer,0,$n,$null,0)
        }
        Invoke-NasIo 'copy_flush_target' $target { $outputStream.Flush($true) }
        $null = $hash.TransformFinalBlock([byte[]]@(),0,0)
        return ([BitConverter]::ToString($hash.Hash)).Replace('-','').ToLowerInvariant()
    } finally {
        Invoke-NasIo 'copy_close_source' $source { $inputStream.Dispose() }; if ($null -ne $outputStream) { Invoke-NasIo 'copy_close_target' $target { $outputStream.Dispose() } }
        $hash.Dispose()
    }
}

function Assert-Source($row) {
    Assert-PlainPath $row.source
    $item = Invoke-NasIo 'source_stat' $row.source { Get-Item -LiteralPath $row.source }
    if ($item.Length -ne $row.size) { throw "Source size changed: $($row.relative)" }
    # Python stat nanoseconds / 100 matches Windows FILETIME ticks (100 ns).
    $actualTicks = $item.LastWriteTimeUtc.Ticks - 621355968000000000L
    $expectedTicks = [long]([decimal]$row.mtime_ns / 100)
    if ([Math]::Abs($actualTicks-$expectedTicks) -gt 1) {
        throw "Source timestamp changed: $($row.relative)"
    }
}

function Get-SafeFiles([string]$root) {
    Assert-PlainPath $root
    foreach ($file in [IO.Directory]::EnumerateFiles($root)) {
        Update-Transfer
        Assert-PlainPath $file
        $file
    }
    foreach ($directory in [IO.Directory]::EnumerateDirectories($root)) {
        Update-Transfer
        Assert-PlainPath $directory
        Get-SafeFiles $directory
    }
}


function Publish-NasDirectory {
    param(
        [string]$pending, [string]$destination,
        [ValidateRange(1,5)][int]$maxAttempts=5,
        [ValidateRange(0,10000)][int]$retryDelayMilliseconds=2000
    )
    # Retry only Windows sharing/lock violations. Never overwrite a destination,
    # reconnect a user's drive, or mask an unrelated permissions/network error.
    for($attempt=1; $attempt -le $maxAttempts; $attempt++){
        Update-Transfer
        try {
            Invoke-NasIo 'publish_directory' $destination { [IO.Directory]::Move($pending,$destination) }
            return
        } catch {
            $cause=$_.Exception.GetBaseException()
            $win32=$cause.HResult -band 0xffff
            if($cause -isnot [IO.IOException] -or $win32 -notin @(32,33)){throw}
            if($attempt -eq $maxAttempts){
                throw [IO.IOException]::new(
                    "NAS final folder rename blocked after $attempt attempts (Win32=$win32). " +
                    "All files passed readback verification; pending folder retained: $pending. " +
                    "A sharing violation does not identify the lock owner. Inspect open handles and " +
                    "the drive provider's log; persistent provider references can require a normal reconnect.",
                    $cause)
            }
            $delay=[int]($retryDelayMilliseconds * [Math]::Pow(2,$attempt-1))
            Write-State 'publishing' "Final folder rename blocked (Win32=$win32); retry $($attempt+1)/$maxAttempts in $($delay/1000)s"
            Write-Host "[NAS] Final folder rename waiting: Win32=$win32; retry=$($attempt+1)/$maxAttempts; wait=$($delay/1000)s; path=$destination"
            Invoke-NasIo 'publish_retry_wait' $destination { Start-Sleep -Milliseconds $delay }
        }
    }
}

function Invoke-NasJob($job) {
    Initialize-NativeWatchdog
    if ($job.schema -ne 'morai_nas_transfer_v1') { throw 'Unknown transfer schema' }
    $script:state = [IO.Path]::GetFullPath($job.state)
    Assert-PlainPath $script:state
    $null = [IO.Directory]::CreateDirectory($script:state)
    $script:phase='starting'
    $script:currentFile=''
    $script:totalFiles=[long]$job.files
    $script:processedFiles=0L; $script:copiedFiles=0L; $script:resumedFiles=0L; $script:verifiedFiles=0L
    Set-NasProgress
    $lock = $null
    $lease = Join-Path $script:state 'parent.lease'
    # Windows-exclusive global lock protects workers even across WSL checkouts.
    $mutex = New-Object Threading.Mutex($false,'Global\MoraiDriveNasTransferV1')
    $owned = $false
    try {
        try { $owned=$mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $owned=$true }
        if (!$owned) { throw 'Another NAS worker is running on this computer' }
        $lock = [IO.File]::Open((Join-Path $script:state 'worker.lock'),
            [IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
        [MoraiNasWatchdog]::Watch($lease)
        $destination = [IO.Path]::GetFullPath($job.destination).TrimEnd('\')
        Assert-PlainPath $destination
        foreach ($source in $job.sources) {
            $s=[IO.Path]::GetFullPath($source).TrimEnd('\')
            if ($script:state.Equals($s,[StringComparison]::OrdinalIgnoreCase) -or
                $script:state.StartsWith($s+'\',[StringComparison]::OrdinalIgnoreCase)) {
                throw 'Transfer state overlaps source'
            }
            if ($destination.Equals($s,[StringComparison]::OrdinalIgnoreCase) -or
                $destination.StartsWith($s+'\',[StringComparison]::OrdinalIgnoreCase) -or
                $s.StartsWith($destination+'\',[StringComparison]::OrdinalIgnoreCase)) {
                throw 'Destination overlaps local sources'
            }
        }
        $statePrefix=$script:state.TrimEnd('\')+'\'
        if ($destination.StartsWith($statePrefix,[StringComparison]::OrdinalIgnoreCase) -or
            $script:state.StartsWith($destination+'\',[StringComparison]::OrdinalIgnoreCase) -or
            $script:state.Equals($destination,[StringComparison]::OrdinalIgnoreCase)) {
            throw 'Transfer state overlaps destination'
        }
        $script:clock=[Diagnostics.Stopwatch]::StartNew()
        $script:nextLog=0
        # A streaming buffer, not a throughput or memory-pressure policy.
        $script:buffer=New-Object byte[] 1MB
        Write-State 'starting'
        Update-Transfer
        if ((Get-FileContentHash $job.inventory) -ne $job.inventory_sha256) { throw 'Inventory changed' }
        $parent=[IO.Path]::GetDirectoryName($destination)
        Assert-PlainPath $parent
        $null=(Invoke-NasIo 'create_directory' ($parent) { [IO.Directory]::CreateDirectory($parent) })
        $pending=Join-Path $parent ('.nas-pending-'+[IO.Path]::GetFileName($destination))
        Assert-PlainPath $pending
        $alreadyFinal=(Invoke-NasIo 'directory_exists' $destination { [IO.Directory]::Exists($destination) })
        $working=if($alreadyFinal){$destination}else{$pending}
        $owner=Join-Path $working '.NAS_TRANSFER_ID.json'
        if ((Invoke-NasIo 'directory_exists' $working { [IO.Directory]::Exists($working) })) {
            if (!(Invoke-NasIo 'file_exists' $owner { [IO.File]::Exists($owner) }) -or (Read-Json $owner).identity -ne $job.identity) {
                throw 'Existing destination belongs to another snapshot; preserved without overwrite'
            }
        } else {
            $null=(Invoke-NasIo 'create_directory' ($working) { [IO.Directory]::CreateDirectory($working) })
            Write-JsonAtomic $owner @{identity=$job.identity; source_roots=$job.sources}
        }
        $receiptRoot=Join-Path $script:state 'receipts'
        $null=[IO.Directory]::CreateDirectory($receiptRoot)
        $count=0L; $bytes=0L
        Write-State 'copying'
        $reader=[IO.File]::OpenText($job.inventory)
        try {
            while ($null -ne ($line=$reader.ReadLine())) {
                $row=$line | ConvertFrom-Json
                $script:currentFile=$row.relative
                Update-Transfer
                $target=Join-Safe $working $row.relative
                if ($row.kind -eq 'directory') {
                    if ($alreadyFinal -and !(Invoke-NasIo 'directory_exists' $target { [IO.Directory]::Exists($target) })) {
                        throw 'Published directory is missing; refusing modification'
                    }
                    $null=(Invoke-NasIo 'create_directory' ($target) { [IO.Directory]::CreateDirectory($target) }); continue
                }
                Assert-Source $row
                $receipt=Join-Path $receiptRoot ((Get-RowKey $row.relative)+'.json')
                $null=(Invoke-NasIo 'create_directory' ([IO.Path]::GetDirectoryName($target)) { [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target)) })
                if ([IO.File]::Exists($receipt) -and (Invoke-NasIo 'file_exists' $target { [IO.File]::Exists($target) })) {
                    # Do not trust size/timestamp alone on restart.
                    $old=Read-Json $receipt
                    if ((Get-FileContentHash $row.source) -eq $old.sha256 -and
                        (Invoke-NasIo 'target_stat' $target { Get-Item -LiteralPath $target }).Length -eq $row.size) {
                        $script:processedFiles++; $script:resumedFiles++; Set-NasProgress; continue
                    }
                }
                if ($alreadyFinal) { throw 'Published folder has changed; refusing overwrite' }
                $temp=Join-Path $working ('.nas-temp-'+(Get-RowKey $row.relative))
                if ((Invoke-NasIo 'file_exists' $temp { [IO.File]::Exists($temp) })) { Assert-PlainPath $temp }
                $success=$false
                for ($attempt=0; $attempt -lt 3; $attempt++) {
                    try {
                        Update-Transfer
                        $hash=Copy-FileWithHash $row.source $temp
                        Assert-Source $row
                        if ((Invoke-NasIo 'file_exists' $target { [IO.File]::Exists($target) })) { Invoke-NasIo 'delete_previous_target' $target { [IO.File]::Delete($target) } }
                        Invoke-NasIo 'rename_copied_file' $target { [IO.File]::Move($temp,$target) }
                        Write-JsonAtomic $receipt @{sha256=$hash; relative=$row.relative; size=$row.size}
                        if ($script:phase -eq 'retrying') { Write-State 'copying' }
                        $script:processedFiles++; $script:copiedFiles++; Set-NasProgress
                        $success=$true; break
                    } catch [IO.IOException] {
                        if ($attempt -eq 2) { throw }
                        Write-State 'retrying' $_.Exception.Message
                        Write-Host ('[NAS] I/O retry {0}/3: {1}' -f ($attempt+2),$_.Exception.Message)
                        Update-Transfer
                    }
                }
                if (!$success) { throw 'Copy failed' }
            }
        } finally { $reader.Dispose() }
        Write-State 'verifying'
        $reader=[IO.File]::OpenText($job.inventory)
        try {
            while ($null -ne ($line=$reader.ReadLine())) {
                $row=$line | ConvertFrom-Json
                $script:currentFile=$row.relative
                Update-Transfer
                $target=Join-Safe $working $row.relative
                if ($row.kind -eq 'directory') {
                    if (!(Invoke-NasIo 'directory_exists' $target { [IO.Directory]::Exists($target) })) { throw 'Missing destination directory' }
                    continue
                }
                Assert-Source $row
                $receipt=Join-Path $receiptRoot ((Get-RowKey $row.relative)+'.json')
                $expected=Read-Json $receipt
                if ((Invoke-NasIo 'target_stat' $target { Get-Item -LiteralPath $target }).Length -ne $row.size -or
                    (Get-FileContentHash $target) -ne $expected.sha256) {
                    if (!$alreadyFinal) { [IO.File]::Delete($receipt) }
                    throw "Destination readback mismatch: $($row.relative)"
                }
                Assert-Source $row
                $count++; $bytes += $row.size
                $script:verifiedFiles++; Set-NasProgress
            }
        } finally { $reader.Dispose() }
        # Detect extra files without constructing a complete in-memory list.
        Write-State 'checking_inventory'
        $observed=0L
        Get-SafeFiles $working | ForEach-Object {
            $file=$_
            # Get-SafeFiles already checked this entry.
            if ($file -ne $owner -and $file -ne (Join-Path $working '.NAS_TRANSFER_COMPLETE.json')) { $observed++ }
        }
        if($count -ne $job.files -or $observed -ne $job.files -or $bytes -ne $job.bytes){
            throw 'Destination inventory/count/size mismatch'
        }
        Write-State 'publishing'
        if(!$alreadyFinal){Publish-NasDirectory $pending $destination}
        Write-JsonAtomic (Join-Path $destination '.NAS_TRANSFER_COMPLETE.json') @{
            schema='morai_nas_transfer_v1'; identity=$job.identity; files=$count; bytes=$bytes;
            status='copy_and_readback_verified'; verification_scope='client_readback';
            remote_durability_verified=$false; local_sources_retained=$true;
            utc=[DateTime]::UtcNow.ToString('o')
        }
        Write-State 'copy_and_readback_verified'
        Write-Host "[NAS] Copy and readback verified: $destination (local originals retained)"
    } catch {
        if($owned -and (Test-Path -LiteralPath $script:state)){
            Write-State 'failed' $_.Exception.Message
        }
        throw
    } finally {
        [MoraiNasWatchdog]::Stop()
        if($null -ne $lock){$lock.Dispose()}
        if($owned){$mutex.ReleaseMutex()}
        $mutex.Dispose()
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    try {
        if (!$JobFile) { throw 'JobFile is required' }
        Invoke-NasJob (Read-Json $JobFile)
    } catch {
        [Console]::Error.WriteLine('[NAS] '+$_.Exception.Message)
        exit 4
    }
}
