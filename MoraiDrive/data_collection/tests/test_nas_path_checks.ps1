$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../internal_sensor_record/tools/publish_sensor_record_nas.ps1')
$script:pathQueries=[Collections.Generic.List[object]]::new()
$script:injectedPath=''
$script:injectedError=''
function Invoke-NasIo([string]$operation,[string]$path,[scriptblock]$action) {
    $script:pathQueries.Add([pscustomobject]@{operation=$operation;path=$path})
    if($path -eq $script:injectedPath){
        if($script:injectedError -eq 'access'){throw [UnauthorizedAccessException]::new('Injected access failure')}
        throw [IO.IOException]::new('Injected network failure')
    }
    & $action
}
function Assert-OneLookupPerComponent([string]$path) {
    $script:pathQueries.Clear()
    Assert-PlainPath $path
    $expected=0
    $cursor=[IO.Path]::GetFullPath($path)
    while($cursor){
        $expected++
        $parent=[IO.Directory]::GetParent($cursor)
        $cursor=if($parent){$parent.FullName}else{$null}
    }
    if($script:pathQueries.Count -ne $expected){throw "Wrong lookup count for $path"}
    if(@($script:pathQueries | Where-Object {$_.operation -ne 'get_attributes'}).Count){throw 'Redundant existence query returned'}
    if(@($script:pathQueries | Group-Object path | Where-Object {$_.Count -ne 1}).Count){throw 'A path was queried twice'}
}
$fixture=Join-Path $env:TEMP ('morai-nas-path-test-'+[Guid]::NewGuid().ToString('N'))
$link=Join-Path $fixture 'junction'
try {
    $directory=Join-Path $fixture 'plain\child'
    $null=[IO.Directory]::CreateDirectory($directory)
    $file=Join-Path $directory 'data.txt'
    [IO.File]::WriteAllText($file,'fixture')
    foreach($path in @($file,$directory,(Join-Path $directory 'missing.txt'),(Join-Path $fixture 'new\child\data.txt'))){
        Assert-OneLookupPerComponent $path
    }
    $null=New-Item -ItemType Junction -Path $link -Target (Join-Path $fixture 'plain')
    foreach($target in @($link,(Join-Path $link 'child\data.txt'),(Join-Path $link 'missing.txt'))){
        $rejected=$false
        try {Assert-PlainPath $target}
        catch {if($_.Exception.Message -like '*Linked path refused*'){$rejected=$true}else{throw}}
        if(!$rejected){throw 'Junction or ancestor junction was accepted'}
    }
    $script:injectedPath=$directory
    foreach($errorKind in @('access','network')){
        $script:injectedError=$errorKind
        $rejected=$false
        try {Assert-PlainPath $file}
        catch [UnauthorizedAccessException] {if($errorKind -eq 'access'){$rejected=$true}else{throw}}
        catch [IO.IOException] {if($errorKind -eq 'network'){$rejected=$true}else{throw}}
        if(!$rejected){throw "$errorKind failure was swallowed"}
    }
    Write-Host 'PASS: one lookup per component; missing paths allowed; junctions and access/network errors rejected'
} finally {
    $script:injectedPath=''
    if([IO.Directory]::Exists($link)){[IO.Directory]::Delete($link)}
    $resolved=[IO.Path]::GetFullPath($fixture)
    $tempRoot=[IO.Path]::GetFullPath($env:TEMP).TrimEnd('\')+'\'
    if(!$resolved.StartsWith($tempRoot,[StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($resolved) -notlike 'morai-nas-path-test-*'){throw 'Unsafe test cleanup'}
    if([IO.Directory]::Exists($resolved)){Remove-Item -LiteralPath $resolved -Recurse -Force}
}
