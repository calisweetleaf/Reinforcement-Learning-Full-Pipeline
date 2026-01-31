<#
.SYNOPSIS
 Master Hash Integrity & Provenance System - Production Grade File Verification

.DESCRIPTION
 Comprehensive file integrity system with cryptographic hashing, metadata extraction,
 SBOM generation, historical tracking, multi-signature support, and experimental
 frequency-based signatures for advanced research applications.

.AUTHOR
 Christian Trey Rowell
#>

param(
    [string]$Path = ".",
    [ValidateSet('SHA256', 'SHA1', 'SHA384', 'SHA512', 'BLAKE2B')]
    [string]$Algorithm = 'SHA256',
    [string]$IndexFile = '',
    [switch]$SkipExisting,
    [switch]$UseSidecar,
    [switch]$Verify,
    [switch]$SignWithGPG,
    [switch]$TimestampProof,
    [switch]$ExtractMetadata,
    [switch]$InjectMetadata,
    [hashtable]$MetadataToInject = @{},
    [switch]$GenerateSBOM,
    [switch]$EnableHistory,
    [switch]$ShowHistory,
    [switch]$DiffMode,
    [string]$DiffWith = '',
    [switch]$GenerateReport,
    [string]$ReportFormat = 'HTML',
    [switch]$ParallelVerify,
    [switch]$DeepArchives,
    [switch]$PerceptualHash,
    [switch]$FrequencySignature,
    [switch]$EntropyAnalysis,
    [switch]$ExportJSON,
    [switch]$ExportCSV,
    [string]$ImportFrom = '',
    [string]$ManifestVersion = "2.0",
    [string]$ReleaseVersion = "",
    [string]$ReleaseTitle = "",
    [string[]]$Exclude = @('.git', '.venv', 'node_modules'),
    [switch]$UseHashIgnore,
    [switch]$UseGitIgnore, # NEW: Read from .gitignore
    [switch]$All, # NEW: Run all hash algorithms
    [int]$MinFileSize = 0,
    [int]$MaxFileSize = 0,
    [switch]$Verbose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

#region Core Configuration
$script:Config = @{
    DatabaseFile    = '.hash-index.db'
    HashIgnoreFile  = '.hashignore'
    MetadataDir     = '.metadata'
    ReportDir       = '.reports'
    SBOMDir         = '.sbom'
    LogFile         = '.hash-index.log'
    MaxParallelJobs = [Math]::Min(8, [Environment]::ProcessorCount)
}
#endregion

#region Logging System
function Write-Log {
    param(
        [Parameter(Mandatory)][AllowEmptyString()][string]$Message,
        [ValidateSet('DEBUG', 'INFO', 'WARN', 'ERROR', 'SUCCESS')]
        [string]$Level = 'INFO'
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logLine = "[$timestamp] [$Level] $Message"

    try {
        Add-Content -Path $script:Config.LogFile -Value $logLine -ErrorAction SilentlyContinue
    }
    catch {}

    if ($Verbose -or $Level -ne 'DEBUG') {
        $color = switch ($Level) {
            'DEBUG' { 'Gray' }
            'INFO' { 'Cyan' }
            'WARN' { 'Yellow' }
            'ERROR' { 'Red' }
            'SUCCESS' { 'Green' }
        }
        Write-Host "[$Level] $Message" -ForegroundColor $color
    }
}

function Write-Info([string]$msg) { Write-Log $msg 'INFO' }
function Write-Success([string]$msg) { Write-Log $msg 'SUCCESS' }
function Write-Warn([string]$msg) { Write-Log $msg 'WARN' }
function Write-Err([string]$msg) { Write-Log $msg 'ERROR' }
function Write-Debug([string]$msg) { Write-Log $msg 'DEBUG' }
#endregion

#region Database Layer
function Get-PythonExecutable {
    # Prefer project venv python if present; otherwise fall back to PATH python.
    try {
        if ($env:VIRTUAL_ENV) {
            $venvPy = Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe'
            if (Test-Path $venvPy) { return $venvPy }
        }

        if ($PSScriptRoot) {
            # Prefer sibling venv at repo root
            $repoRoot = Split-Path -Parent $PSScriptRoot
            $repoVenvPy = Join-Path $repoRoot '.venv/Scripts/python.exe'
            if (Test-Path $repoVenvPy) { return $repoVenvPy }
        }

        $cmd = Get-Command python -ErrorAction Stop
        return $cmd.Source
    }
    catch {
        return $null
    }
}

function Invoke-PythonSqlite {
    param(
        [Parameter(Mandatory)][string]$DbPath,
        [Parameter(Mandatory)][ValidateSet('executescript', 'execute_json', 'query_json')][string]$Mode,
        [string]$Sql = '',
        [hashtable]$Data = @{}
    )

    $py = Get-PythonExecutable
    if (-not $py) {
        throw "Python executable not found; cannot use sqlite backend."
    }

    $program = switch ($Mode) {
        'executescript' {
            @"
import sys, sqlite3
db = sys.argv[1]
sql = sys.stdin.read()
con = sqlite3.connect(db)
try:
 con.executescript(sql)
 con.commit()
finally:
 con.close()
"@
        }
        'execute_json' {
            @"
import sys, json, sqlite3
db = sys.argv[1]
payload = json.load(sys.stdin)
sql = payload["sql"]
params = payload.get("params", [])
con = sqlite3.connect(db)
try:
 con.execute(sql, params)
 con.commit()
finally:
 con.close()
"@
        }
        'query_json' {
            @"
import sys, json, sqlite3
db = sys.argv[1]
payload = json.load(sys.stdin)
sql = payload["sql"]
params = payload.get("params", [])
con = sqlite3.connect(db)
try:
 cur = con.execute(sql, params)
 cols = [d[0] for d in cur.description] if cur.description else []
 rows = [dict(zip(cols, r)) for r in cur.fetchall()]
 sys.stdout.write(json.dumps(rows))
finally:
 con.close()
"@
        }
    }

    # Some PS 7.6 preview builds can throw a native-process encoding error when invoking
    # executables directly; use Start-Process with redirected streams for reliability.
    $tmpProgram = New-TemporaryFile
    $tmpIn = New-TemporaryFile
    $tmpOut = New-TemporaryFile
    $tmpErr = New-TemporaryFile
    try {
        Set-Content -LiteralPath $tmpProgram.FullName -Value $program -Encoding UTF8

        if ($Mode -eq 'executescript') {
            Set-Content -LiteralPath $tmpIn.FullName -Value $Sql -Encoding UTF8
        }
        else {
            $payload = @{
                sql    = $Sql
                params = @()
            }
            if ($Data -and $Data.ContainsKey('params')) {
                $payload.params = @($Data.params)
            }
            $json = $payload | ConvertTo-Json -Compress -Depth 6
            Set-Content -LiteralPath $tmpIn.FullName -Value $json -Encoding UTF8
        }

        $proc = Start-Process -FilePath $py -ArgumentList @($tmpProgram.FullName, $DbPath) `
            -RedirectStandardInput $tmpIn.FullName `
            -RedirectStandardOutput $tmpOut.FullName `
            -RedirectStandardError $tmpErr.FullName `
            -NoNewWindow -Wait -PassThru -ErrorAction Stop

        if ($proc.ExitCode -ne 0) {
            $stderr = (Get-Content -LiteralPath $tmpErr.FullName -Raw -ErrorAction SilentlyContinue)
            throw "python sqlite helper failed (exit $($proc.ExitCode)): $stderr"
        }

        if ($Mode -eq 'query_json') {
            return (Get-Content -LiteralPath $tmpOut.FullName -Raw -ErrorAction SilentlyContinue)
        }
        return $null
    }
    finally {
        foreach ($t in @($tmpProgram, $tmpIn, $tmpOut, $tmpErr)) {
            try { if ($t) { Remove-Item -LiteralPath $t.FullName -Force -ErrorAction SilentlyContinue } } catch {}
        }
    }
}

function Initialize-Database {
    param([string]$dbPath)

    Write-Debug "Initializing database at $dbPath"

    try {
        $py = Get-PythonExecutable
        if (-not $py) { throw "Python not found" }
    }
    catch {
        Write-Warn "Python not found; SQLite history features disabled."
        return $false
    }

    try {
        $schema = @"
CREATE TABLE IF NOT EXISTS file_history (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 file_path TEXT NOT NULL,
 hash_value TEXT NOT NULL,
 algorithm TEXT NOT NULL,
 file_size INTEGER,
 modified_time TEXT,
 scan_time TEXT NOT NULL,
 git_commit TEXT,
 git_tag TEXT,
 metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_file_path ON file_history(file_path);
CREATE INDEX IF NOT EXISTS idx_scan_time ON file_history(scan_time);
CREATE INDEX IF NOT EXISTS idx_hash ON file_history(hash_value);

CREATE TABLE IF NOT EXISTS scan_sessions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 scan_time TEXT NOT NULL,
 root_path TEXT NOT NULL,
 algorithm TEXT NOT NULL,
 file_count INTEGER,
 total_size INTEGER,
 duration_seconds REAL,
 git_commit TEXT,
 git_tag TEXT,
 release_version TEXT
);

CREATE TABLE IF NOT EXISTS metadata_store (
 file_path TEXT PRIMARY KEY,
 content_type TEXT,
 encoding TEXT,
 dimensions TEXT,
 duration REAL,
 author TEXT,
 title TEXT,
 subject TEXT,
 keywords TEXT,
 created_date TEXT,
 modified_date TEXT,
 extra_metadata TEXT
);

CREATE TABLE IF NOT EXISTS frequency_signatures (
 file_path TEXT PRIMARY KEY,
 dct_coefficients TEXT,
 fft_peaks TEXT,
 spectral_centroid REAL,
 spectral_rolloff REAL,
 zero_crossing_rate REAL,
 entropy REAL,
 signature_time TEXT
);
"@

        Invoke-PythonSqlite -DbPath $dbPath -Mode executescript -Sql $schema
        Write-Success "Database initialized successfully"
        return $true
    }
    catch {
        Write-Err "Database initialization failed: $_"
        return $false
    }
}

function Add-HistoryRecord {
    param(
        [string]$dbPath,
        [string]$filePath,
        [string]$hash,
        [string]$algorithm,
        [long]$fileSize,
        [datetime]$modifiedTime,
        [string]$gitCommit,
        [string]$gitTag,
        [hashtable]$metadata
    )

    try {
        $scanTime = (Get-Date).ToUniversalTime().ToString("o")
        $modTime = $modifiedTime.ToUniversalTime().ToString("o")
        $metaJson = if ($metadata) { ($metadata | ConvertTo-Json -Compress -Depth 8) } else { '' }

        $sql = @"
INSERT INTO file_history (file_path, hash_value, algorithm, file_size, modified_time, scan_time, git_commit, git_tag, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"@

        $params = @(
            $filePath,
            $hash,
            $algorithm,
            [long]$fileSize,
            $modTime,
            $scanTime,
            $gitCommit,
            $gitTag,
            $metaJson
        )

        Invoke-PythonSqlite -DbPath $dbPath -Mode execute_json -Sql $sql -Data @{ params = $params } | Out-Null
        Write-Debug "History record added for $filePath"
    }
    catch {
        Write-Warn "Failed to add history record for $filePath : $_"
    }
}

function Get-FileHistory {
    param(
        [string]$dbPath,
        [string]$filePath
    )

    try {
        $sql = "SELECT scan_time AS ScanTime, hash_value AS Hash, algorithm AS Algorithm, file_size AS Size FROM file_history WHERE file_path = ? ORDER BY scan_time DESC;"
        $out = Invoke-PythonSqlite -DbPath $dbPath -Mode query_json -Sql $sql -Data @{ params = @($filePath) }
        if (-not $out) { return @() }
        return ($out | ConvertFrom-Json)
    }
    catch {
        Write-Warn "Failed to retrieve history for $filePath : $_"
        return @()
    }
}

function Add-ScanSession {
    param(
        [string]$dbPath,
        [string]$rootPath,
        [string]$algorithm,
        [int]$fileCount,
        [long]$totalSize,
        [double]$duration,
        [string]$gitCommit,
        [string]$gitTag,
        [string]$releaseVersion
    )

    try {
        $scanTime = (Get-Date).ToUniversalTime().ToString("o")
        $sql = @"
INSERT INTO scan_sessions (scan_time, root_path, algorithm, file_count, total_size, duration_seconds, git_commit, git_tag, release_version)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"@

        $params = @(
            $scanTime,
            $rootPath,
            $algorithm,
            [int]$fileCount,
            [long]$totalSize,
            [double]$duration,
            $gitCommit,
            $gitTag,
            $releaseVersion
        )
        Invoke-PythonSqlite -DbPath $dbPath -Mode execute_json -Sql $sql -Data @{ params = $params } | Out-Null
        Write-Debug "Scan session recorded"
    }
    catch {
        Write-Warn "Failed to record scan session: $_"
    }
}
#endregion

#region Metadata Extraction
function Get-FileMetadata {
    param([string]$filePath)

    $meta = @{
        Path        = $filePath
        Size        = 0
        Extension   = ''
        ContentType = ''
        Encoding    = ''
        Created     = ''
        Modified    = ''
    }

    try {
        $file = Get-Item -LiteralPath $filePath -ErrorAction Stop
        $meta.Size = $file.Length
        $meta.Extension = $file.Extension.ToLower()
        $meta.Created = $file.CreationTime.ToString("o")
        $meta.Modified = $file.LastWriteTime.ToString("o")

        # Detect content type via magic bytes
        $magic = Get-MagicBytes $filePath
        $meta.ContentType = Get-ContentType $magic $meta.Extension

        # Extension-specific metadata
        switch ($meta.Extension) {
            { $_ -in '.pdf' } { $meta += Get-PDFMetadata $filePath }
            { $_ -in '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff' } { $meta += Get-ImageMetadata $filePath }
            { $_ -in '.mp3', '.wav', '.flac', '.m4a' } { $meta += Get-AudioMetadata $filePath }
            { $_ -in '.mp4', '.avi', '.mkv', '.mov' } { $meta += Get-VideoMetadata $filePath }
            { $_ -in '.docx', '.xlsx', '.pptx' } { $meta += Get-OfficeMetadata $filePath }
            { $_ -in '.exe', '.dll' } { $meta += Get-PEMetadata $filePath }
            { $_ -in '.zip', '.tar', '.gz', '.7z', '.rar' } { $meta += Get-ArchiveMetadata $filePath }
        }

        return $meta
    }
    catch {
        Write-Warn "Failed to extract metadata from $filePath : $_"
        return $meta
    }
}

function Get-MagicBytes {
    param([string]$path)

    try {
        $bytes = [byte[]](Get-Content -LiteralPath $path -Encoding Byte -TotalCount 16 -ErrorAction Stop)
        return $bytes
    }
    catch {
        return @()
    }
}

function Get-ContentType {
    param([byte[]]$magic, [string]$ext)

    $len = 0
    if ($null -eq $magic) { $len = 0 }
    elseif ($magic -is [System.Array]) { $len = $magic.Length }
    else { $len = 1 }
    if ($len -lt 4) { return 'application/octet-stream' }

    # Common magic bytes
    if ($magic[0] -eq 0x25 -and $magic[1] -eq 0x50 -and $magic[2] -eq 0x44 -and $magic[3] -eq 0x46) { return 'application/pdf' }
    if ($magic[0] -eq 0xFF -and $magic[1] -eq 0xD8 -and $magic[2] -eq 0xFF) { return 'image/jpeg' }
    if ($magic[0] -eq 0x89 -and $magic[1] -eq 0x50 -and $magic[2] -eq 0x4E -and $magic[3] -eq 0x47) { return 'image/png' }
    if ($magic[0] -eq 0x47 -and $magic[1] -eq 0x49 -and $magic[2] -eq 0x46) { return 'image/gif' }
    if ($magic[0] -eq 0x50 -and $magic[1] -eq 0x4B) { return 'application/zip' }
    if ($magic[0] -eq 0x4D -and $magic[1] -eq 0x5A) { return 'application/x-msdownload' }
    if ($magic[0] -eq 0x1F -and $magic[1] -eq 0x8B) { return 'application/gzip' }
    if ($magic[0] -eq 0xFD -and $magic[1] -eq 0x37 -and $magic[2] -eq 0x7A -and $magic[3] -eq 0x58) { return 'application/x-xz' }

    # Fallback to extension
    $mimeTypes = @{
        '.txt'  = 'text/plain'
        '.json' = 'application/json'
        '.xml'  = 'application/xml'
        '.html' = 'text/html'
        '.css'  = 'text/css'
        '.js'   = 'application/javascript'
        '.py'   = 'text/x-python'
        '.ps1'  = 'text/plain'
        '.sh'   = 'text/x-shellscript'
    }

    return $mimeTypes[$ext] ?? 'application/octet-stream'
}

function Get-PDFMetadata {
    param([string]$path)

    $meta = @{}
    try {
        # Read first 4KB to find PDF metadata
        $content = Get-Content -LiteralPath $path -Encoding Byte -TotalCount 4096
        $text = [System.Text.Encoding]::ASCII.GetString($content)

        if ($text -match '/Title\s*\(([^)]+)\)') { $meta.Title = $matches[1] }
        if ($text -match '/Author\s*\(([^)]+)\)') { $meta.Author = $matches[1] }
        if ($text -match '/Subject\s*\(([^)]+)\)') { $meta.Subject = $matches[1] }
        if ($text -match '/Keywords\s*\(([^)]+)\)') { $meta.Keywords = $matches[1] }
    }
    catch {}

    return $meta
}

function Get-ImageMetadata {
    param([string]$path)

    $meta = @{}
    try {
        Add-Type -AssemblyName System.Drawing -ErrorAction SilentlyContinue
        $img = [System.Drawing.Image]::FromFile($path)
        $meta.Width = $img.Width
        $meta.Height = $img.Height
        $meta.Dimensions = "$($img.Width)x$($img.Height)"
        $meta.PixelFormat = $img.PixelFormat.ToString()
        $meta.HorizontalResolution = $img.HorizontalResolution
        $meta.VerticalResolution = $img.VerticalResolution
        $img.Dispose()
    }
    catch {
        Write-Debug "Could not load image metadata: $_"
    }

    return $meta
}

function Get-AudioMetadata {
    param([string]$path)

    $meta = @{}
    # Would use TagLib# or similar in production
    # For now, basic file info
    $meta.MediaType = 'audio'
    return $meta
}

function Get-VideoMetadata {
    param([string]$path)

    $meta = @{}
    $meta.MediaType = 'video'
    return $meta
}

function Get-OfficeMetadata {
    param([string]$path)

    $meta = @{}
    try {
        # Office files are ZIP archives
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip = [System.IO.Compression.ZipFile]::OpenRead($path)
        $coreProps = $zip.Entries | Where-Object { $_.FullName -eq 'docProps/core.xml' }

        if ($coreProps) {
            $stream = $coreProps.Open()
            $reader = [System.IO.StreamReader]::new($stream)
            $xml = $reader.ReadToEnd()
            $reader.Close()
            $stream.Close()

            if ($xml -match '<dc:title>([^<]+)</dc:title>') { $meta.Title = $matches[1] }
            if ($xml -match '<dc:creator>([^<]+)</dc:creator>') { $meta.Author = $matches[1] }
            if ($xml -match '<dc:subject>([^<]+)</dc:subject>') { $meta.Subject = $matches[1] }
        }

        $zip.Dispose()
    }
    catch {
        Write-Debug "Could not extract Office metadata: $_"
    }

    return $meta
}

function Get-PEMetadata {
    param([string]$path)

    $meta = @{}
    try {
        $version = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($path)
        $meta.ProductName = $version.ProductName
        $meta.ProductVersion = $version.ProductVersion
        $meta.FileVersion = $version.FileVersion
        $meta.CompanyName = $version.CompanyName
        $meta.Copyright = $version.LegalCopyright
        $meta.Description = $version.FileDescription
    }
    catch {}

    return $meta
}

function Get-ArchiveMetadata {
    param([string]$path)

    $meta = @{}
    try {
        if ($path -match '\.(zip|jar|war|ear)$') {
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            $zip = [System.IO.Compression.ZipFile]::OpenRead($path)
            $meta.EntryCount = $zip.Entries.Count
            $meta.UncompressedSize = ($zip.Entries | Measure-Object -Property Length -Sum).Sum
            $zip.Dispose()
        }
    }
    catch {}

    return $meta
}

function Export-MetadataToFile {
    param(
        [string]$filePath,
        [hashtable]$metadata,
        [string]$metadataDir
    )

    try {
        if (-not (Test-Path $metadataDir)) {
            New-Item -ItemType Directory -Path $metadataDir -Force | Out-Null
        }

        $baseName = [System.IO.Path]::GetFileName($filePath)
        $metaFile = Join-Path $metadataDir "$baseName.metadata.json"

        $metadata | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $metaFile -Encoding UTF8
        Write-Debug "Metadata exported to $metaFile"
    }
    catch {
        Write-Warn "Failed to export metadata for $filePath : $_"
    }
}
#endregion

#region Frequency-Based Signatures (Experimental)
function Get-FrequencySignature {
    param([string]$filePath)

    Write-Debug "Computing frequency signature for $filePath"

    try {
        $bytes = [System.IO.File]::ReadAllBytes($filePath)

        # Limit to first 64KB for performance
        $sampleSize = [Math]::Min($bytes.Length, 65536)

        # Pre-allocate array for signal (much faster than += concatenation)
        [double[]]$signal = [double[]]::new($sampleSize)
        for ($i = 0; $i -lt $sampleSize; $i++) {
            $signal[$i] = [double]$bytes[$i] / 255.0
        }

        # Compute entropy (information density)
        $entropy = Get-Entropy $signal

        # Subsample for DCT to ~512 points (fast and numerically stable)
        $dctTargetSize = 512
        $dctSubsampleRate = [Math]::Max(1, [Math]::Floor($signal.Length / $dctTargetSize))
        $dctSampleCount = [Math]::Ceiling($signal.Length / $dctSubsampleRate)
        [double[]]$dctSignal = [double[]]::new($dctSampleCount)
        $dctIndex = 0
        for ($i = 0; $i -lt $signal.Length -and $dctIndex -lt $dctSampleCount; $i += $dctSubsampleRate) {
            $dctSignal[$dctIndex++] = $signal[$i]
        }

        # Compute spectral features (simplified DCT) on subsampled signal
        $dctCoeffs = Get-SimplifiedDCT $dctSignal 32

        # Zero-crossing rate (signal oscillation measure)
        $zcr = Get-ZeroCrossingRate $signal

        # Spectral centroid (frequency "center of mass")
        $centroid = Get-SpectralCentroid $signal

        $signature = @{
            Entropy          = [Math]::Round($entropy, 4)
            ZeroCrossingRate = [Math]::Round($zcr, 4)
            SpectralCentroid = [Math]::Round($centroid, 4)
            DCTCoefficients  = ($dctCoeffs | ForEach-Object { [Math]::Round($_, 4) }) -join ','
            SignatureTime    = (Get-Date).ToUniversalTime().ToString("o")
        }

        return $signature
    }
    catch {
        Write-Warn "Failed to compute frequency signature for $filePath : $_"
        return $null
    }
}

function Get-Entropy {
    param([double[]]$signal)

    # Quantize to bins
    $bins = @{}
    foreach ($val in $signal) {
        $bin = [Math]::Floor($val * 255)
        if ($bins.ContainsKey($bin)) {
            $bins[$bin]++
        }
        else {
            $bins[$bin] = 1
        }
    }

    # Shannon entropy
    $entropy = 0.0
    $total = $signal.Length
    foreach ($count in $bins.Values) {
        $p = $count / $total
        $entropy -= $p * [Math]::Log($p, 2)
    }

    return $entropy
}

function Get-ZeroCrossingRate {
    param([double[]]$signal)

    $crossings = 0
    $mean = ($signal | Measure-Object -Average).Average

    for ($i = 1; $i -lt $signal.Length; $i++) {
        if (($signal[$i] -ge $mean -and $signal[$i - 1] -lt $mean) -or
            ($signal[$i] -lt $mean -and $signal[$i - 1] -ge $mean)) {
            $crossings++
        }
    }

    return $crossings / $signal.Length
}

function Get-SpectralCentroid {
    param([double[]]$signal)

    $weighted = 0.0
    $total = 0.0

    for ($i = 0; $i -lt $signal.Length; $i++) {
        $weighted += $i * [Math]::Abs($signal[$i])
        $total += [Math]::Abs($signal[$i])
    }

    if ($total -eq 0) { return 0 }
    return $weighted / $total / $signal.Length
}

function Get-SimplifiedDCT {
    param([double[]]$signal, [int]$numCoeffs)

    $N = $signal.Length
    $coeffs = @()

    for ($k = 0; $k -lt $numCoeffs; $k++) {
        $sum = 0.0
        for ($n = 0; $n -lt $N; $n++) {
            $sum += $signal[$n] * [Math]::Cos([Math]::PI * $k * (2 * $n + 1) / (2 * $N))
        }
        $coeffs += $sum / $N
    }

    return $coeffs
}

function Get-PerceptualHash {
    param([string]$filePath)

    # Simplified perceptual hash (for non-image files, based on content structure)
    try {
        $bytes = [System.IO.File]::ReadAllBytes($filePath)
        $sampleSize = [Math]::Min($bytes.Length, 1024)
        $sample = $bytes[0..($sampleSize - 1)]

        # Compute 8x8 "image" from byte distribution
        $grid = New-Object 'double[,]' 8, 8
        for ($i = 0; $i -lt $sample.Length; $i++) {
            $x = ($i % 8)
            $y = [Math]::Floor($i / 8) % 8
            $grid[$x, $y] += $sample[$i]
        }

        # Average
        $avg = 0
        for ($x = 0; $x -lt 8; $x++) {
            for ($y = 0; $y -lt 8; $y++) {
                $avg += $grid[$x, $y]
            }
        }
        $avg /= 64

        # Build hash
        $hash = ''
        for ($x = 0; $x -lt 8; $x++) {
            for ($y = 0; $y -lt 8; $y++) {
                $hash += if ($grid[$x, $y] -gt $avg) { '1' } else { '0' }
            }
        }

        # Convert to hex
        $hexHash = ''
        for ($i = 0; $i -lt $hash.Length; $i += 4) {
            $nibble = [Convert]::ToInt32($hash.Substring($i, 4), 2)
            $hexHash += $nibble.ToString('X')
        }

        return $hexHash
    }
    catch {
        return $null
    }
}
#endregion

#region SBOM Generation
function New-SBOM {
    param(
        [string]$rootPath,
        [array]$files,
        [string]$format = 'SPDX'
    )

    Write-Info "Generating SBOM in $format format..."

    $sbomData = @{
        SPDXVersion       = "SPDX-2.3"
        DataLicense       = "CC0-1.0"
        SPDXID            = "SPDXRef-DOCUMENT"
        DocumentName      = "Hash-Index-SBOM"
        DocumentNamespace = "https://example.com/sbom/$(New-Guid)"
        CreationInfo      = @{
            Created            = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
            Creators           = @("Tool: hash-index.ps1")
            LicenseListVersion = "3.21"
        }
        Packages          = @()
        Relationships     = @()
    }

    $packageIndex = 1
    foreach ($file in $files) {
        $package = @{
            SPDXID           = "SPDXRef-Package-$packageIndex"
            Name             = [System.IO.Path]::GetFileName($file.FullPath ?? $file.RelativePath)
            VersionInfo      = "1.0"
            Supplier         = "Organization: Unknown"
            DownloadLocation = "NOASSERTION"
            FilesAnalyzed    = $true
            VerificationCode = @{
                Value     = $file.Hash
                Algorithm = $file.Algorithm
            }
            Checksums        = @(
                @{
                    Algorithm = $file.Algorithm
                    Value     = $file.Hash
                }
            )
            LicenseConcluded = "NOASSERTION"
            LicenseDeclared  = "NOASSERTION"
            CopyrightText    = "NOASSERTION"
            FilePath         = $file.RelativePath
        }

        $sbomData.Packages += $package
        $sbomData.Relationships += @{
            spdxElementId      = "SPDXRef-DOCUMENT"
            relationshipType   = "DESCRIBES"
            relatedSpdxElement = "SPDXRef-Package-$packageIndex"
        }

        $packageIndex++
    }

    return $sbomData
}

function Export-SBOM {
    param(
        [hashtable]$sbomData,
        [string]$outputPath,
        [string]$format = 'JSON'
    )

    try {
        if (-not (Test-Path (Split-Path $outputPath))) {
            New-Item -ItemType Directory -Path (Split-Path $outputPath) -Force | Out-Null
        }

        if ($format -eq 'JSON') {
            $sbomData | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $outputPath -Encoding UTF8
        }
        elseif ($format -eq 'YAML') {
            # Simple YAML conversion
            $yaml = ConvertTo-SimpleYAML $sbomData
            Set-Content -LiteralPath $outputPath -Value $yaml -Encoding UTF8
        }

        Write-Success "SBOM exported to $outputPath"
    }
    catch {
        Write-Err "Failed to export SBOM: $_"
    }
}

function ConvertTo-SimpleYAML {
    param([hashtable]$data, [int]$indent = 0)

    $yaml = ""
    $spaces = " " * $indent

    foreach ($key in $data.Keys) {
        $value = $data[$key]

        if ($value -is [hashtable]) {
            $yaml += "$spaces$key`:`n"
            $yaml += ConvertTo-SimpleYAML $value ($indent + 2)
        }
        elseif ($value -is [array]) {
            $yaml += "$spaces$key`:`n"
            foreach ($item in $value) {
                if ($item -is [hashtable]) {
                    $yaml += "$spaces- `n"
                    $yaml += ConvertTo-SimpleYAML $item ($indent + 4)
                }
                else {
                    $yaml += "$spaces- $item`n"
                }
            }
        }
        else {
            $yaml += "$spaces$key`: $value`n"
        }
    }

    return $yaml
}
#endregion

#region Hash Ignore System
function Get-HashIgnorePatterns {
    param([string]$rootPath)

    $ignoreFile = Join-Path $rootPath $script:Config.HashIgnoreFile
    $patterns = @()

    if (Test-Path $ignoreFile) {
        Write-Debug "Loading .hashignore patterns"
        $lines = Get-Content $ignoreFile

        foreach ($line in $lines) {
            $line = $line.Trim()
            if ($line -and -not $line.StartsWith('#')) {
                $patterns += $line
            }
        }

        Write-Info "Loaded $($patterns.Count) patterns from .hashignore"
    }

    return $patterns
}

function Get-GitIgnorePatterns {
    param([string]$rootPath)

    $ignoreFile = Join-Path $rootPath '.gitignore'
    $patterns = @()

    if (Test-Path $ignoreFile) {
        Write-Debug "Loading .gitignore patterns"
        $lines = Get-Content $ignoreFile

        foreach ($line in $lines) {
            $line = $line.Trim()
            if ($line -and -not $line.StartsWith('#')) {
                $patterns += $line
            }
        }

        Write-Info "Loaded $($patterns.Count) patterns from .gitignore"
    }

    return $patterns
}

function Test-ShouldIgnore {
    param(
        [string]$path,
        [string[]]$patterns
    )

    foreach ($pattern in $patterns) {
        # Simple wildcard matching
        if ($path -like "*$pattern*") {
            return $true
        }

        # Directory matching
        if ($pattern.EndsWith('/') -and $path -like "*$($pattern.TrimEnd('/'))*") {
            return $true
        }
    }

    return $false
}
#endregion

#region Report Generation
function New-HTMLReport {
    param(
        [array]$files,
        [string]$outputPath,
        [hashtable]$stats
    )

    Write-Info "Generating HTML report..."

    $html = @"
<!DOCTYPE html>
<html lang="en">
<head>
 <meta charset="UTF-8">
 <meta name="viewport" content="width=device-width, initial-scale=1.0">
 <title>Hash Index Report</title>
 <style>
 * { margin: 0; padding: 0; box-sizing: border-box; }
 body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0e27; color: #e0e0e0; padding: 20px; }
 .container { max-width: 1400px; margin: 0 auto; }
 h1 { font-size: 2.5em; margin-bottom: 10px; color: #00d4ff; text-shadow: 0 0 20px rgba(0,212,255,0.5); }
 .subtitle { color: #888; margin-bottom: 30px; }
 .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
 .stat-card { background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 100%); padding: 20px; border-radius: 10px; border: 1px solid #00d4ff33; }
 .stat-label { color: #888; font-size: 0.9em; margin-bottom: 5px; }
 .stat-value { font-size: 2em; color: #00d4ff; font-weight: bold; }
 .files-table { width: 100%; border-collapse: collapse; background: #1a1f3a; border-radius: 10px; overflow: hidden; }
 .files-table th { background: #2d3561; padding: 15px; text-align: left; color: #00d4ff; border-bottom: 2px solid #00d4ff; }
 .files-table td { padding: 12px 15px; border-bottom: 1px solid #2d3561; }
 .files-table tr:hover { background: #252b4a; }
 .hash { font-family: 'Courier New', monospace; font-size: 0.85em; color: #00ff88; word-break: break-all; }
 .size { color: #ffa500; }
 .footer { margin-top: 40px; text-align: center; color: #666; font-size: 0.9em; }
 </style>
</head>
<body>
 <div class="container">
 <h1>🔐 Hash Index Report</h1>
 <p class="subtitle">Generated on $(Get-Date -Format "yyyy-MM-dd HH:mm:ss UTC")</p>

 <div class="stats">
 <div class="stat-card">
 <div class="stat-label">Total Files</div>
 <div class="stat-value">$($stats.FileCount)</div>
 </div>
 <div class="stat-card">
 <div class="stat-label">Total Size</div>
 <div class="stat-value">$(Format-FileSize $stats.TotalSize)</div>
 </div>
 <div class="stat-card">
 <div class="stat-label">Algorithm</div>
 <div class="stat-value">$($stats.Algorithm)</div>
 </div>
 <div class="stat-card">
 <div class="stat-label">Scan Duration</div>
 <div class="stat-value">$([Math]::Round($stats.Duration, 2))s</div>
 </div>
 </div>

 <table class="files-table">
 <thead>
 <tr>
 <th>File Path</th>
 <th>Hash</th>
 <th>Size</th>
 <th>Modified</th>
 </tr>
 </thead>
 <tbody>
"@

    foreach ($file in $files) {
        $html += @"
 <tr>
 <td>$($file.RelativePath)</td>
 <td class="hash">$($file.Hash)</td>
 <td class="size">$(Format-FileSize $file.Size)</td>
 <td>$($file.Modified)</td>
 </tr>
"@
    }

    $html += @"
 </tbody>
 </table>

 <div class="footer">
 <p>Generated by hash-index.ps1 - Master Hash Integrity System</p>
 </div>
 </div>
</body>
</html>
"@

    Set-Content -LiteralPath $outputPath -Value $html -Encoding UTF8
    Write-Success "HTML report saved to $outputPath"
}

function Format-FileSize {
    param([long]$bytes)

    if ($bytes -lt 1KB) { return "$bytes B" }
    if ($bytes -lt 1MB) { return "$([Math]::Round($bytes/1KB, 2)) KB" }
    if ($bytes -lt 1GB) { return "$([Math]::Round($bytes/1MB, 2)) MB" }
    return "$([Math]::Round($bytes/1GB, 2)) GB"
}
#endregion

#region Core Hashing Functions
function Get-ExpectedDigestLength([string]$alg) {
    switch ($alg.ToUpper()) {
        'SHA1' { 40 }
        'SHA256' { 64 }
        'SHA384' { 96 }
        'SHA512' { 128 }
        'BLAKE2B' { 128 }
        default { throw "Unsupported algorithm: $alg" }
    }
}

function Get-Blake2bHash {
    <#
 .SYNOPSIS
 Compute BLAKE2b hash using Python hashlib.
 .DESCRIPTION
 PowerShell's built-in Get-FileHash does not support BLAKE2b.
 This function uses Python's hashlib.blake2b to compute the hash.
 .PARAMETER FilePath
 Path to the file to hash.
 .PARAMETER DigestSize
 Digest size in bytes (default: 64 = 512-bit = 128 hex chars).
 .OUTPUTS
 Lowercase hex string of the BLAKE2b hash, or $null on failure.
 #>
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [int]$DigestSize = 64
    )

    $py = Get-PythonExecutable
    if (-not $py) {
        Write-Err "Python not found. BLAKE2b hashing requires Python with hashlib."
        return $null
    }

    # Validate file exists
    if (-not (Test-Path -LiteralPath $FilePath)) {
        Write-Err "File not found: $FilePath"
        return $null
    }

    $program = @"
import sys
import hashlib

try:
 file_path = sys.argv[1]
 digest_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64

 h = hashlib.blake2b(digest_size=digest_size)
 with open(file_path, 'rb') as f:
 for chunk in iter(lambda: f.read(65536), b''):
 h.update(chunk)

 print(h.hexdigest())
except Exception as e:
 sys.stderr.write(str(e))
 sys.exit(1)
"@

    $tmpProgram = $null
    $tmpOut = $null
    $tmpErr = $null

    try {
        $tmpProgram = New-TemporaryFile
        $tmpOut = New-TemporaryFile
        $tmpErr = New-TemporaryFile

        Set-Content -LiteralPath $tmpProgram.FullName -Value $program -Encoding UTF8

        $proc = Start-Process -FilePath $py -ArgumentList @($tmpProgram.FullName, $FilePath, $DigestSize.ToString()) `
            -RedirectStandardOutput $tmpOut.FullName `
            -RedirectStandardError $tmpErr.FullName `
            -NoNewWindow -Wait -PassThru -ErrorAction Stop

        if ($proc.ExitCode -ne 0) {
            $stderr = (Get-Content -LiteralPath $tmpErr.FullName -Raw -ErrorAction SilentlyContinue)
            Write-Err "BLAKE2b hashing failed for $FilePath : $stderr"
            return $null
        }

        $hashValue = (Get-Content -LiteralPath $tmpOut.FullName -Raw -ErrorAction SilentlyContinue).Trim()

        if ($hashValue -and $hashValue.Length -eq ($DigestSize * 2)) {
            return $hashValue.ToLower()
        }
        else {
            Write-Err "Invalid BLAKE2b hash length for $FilePath (expected $($DigestSize * 2), got $($hashValue.Length))"
            return $null
        }
    }
    catch {
        Write-Err "BLAKE2b hashing exception for $FilePath : $_"
        return $null
    }
    finally {
        foreach ($t in @($tmpProgram, $tmpOut, $tmpErr)) {
            try { if ($t) { Remove-Item -LiteralPath $t.FullName -Force -ErrorAction SilentlyContinue } } catch {}
        }
    }
}

function Get-UnifiedFileHash {
    <#
 .SYNOPSIS
 Unified file hashing that supports all declared algorithms including BLAKE2b.
 .DESCRIPTION
 Routes native algorithms (SHA1/SHA256/SHA384/SHA512) to Get-FileHash
 and BLAKE2b to the Python-based Get-Blake2bHash function.
 .PARAMETER FilePath
 Path to the file to hash.
 .PARAMETER Algorithm
 Hash algorithm: SHA1, SHA256, SHA384, SHA512, or BLAKE2B.
 .OUTPUTS
 Lowercase hex string of the hash.
 .EXAMPLE
 Get-UnifiedFileHash -FilePath "myfile.txt" -Algorithm "BLAKE2B"
 #>
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][ValidateSet('SHA1', 'SHA256', 'SHA384', 'SHA512', 'BLAKE2B')]
        [string]$Algorithm
    )

    if ($Algorithm.ToUpper() -eq 'BLAKE2B') {
        $hash = Get-Blake2bHash -FilePath $FilePath
        if (-not $hash) {
            throw "BLAKE2b hashing failed for $FilePath"
        }
        return $hash
    }
    else {
        # Use native PowerShell Get-FileHash for standard algorithms
        $result = Get-FileHash -Algorithm $Algorithm -LiteralPath $FilePath -ErrorAction Stop
        return $result.Hash.ToLower()
    }
}

function Get-SidecarHash([string]$filePath, [string]$alg) {
    $algLower = $alg.ToLower()
    $candidates = @("$filePath.$algLower")

    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) {
            $content = Get-Content -LiteralPath $c -Raw
            $m = [regex]::Match($content, '([A-Fa-f0-9]{40,128})')
            if ($m.Success) { return $m.Groups[1].Value.ToLower() }
        }
    }
    return $null
}

function Get-FileHashParallel {
    param(
        [array]$files,
        [string]$algorithm
    )

    Write-Info "Computing hashes in parallel..."

    # BLAKE2B must use sequential mode because parallel jobs can't access the Python helper
    if ($algorithm.ToUpper() -eq 'BLAKE2B') {
        Write-Info "BLAKE2B detected - using sequential mode (Python-based hashing)"
        $seq = @()
        $processed = 0
        $total = $files.Count
        foreach ($f in $files) {
            try {
                $h = Get-UnifiedFileHash -FilePath $f.FullName -Algorithm $algorithm
                $seq += @{ Path = $f.FullName; Hash = $h; Success = $true }
            }
            catch {
                $seq += @{ Path = $f.FullName; Error = $_.Exception.Message; Success = $false }
            }
            $processed++
            if ($processed % 10 -eq 0) {
                Write-Progress -Activity "Hashing files (BLAKE2B)" -Status "$processed / $total" -PercentComplete (($processed / $total) * 100)
            }
        }
        Write-Progress -Activity "Hashing files (BLAKE2B)" -Completed
        return $seq
    }

    # Some environments (or policies) block background jobs. Probe once and fall back to
    # sequential hashing if jobs cannot be started or received without errors.
    $jobsSupported = $true
    try {
        $probe = Start-Job -ScriptBlock { 1 }
        Wait-Job $probe | Out-Null
        $null = Receive-Job $probe -ErrorAction Stop
        Remove-Job $probe -Force -ErrorAction SilentlyContinue
    }
    catch {
        $jobsSupported = $false
        try { if ($probe) { Remove-Job $probe -Force -ErrorAction SilentlyContinue } } catch {}
    }

    if (-not $jobsSupported) {
        Write-Warn "Parallel jobs not available. Falling back to sequential hashing."
        $seq = @()
        $processed = 0
        $total = $files.Count
        foreach ($f in $files) {
            try {
                $h = Get-UnifiedFileHash -FilePath $f.FullName -Algorithm $algorithm
                $seq += @{ Path = $f.FullName; Hash = $h; Success = $true }
            }
            catch {
                $seq += @{ Path = $f.FullName; Error = $_.Exception.Message; Success = $false }
            }

            $processed++
            if ($processed % 10 -eq 0) {
                Write-Progress -Activity "Hashing files" -Status "$processed / $total" -PercentComplete (($processed / $total) * 100)
            }
        }
        Write-Progress -Activity "Hashing files" -Completed
        return $seq
    }

    $jobs = @()
    $results = [System.Collections.Concurrent.ConcurrentBag[object]]::new()

    $scriptBlock = {
        param($file, $alg)

        try {
            $hash = (Get-FileHash -Algorithm $alg -LiteralPath $file.FullName).Hash.ToLower()
            return @{
                Path    = $file.FullName
                Hash    = $hash
                Success = $true
            }
        }
        catch {
            return @{
                Path    = $file.FullName
                Error   = $_.Exception.Message
                Success = $false
            }
        }
    }

    $maxJobs = $script:Config.MaxParallelJobs
    $processed = 0
    $total = $files.Count

    $canUseJobs = $true
    foreach ($file in $files) {
        if (-not $canUseJobs) { break }

        while (@(Get-Job -State Running -ErrorAction SilentlyContinue).Count -ge $maxJobs) {
            Start-Sleep -Milliseconds 100

            $completed = @(Get-Job -State Completed -ErrorAction SilentlyContinue)
            foreach ($job in $completed) {
                $result = Receive-Job $job
                $results.Add($result)
                Remove-Job $job
                $processed++

                if ($processed % 10 -eq 0) {
                    Write-Progress -Activity "Hashing files" -Status "$processed / $total" -PercentComplete (($processed / $total) * 100)
                }
            }
        }

        try {
            $jobs += Start-Job -ScriptBlock $scriptBlock -ArgumentList $file, $algorithm
        }
        catch {
            Write-Warn "Parallel jobs not available ($($_.Exception.Message)). Falling back to sequential hashing."
            $canUseJobs = $false
        }
    }

    if (-not $canUseJobs) {
        $seq = @()
        foreach ($f in $files) {
            try {
                $h = (Get-FileHash -Algorithm $algorithm -LiteralPath $f.FullName).Hash.ToLower()
                $seq += @{ Path = $f.FullName; Hash = $h; Success = $true }
            }
            catch {
                $seq += @{ Path = $f.FullName; Error = $_.Exception.Message; Success = $false }
            }
        }
        return $seq
    }

    # Wait for remaining jobs
    try {
        Wait-Job $jobs | Out-Null
    }
    catch {
        Write-Warn "Parallel wait failed ($($_.Exception.Message)). Falling back to sequential hashing."
        foreach ($j in @($jobs)) { try { Remove-Job $j -Force -ErrorAction SilentlyContinue } catch {} }
        $seq = @()
        $processed = 0
        $total = $files.Count
        foreach ($f in $files) {
            try {
                $h = Get-UnifiedFileHash -FilePath $f.FullName -Algorithm $algorithm
                $seq += @{ Path = $f.FullName; Hash = $h; Success = $true }
            }
            catch {
                $seq += @{ Path = $f.FullName; Error = $_.Exception.Message; Success = $false }
            }
            $processed++
            if ($processed % 10 -eq 0) {
                Write-Progress -Activity "Hashing files" -Status "$processed / $total" -PercentComplete (($processed / $total) * 100)
            }
        }
        Write-Progress -Activity "Hashing files" -Completed
        return $seq
    }

    foreach ($job in $jobs) {
        try {
            $result = Receive-Job $job -ErrorAction Stop
            $results.Add($result)
        }
        catch {
            $results.Add(@{ Path = '<job>'; Error = $_.Exception.Message; Success = $false })
        }
        finally {
            try { Remove-Job $job -Force -ErrorAction SilentlyContinue } catch {}
        }
    }

    Write-Progress -Activity "Hashing files" -Completed

    return $results
}

function Get-RelativePath([string]$root, [string]$full) {
    $root = $root.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if ($full.StartsWith($root)) {
        return $full.Substring($root.Length)
    }
    return $full
}
#endregion

#region Git Integration
function Get-GitCommit {
    if (Test-Path .git) {
        try {
            $commit = git rev-parse HEAD 2>$null
            if ($LASTEXITCODE -eq 0) { return $commit }
        }
        catch {}
    }
    return "N/A"
}

function Get-GitTag {
    if (Test-Path .git) {
        try {
            $tag = git describe --tags --exact-match 2>$null
            if ($LASTEXITCODE -eq 0) { return $tag }
        }
        catch {}
    }
    return "N/A"
}
#endregion

#region Main Execution
try {
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    $root = Resolve-Path -LiteralPath $Path
    Write-Info "Root path: $root"

    # Handle -All flag: run all hash algorithms WITH ALL EXPERIMENTAL FEATURES
    if ($All) {
        $allAlgorithms = @('SHA256', 'SHA1', 'SHA384', 'SHA512', 'BLAKE2B')
        Write-Info "═══════════════════════════════════════════════"
        Write-Info "Running ALL hash algorithms with ALL features enabled"
        Write-Info "Algorithms: $($allAlgorithms -join ', ')"
        Write-Info "Features: FrequencySignature, EntropyAnalysis, PerceptualHash, ExtractMetadata, GenerateSBOM, EnableHistory, DeepArchives, GenerateReport"
        Write-Info "═══════════════════════════════════════════════"

        foreach ($algo in $allAlgorithms) {
            Write-Info ""
            Write-Info "▶ Starting $algo hashing with ALL features..."

            # Recursively call this script with the current algorithm using hashtable splatting
            $splatParams = @{
                Path               = $Path
                Algorithm          = $algo
                FrequencySignature = $true
                EntropyAnalysis    = $true
                PerceptualHash     = $true
                ExtractMetadata    = $true
                GenerateSBOM       = $true
                EnableHistory      = $true
                DeepArchives       = $true
                GenerateReport     = $true
                ParallelVerify     = $true
                ExportJSON         = $true
                ExportCSV          = $true
            }

            # Pass through user-specified flags
            if ($SkipExisting) { $splatParams.SkipExisting = $true }
            if ($UseSidecar) { $splatParams.UseSidecar = $true }
            if ($Verify) { $splatParams.Verify = $true }
            if ($SignWithGPG) { $splatParams.SignWithGPG = $true }
            if ($TimestampProof) { $splatParams.TimestampProof = $true }
            if ($UseHashIgnore) { $splatParams.UseHashIgnore = $true }
            if ($UseGitIgnore) { $splatParams.UseGitIgnore = $true }
            if ($Verbose) { $splatParams.Verbose = $true }
            if ($ReportFormat) { $splatParams.ReportFormat = $ReportFormat }
            if ($ReleaseVersion) { $splatParams.ReleaseVersion = $ReleaseVersion }
            if ($ReleaseTitle) { $splatParams.ReleaseTitle = $ReleaseTitle }
            if ($MinFileSize -gt 0) { $splatParams.MinFileSize = $MinFileSize }
            if ($MaxFileSize -gt 0) { $splatParams.MaxFileSize = $MaxFileSize }
            if (-not $UseGitIgnore -and $Exclude.Count -gt 0) { $splatParams.Exclude = $Exclude }

            & $PSCommandPath @splatParams

            Write-Success "✓ Completed $algo hashing"
        }

        $stopwatch.Stop()
        Write-Info ""
        Write-Info "═══════════════════════════════════════════════"
        Write-Success "ALL algorithms completed in $([Math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds"
        Write-Info "═══════════════════════════════════════════════"

        exit 0
    }

    # Auto-detect index filename based on algorithm
    if (-not $IndexFile) {
        $IndexFile = switch ($Algorithm) {
            'SHA1' { 'SHA1SUMS.txt' }
            'SHA384' { 'SHA384SUMS.txt' }
            'SHA512' { 'SHA512SUMS.txt' }
            'BLAKE2B' { 'BLAKE2BSUMS.txt' }
            default { 'SHA256SUMS.txt' }
        }
    }

    $indexPath = Join-Path $root $IndexFile

    # Initialize database if history enabled
    $dbPath = Join-Path $root $script:Config.DatabaseFile
    $dbEnabled = $false
    if ($EnableHistory -or $ShowHistory -or $DiffMode) {
        $dbEnabled = Initialize-Database $dbPath
    }

    # Show history mode
    if ($ShowHistory) {
        if (-not $dbEnabled) {
            Write-Err "Database not available. Cannot show history."
            exit 1
        }

        Write-Info "File History Report"
        Write-Info "==================="
        # Implementation would query and display history
        exit 0
    }

    # Load hash ignore patterns
    $ignorePatterns = @()
    if ($UseHashIgnore) {
        $ignorePatterns = Get-HashIgnorePatterns $root
    }
    if ($UseGitIgnore) {
        $ignorePatterns += Get-GitIgnorePatterns $root
    }
    # Always exclude tool-generated artifacts to keep runs deterministic.
    $ignorePatterns += $Exclude
    $ignorePatterns += @(
        $script:Config.DatabaseFile,
        $script:Config.LogFile,
        $script:Config.MetadataDir,
        $script:Config.ReportDir,
        $script:Config.SBOMDir
    )

    # Git info
    $gitCommit = Get-GitCommit
    $gitTag = Get-GitTag

    Write-Info "Scanning files..."

    # Get all files
    $allFiles = Get-ChildItem -LiteralPath $root -Recurse -File -Force -ErrorAction SilentlyContinue

    $filesToProcess = @()
    foreach ($file in $allFiles) {
        # Skip system files
        if ($file.Attributes -band [System.IO.FileAttributes]::System) { continue }

        # Check ignore patterns
        $relPath = Get-RelativePath $root $file.FullName
        if (Test-ShouldIgnore $relPath $ignorePatterns) {
            Write-Debug "Skipping (ignored): $relPath"
            continue
        }

        # Size filters
        if ($MinFileSize -gt 0 -and $file.Length -lt $MinFileSize) { continue }
        if ($MaxFileSize -gt 0 -and $file.Length -gt $MaxFileSize) { continue }

        # Skip index file itself
        if ($file.FullName -ieq $indexPath) { continue }

        $filesToProcess += $file
    }

    Write-Info "Found $($filesToProcess.Count) files to process"

    # Compute hashes
    $fileData = @()

    if ($ParallelVerify -and $filesToProcess.Count -gt 10) {
        $results = Get-FileHashParallel $filesToProcess $Algorithm

        foreach ($result in $results) {
            if ($result.Success) {
                $file = $filesToProcess | Where-Object { $_.FullName -eq $result.Path } | Select-Object -First 1
                if ($file) {
                    $relPath = Get-RelativePath $root $file.FullName

                    $fileInfo = @{
                        FullPath     = $file.FullName
                        RelativePath = $relPath
                        Hash         = $result.Hash
                        Algorithm    = $Algorithm
                        Size         = $file.Length
                        Modified     = $file.LastWriteTime.ToString("o")
                    }

                    # Extract metadata if requested
                    if ($ExtractMetadata) {
                        $fileInfo.Metadata = Get-FileMetadata $file.FullName
                        Export-MetadataToFile $file.FullName $fileInfo.Metadata (Join-Path $root $script:Config.MetadataDir)
                    }

                    # Frequency signature if requested
                    if ($FrequencySignature) {
                        $fileInfo.FrequencySignature = Get-FrequencySignature $file.FullName
                    }

                    # Perceptual hash if requested
                    if ($PerceptualHash) {
                        $fileInfo.PerceptualHash = Get-PerceptualHash $file.FullName
                    }

                    $fileData += $fileInfo

                    # Add to history database
                    if ($dbEnabled -and $EnableHistory) {
                        Add-HistoryRecord -dbPath $dbPath -filePath $relPath -hash $result.Hash -algorithm $Algorithm `
                            -fileSize $file.Length -modifiedTime $file.LastWriteTime -gitCommit $gitCommit -gitTag $gitTag `
                            -metadata $fileInfo.Metadata
                    }
                }
            }
        }
    }
    else {
        $i = 0
        foreach ($file in $filesToProcess) {
            $i++
            Write-Progress -Activity "Hashing files" -Status "$i / $($filesToProcess.Count)" -PercentComplete (($i / $filesToProcess.Count) * 100)

            $relPath = Get-RelativePath $root $file.FullName

            try {
                $hash = $null
                if ($UseSidecar) {
                    $hash = Get-SidecarHash $file.FullName $Algorithm
                }

                if (-not $hash) {
                    $hash = Get-UnifiedFileHash -FilePath $file.FullName -Algorithm $Algorithm
                }

                $fileInfo = @{
                    FullPath     = $file.FullName
                    RelativePath = $relPath
                    Hash         = $hash
                    Algorithm    = $Algorithm
                    Size         = $file.Length
                    Modified     = $file.LastWriteTime.ToString("o")
                }

                if ($ExtractMetadata) {
                    $fileInfo.Metadata = Get-FileMetadata $file.FullName
                    Export-MetadataToFile $file.FullName $fileInfo.Metadata (Join-Path $root $script:Config.MetadataDir)
                }

                if ($FrequencySignature) {
                    $fileInfo.FrequencySignature = Get-FrequencySignature $file.FullName
                }

                if ($PerceptualHash) {
                    $fileInfo.PerceptualHash = Get-PerceptualHash $file.FullName
                }

                $fileData += $fileInfo

                if ($dbEnabled -and $EnableHistory) {
                    Add-HistoryRecord -dbPath $dbPath -filePath $relPath -hash $hash -algorithm $Algorithm `
                        -fileSize $file.Length -modifiedTime $file.LastWriteTime -gitCommit $gitCommit -gitTag $gitTag `
                        -metadata $fileInfo.Metadata
                }
            }
            catch {
                Write-Warn "Failed to hash $relPath : $_"
            }
        }

        Write-Progress -Activity "Hashing files" -Completed
    }

    # Write index file
    $lines = $fileData | Sort-Object RelativePath | ForEach-Object { "$($_.Hash) $($_.RelativePath)" }
    Set-Content -LiteralPath $indexPath -Value $lines -Encoding UTF8

    Write-Success "Wrote $IndexFile with $($fileData.Count) entries"

    # Generate SBOM if requested
    if ($GenerateSBOM) {
        $sbomPath = Join-Path $root $script:Config.SBOMDir
        if (-not (Test-Path $sbomPath)) {
            New-Item -ItemType Directory -Path $sbomPath -Force | Out-Null
        }

        $sbom = New-SBOM -rootPath $root -files $fileData -format 'SPDX'
        Export-SBOM -sbomData $sbom -outputPath (Join-Path $sbomPath "sbom.spdx.json") -format 'JSON'
    }

    # Generate HTML report if requested
    if ($GenerateReport) {
        $reportPath = Join-Path $root $script:Config.ReportDir
        if (-not (Test-Path $reportPath)) {
            New-Item -ItemType Directory -Path $reportPath -Force | Out-Null
        }

        $stopwatch.Stop()
        $stats = @{
            FileCount = $fileData.Count
            TotalSize = ($fileData | Measure-Object -Property Size -Sum).Sum
            Algorithm = $Algorithm
            Duration  = $stopwatch.Elapsed.TotalSeconds
        }

        $reportFile = Join-Path $reportPath "report-$(Get-Date -Format 'yyyyMMdd-HHmmss').html"
        New-HTMLReport -files $fileData -outputPath $reportFile -stats $stats

        if ($PSVersionTable.PSVersion.Major -ge 6) {
            Write-Info "Opening report in browser..."
            try {
                Start-Process $reportFile
            }
            catch {
                Write-Warn "Could not open report automatically: $($_.Exception.Message)"
            }
        }
    }

    # Export to JSON if requested
    if ($ExportJSON) {
        $jsonPath = Join-Path $root "$([System.IO.Path]::GetFileNameWithoutExtension($IndexFile)).json"
        $exportData = @{
            Generated      = (Get-Date).ToUniversalTime().ToString("o")
            Algorithm      = $Algorithm
            GitCommit      = $gitCommit
            GitTag         = $gitTag
            ReleaseVersion = $ReleaseVersion
            Files          = $fileData
        }
        $exportData | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
        Write-Success "Exported JSON to $jsonPath"
    }

    # Export to CSV if requested
    if ($ExportCSV) {
        $csvPath = Join-Path $root "$([System.IO.Path]::GetFileNameWithoutExtension($IndexFile)).csv"
        $fileData | Select-Object RelativePath, Hash, Size, Modified | Export-Csv -LiteralPath $csvPath -NoTypeInformation
        Write-Success "Exported CSV to $csvPath"
    }

    # GPG signing
    if ($SignWithGPG) {
        if (Test-Command 'gpg') {
            Write-Info "Signing with GPG..."
            $sigPath = "$indexPath.asc"

            if (Test-Path $sigPath) {
                Remove-Item $sigPath -Force
            }

            & gpg --detach-sign --armor --output $sigPath $indexPath 2>$null

            if ($LASTEXITCODE -eq 0) {
                Write-Success "GPG signature created: $sigPath"
            }
            else {
                Write-Warn "GPG signing failed"
            }
        }
        else {
            Write-Warn "GPG not found. Skipping signature."
        }
    }

    # OpenTimestamps proof
    if ($TimestampProof) {
        if (Test-Command 'ots') {
            Write-Info "Creating OpenTimestamps proof..."
            $otsPath = "$indexPath.ots"

            if (Test-Path $otsPath) {
                Remove-Item $otsPath -Force
            }

            & ots stamp $indexPath 2>$null

            if ($LASTEXITCODE -eq 0 -and (Test-Path $otsPath)) {
                Write-Success "Timestamp proof created: $otsPath"
                Write-Info "Verify later with: ots verify $otsPath"
            }
            else {
                Write-Warn "Timestamp creation incomplete (requires Bitcoin network)"
            }
        }
        else {
            Write-Warn "OpenTimestamps (ots) not found. Install: pip install opentimestamps-client"
        }
    }

    # Record scan session in database
    if ($dbEnabled -and $EnableHistory) {
        $stopwatch.Stop()
        Add-ScanSession -dbPath $dbPath -rootPath $root -algorithm $Algorithm `
            -fileCount $fileData.Count -totalSize (($fileData | Measure-Object -Property Size -Sum).Sum) `
            -duration $stopwatch.Elapsed.TotalSeconds -gitCommit $gitCommit -gitTag $gitTag `
            -releaseVersion $ReleaseVersion
    }

    $stopwatch.Stop()

    Write-Success "═══════════════════════════════════════════════"
    Write-Success "Scan completed in $([Math]::Round($stopwatch.Elapsed.TotalSeconds, 2)) seconds"
    Write-Success "Files processed: $($fileData.Count)"
    Write-Success "Total size: $(Format-FileSize (($fileData | Measure-Object -Property Size -Sum).Sum))"
    Write-Success "═══════════════════════════════════════════════"

    Write-Info ""
    Write-Info "Next steps:"
    Write-Info " * Review: $IndexFile"
    if ($GenerateReport) { Write-Info " * View report: .reports/report-*.html" }
    if ($SignWithGPG) { Write-Info " * Verify signature: gpg --verify $IndexFile.asc" }
    if ($GenerateSBOM) { Write-Info " * Review SBOM: .sbom/sbom.spdx.json" }
    if ($EnableHistory) { Write-Info " * View history: .\hash-index.ps1 -ShowHistory" }
    Write-Info " * Commit: git add $IndexFile && git commit -S -m 'Update checksums'"
    Write-Info ""
}
catch {
    Write-Err "Fatal error: $_"
    Write-Err $_.ScriptStackTrace
    exit 1
}

function Test-Command {
    param([string]$command)

    try {
        $null = Get-Command $command -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}
