$zipInput = $args[0]
$outputDir = $args[1]

Add-Type -AssemblyName System.IO.Compression.FileSystem
function Unzip
{
    param([string]$zipfile, [string]$outpath)
    write-output 'Extracting ' $zipfile ' into ' $outputDir
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipfile, $outpath)
}

Unzip $zipInput $outputDir