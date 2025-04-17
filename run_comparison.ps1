param(
    [int]$size = 5, # size of environment
    [int]$episodes = 1000, # number of episodes
    [int]$runs = 100, # number of runs to perform
    [string]$folder = "data" # folder to store the data in
)

$algorithms = @("q", "bl", "crm", "bl2", "crm2")

# maximum number of parallel jobs, adjust based on workload
[int]$maxParallel = [math]::Round([System.Environment]::ProcessorCount * 3 / 4)

# activate virtual environment
. ".venv\Scripts\activate.ps1"

$jobList = foreach ($i in 1..$runs) {
    $algorithms | ForEach-Object {
        [PSCustomObject]@{ Algorithm = $_; Iteration = $i }
    }
}

# run in parallel
$jobList | ForEach-Object -Parallel {
    & python "src/run.py" $_.Algorithm -s $using:size -e $using:episodes -x $_.Iteration -f $using:folder
    if ($LASTEXITCODE -ne 0) {
        Write-Error "run.py failed on iteration $($_.Iteration) of algorithm $($_.Algorithm)."
        exit $LASTEXITCODE
    }
    $_
} -ThrottleLimit $maxParallel | ForEach-Object -Begin { $received = 0} -Process {
    $received += 1
    [int] $percentComplete = ($received / $jobList.Count) * 100
    Write-Progress -Activity "Running algorithms on $($size)x$size for $episodes episodes" -Status "$percentComplete% complete" -PercentComplete $percentComplete
    $_
}

# generate plots after completion
Write-Host "`nFinished running algorithms. Generating plots."
& python "src/plot.py" -n $runs -s $size -f $folder
