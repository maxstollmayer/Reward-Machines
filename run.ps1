param (
    [int]$runs = 10,
    [int]$size = 5,
    [int]$episodes = 1010,
    [int]$rolling_length = 10,
    [string]$folder = "data"
)

function Run-Alg {
    param (
        [string]$alg, # algorithm to use [q, crm]
        [int]$i # current iteration
    )
    & python "src/run.py" $alg -s $size -e $episodes -x $i -f $folder
    if ($LASTEXITCODE -ne 0) {
        Write-Error "`nERROR: run.py failed on iteration $i of algorithm $alg."
        exit $LASTEXITCODE
    }
}



. ".venv\Scripts\activate.ps1"

for ($i = 1; $i -le $runs; $i++) {
    Write-Host -NoNewline "`rRunning algorithms on ${size}x$size for $episodes episodes: $i/$runs`r"
    Run-Alg -alg "q" -i $i
    Run-Alg -alg "crm" -i $i
}

Write-Host "`nFinished running algorithms. Generating plot."

& python "src/plot.py" -n $runs -s $size -r $rolling_length -f $folder

if ($LASTEXITCODE -ne 0) {
    Write-Error "`nERROR: plot.py failed."
    exit $LASTEXITCODE
}

