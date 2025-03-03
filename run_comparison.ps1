param(
    [int]$size = 5, # size of environment
    [int]$episodes = 1000, # number of episodes
    [int]$runs = 100, # number of runs to perform
    [string]$folder = "data" # folder to store the data in
)

function Run-Alg {
    param(
        [string]$alg, # algorithm to use [q, bl, crm]
        [int]$i # current iteration
    )
    & python "src/run.py" $alg -s $size -e $episodes -x $i -f $folder
    if ($LASTEXITCODE -ne 0) {
        Write-Error "run.py failed on iteration $i of algorithm $alg."
        exit $LASTEXITCODE
    }
}



. ".venv\Scripts\activate.ps1"

for ($i = 1; $i -le $runs; $i++) {
    Write-Host -NoNewline "`rRunning algorithms on ${size}x$size for $episodes episodes: $i/$runs`r"
    Run-Alg -alg "q" -i $i
    Run-Alg -alg "bl" -i $i
    Run-Alg -alg "crm" -i $i
    Run-Alg -alg "bl2" -i $i
    Run-Alg -alg "crm2" -i $i
}

Write-Host "`nFinished running algorithms. Generating plots."

& python "src/plot.py" -n $runs -s $size -f $folder

if ($LASTEXITCODE -ne 0) {
    Write-Error "plot.py failed."
    exit $LASTEXITCODE
}

