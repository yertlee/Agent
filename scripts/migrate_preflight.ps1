param(
    [Parameter(Mandatory=$true)][string]$InputDb,
    [Parameter(Mandatory=$false)][string]$Schema,
    [Parameter(Mandatory=$false)][string]$Report
)

$arguments = @('-m', 'agent.storage.preflight', '--input', $InputDb)
if ($Schema) { $arguments += @('--schema', $Schema) }
if ($Report) { $arguments += @('--report', $Report) }
python @arguments
exit $LASTEXITCODE
