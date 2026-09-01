param(
    [Parameter(Mandatory=$true)][string]$IsolatedDb,
    [Parameter(Mandatory=$false)][string]$Schema,
    [Parameter(Mandatory=$false)][string]$Report
)

# M0 has no schema-changing migration.  This command is an explicit dry-run
# adapter and accepts only an isolated database path supplied by the caller.
$arguments = @('-m', 'agent.storage.preflight', '--input', $IsolatedDb, '--postflight')
if ($Schema) { $arguments += @('--schema', $Schema) }
if ($Report) { $arguments += @('--report', $Report) }
python @arguments
exit $LASTEXITCODE
