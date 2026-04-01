param(
    [string]$DbPath = "",
    [ValidateSet("stub", "real")]
    [string]$LogisticsMode = "stub",
    [string]$PythonExe = "",
    [switch]$RunInspect
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    param([string]$Preferred)
    if ($Preferred -and (Get-Command $Preferred -ErrorAction SilentlyContinue)) {
        return $Preferred
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    throw "未找到 Python 可执行文件，请安装 Python，或通过 -PythonExe 指定解释器。"
}

function Resolve-DbPath {
    param([string]$InputDbPath)
    if ($InputDbPath) {
        return (Resolve-Path $InputDbPath).Path
    }
    $default = Join-Path $PSScriptRoot "..\ecommerce.db"
    return (Resolve-Path $default).Path
}

Write-Host "== Agent Eval Runner ==" -ForegroundColor Cyan

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runEvalPath = Join-Path $repoRoot "eval\run_eval.py"
$inspectPath = Join-Path $repoRoot "eval\inspect.py"
$reportJsonPath = Join-Path $repoRoot "reports\agent_v3_eval.json"
$reportMdPath = Join-Path $repoRoot "reports\agent_v3_eval.md"

if (-not (Test-Path $runEvalPath)) {
    throw "缺少评测入口文件: $runEvalPath"
}

$pythonCmd = Resolve-Python -Preferred $PythonExe
$resolvedDbPath = Resolve-DbPath -InputDbPath $DbPath

if (-not (Test-Path $resolvedDbPath)) {
    throw "数据库文件不存在: $resolvedDbPath"
}

$env:ECOMMERCE_DB_PATH = $resolvedDbPath
$env:AGENT_EVAL_LOGISTICS_MODE = $LogisticsMode

Write-Host "Python        : $pythonCmd"
Write-Host "DB Path       : $env:ECOMMERCE_DB_PATH"
Write-Host "LogisticsMode : $env:AGENT_EVAL_LOGISTICS_MODE"
Write-Host ""
Write-Host "正在运行 eval/run_eval.py ..." -ForegroundColor Yellow

Push-Location $repoRoot
try {
    if ($pythonCmd -eq "py") {
        & py -3 $runEvalPath
    } else {
        & $pythonCmd $runEvalPath
    }
    if ($LASTEXITCODE -ne 0) {
        throw "run_eval.py 执行失败，退出码: $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "评测完成，报告输出如下：" -ForegroundColor Green
    Write-Host "JSON: $reportJsonPath"
    Write-Host "MD  : $reportMdPath"

    if ($RunInspect) {
        if (-not (Test-Path $inspectPath)) {
            throw "缺少 inspect 文件: $inspectPath"
        }
        Write-Host ""
        Write-Host "正在执行 eval/inspect.py ..." -ForegroundColor Yellow
        if ($pythonCmd -eq "py") {
            & py -3 $inspectPath
        } else {
            & $pythonCmd $inspectPath
        }
        if ($LASTEXITCODE -ne 0) {
            throw "inspect.py 执行失败，退出码: $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
