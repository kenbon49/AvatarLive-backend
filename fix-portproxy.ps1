# fix-portproxy.ps1
# 用法: 管理员 PowerShell 中运行 .\fix-portproxy.ps1
# 作用: 刷新 WSL2 → LAN IP 的 8083 端口转发
# 适用: WSL2 / Docker Desktop 模式下,让其他机器通过 LAN IP 访问容器服务

$Port = 8083
$LanIp = "10.4.124.27"

Write-Host "🔍 获取当前 WSL2 IP..." -ForegroundColor Cyan
$wslIP = (wsl hostname -I).Trim().Split()[0]
if (-not $wslIP) {
    Write-Host "❌ 无法获取 WSL IP,WSL 是否在运行?" -ForegroundColor Red
    exit 1
}
Write-Host "   WSL IP: $wslIP"

Write-Host "🧹 清理旧规则..." -ForegroundColor Cyan
netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0 2>$null

Write-Host "➕ 添加新规则..." -ForegroundColor Cyan
netsh interface portproxy add v4tov4 listenport=$Port listenaddress=0.0.0.0 connectport=$Port connectaddress=$wslIP

Write-Host "📋 当前规则:" -ForegroundColor Cyan
netsh interface portproxy show v4tov4

Write-Host "🧪 测试 LAN 访问 http://$LanIp`:$Port/health..." -ForegroundColor Cyan
try {
    $resp = Invoke-RestMethod "http://$LanIp`:$Port/health" -TimeoutSec 5
    Write-Host "✅ 成功: $($resp | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "⚠️  HTTP 仍不通,尝试加防火墙规则..." -ForegroundColor Yellow
    New-NetFirewallRule -DisplayName "MuseTalk $Port" -Direction Inbound -LocalPort $Port -Protocol TCP -Action Allow -ErrorAction SilentlyContinue | Out-Null
    Start-Sleep 2
    try {
        $resp = Invoke-RestMethod "http://$LanIp`:$Port/health" -TimeoutSec 5
        Write-Host "✅ 加防火墙后成功: $($resp | ConvertTo-Json -Compress)" -ForegroundColor Green
    } catch {
        Write-Host "❌ 仍失败: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "   建议:检查 docker ps 容器是否在跑;检查 WSL 网络" -ForegroundColor Yellow
    }
}
