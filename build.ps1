Write-Host "Building Rust Vector Search Application..." -ForegroundColor Green

# Build the backend
Write-Host "Building backend server..." -ForegroundColor Yellow
Set-Location backend
cargo build --bin server
Set-Location ..

# Check if wasm-pack is installed
$wasmPackExists = Get-Command wasm-pack -ErrorAction SilentlyContinue
if ($wasmPackExists) {
    Write-Host "Building frontend WASM package..." -ForegroundColor Yellow
    Set-Location frontend
    wasm-pack build --target web --out-dir pkg
    Set-Location ..
} else {
    Write-Host "wasm-pack not found. Install it with: cargo install wasm-pack" -ForegroundColor Red
}

Write-Host "Build complete!" -ForegroundColor Green
Write-Host "To run the backend server: cd backend && cargo run --bin server" -ForegroundColor Cyan
Write-Host "Then visit: http://127.0.0.1:8000" -ForegroundColor Cyan