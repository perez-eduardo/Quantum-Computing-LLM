# Run from E:\Personal_projects\Quantum-Computing-LLM\

$folders = @(
    "docs",
    "data\raw\books",
    "data\processed",
    "training\scripts",
    "training\tokenizer",
    "training\model",
    "backend",
    "frontend"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
    Write-Host "Created: $folder"
}

Write-Host "`nDone. Now move your files to the appropriate folders."
