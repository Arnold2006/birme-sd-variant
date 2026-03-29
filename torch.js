module.exports = {
  run: [
    // NVIDIA GPU: install CUDA 12.1 wheels
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "{{params.venv}}",
        message: "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
      }
    },
    // AMD GPU: install ROCm wheels
    {
      when: "{{gpu === 'amd'}}",
      method: "shell.run",
      params: {
        venv: "{{params.venv}}",
        message: "uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1"
      }
    },
    // CPU-only fallback (Apple Silicon, no GPU, etc.)
    {
      when: "{{gpu !== 'nvidia' && gpu !== 'amd'}}",
      method: "shell.run",
      params: {
        venv: "{{params.venv}}",
        message: "uv pip install torch torchvision torchaudio"
      }
    }
  ]
}
