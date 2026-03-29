module.exports = {
  run: [
    // 1. Install PyTorch with automatic CUDA/ROCm/CPU detection (Pinokio v5 built-in)
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "venv",
          torch: "auto",
        }
      }
    },
    // 2. Install remaining Python dependencies into the same venv
    {
      method: "shell.run",
      params: {
        venv: "venv",
        message: "pip install -r requirements.txt",
      }
    }
  ]
}
