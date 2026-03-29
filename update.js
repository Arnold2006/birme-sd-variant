module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    // Re-install PyTorch with the correct GPU/CUDA wheels so that users
    // who had a CPU-only torch (from an earlier install) get CUDA support
    // after running Update.
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env"
        }
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "uv pip install -r requirements.txt"
      }
    }
  ]
}
