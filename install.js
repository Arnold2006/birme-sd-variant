module.exports = {
  run: [
    // Install PyTorch first so the correct GPU/CUDA wheels are already
    // present when requirements.txt is resolved (prevents pip from pulling
    // in a CPU-only torch as a transitive dependency of transformers/accelerate).
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env"
        }
      }
    },
    // Install remaining Python packages
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install flask",
          "uv pip install -r requirements.txt"
        ]
      }
    }
  ]
}
