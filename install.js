module.exports = {
  run: [
    // Install Python packages
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install flask",
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Install PyTorch (handled by torch.js)
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env"
        }
      }
    }
  ]
}
