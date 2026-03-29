module.exports = {
  run: [{
    method: "shell.run",
    params: {
      venv: "{{params.venv}}",
      message: "uv pip install torch torchvision torchaudio"
    }
  }]
}
