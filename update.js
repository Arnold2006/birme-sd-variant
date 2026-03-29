module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "venv",
        message: "pip install -r requirements.txt --upgrade",
      }
    }
  ]
}
