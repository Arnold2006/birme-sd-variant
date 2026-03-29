module.exports = {
  run: [
    // Start the Flask server; advance to the next step once it reports "Running on"
    {
      method: "shell.run",
      params: {
        id: "server",
        venv: "venv",
        message: "python server.py",
        on: [{
          event: "/Running on/",
          done: true
        }]
      }
    },
    // Open the app in the Pinokio browser
    {
      method: "browser.open",
      params: {
        uri: "http://localhost:7861"
      }
    }
  ]
}
