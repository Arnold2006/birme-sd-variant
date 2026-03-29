module.exports = {
  version: "2.5",
  title: "Birme SD Variant",
  description: "Bulk image resizer for Stable Diffusion training datasets with JoyCaption AI captioning.",
  icon: "static/images/favicon.png",
  menu: async (kernel, info) => {
    let running   = await kernel.running("start.js")
    let installed = await kernel.exists("venv")

    if (running) {
      return [
        { text: "Stop",   href: "stop.js",              mode: "stop"     },
        { text: "Open",   href: "http://localhost:7861", target: "_blank" },
        { text: "Update", href: "update.js"                               },
      ]
    } else if (installed) {
      return [
        { text: "Start",  href: "start.js"  },
        { text: "Update", href: "update.js" },
      ]
    } else {
      return [
        { text: "Install", href: "install.js" },
      ]
    }
  }
}
