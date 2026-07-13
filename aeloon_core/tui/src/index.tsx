import { render } from "@opentui/solid"
import { App } from "./app"
import { DARK } from "./theme"

await render(() => <App />, {
  backgroundColor: DARK.background,
  clearOnShutdown: true,
  consoleMode: "disabled",
  enableMouseMovement: true,
  exitOnCtrlC: false,
  maxFps: 60,
  screenMode: "alternate-screen",
  targetFps: 30,
  useMouse: true,
})
