export interface Palette {
  accent: string
  background: string
  border: string
  cancelled: string
  error: string
  foreground: string
  muted: string
  panel: string
  selection: string
  success: string
  warning: string
}

export const DARK: Palette = {
  accent: "#67D4E6",
  background: "#0B0F14",
  border: "#2A3746",
  cancelled: "#8994A3",
  error: "#F07178",
  foreground: "#D7DEE7",
  muted: "#748196",
  panel: "#111821",
  selection: "#23495A",
  success: "#7BD88F",
  warning: "#E5B567",
}

export const LIGHT: Palette = {
  accent: "#006D83",
  background: "#F4F1EA",
  border: "#BAC0C5",
  cancelled: "#6F7881",
  error: "#B4232C",
  foreground: "#1C252C",
  muted: "#66727C",
  panel: "#E9E6DE",
  selection: "#B9DCE3",
  success: "#16713A",
  warning: "#8A5A00",
}
