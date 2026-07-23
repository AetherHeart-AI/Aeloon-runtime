import {
  RGBA,
  SyntaxStyle,
  parseColor,
  type ColorInput,
} from "@opentui/core"

export interface Palette {
  assistant: ColorInput
  background: ColorInput
  border: ColorInput
  borderFocus: ColorInput
  borderStrong: ColorInput
  borderSubtle: ColorInput
  cancelled: ColorInput
  error: ColorInput
  foreground: ColorInput
  muted: ColorInput
  panel: ColorInput
  selection: ColorInput
  success: ColorInput
  system: ColorInput
  userBand: ColorInput
  warning: ColorInput
}

/**
 * Mixes an overlay into a base color without relying on terminal alpha support.
 * Keeping the result as RGBA lets OpenTUI quantize it for the active terminal.
 */
export function blend(base: ColorInput, overlay: ColorInput, amount: number): RGBA {
  const ratio = Math.max(0, Math.min(1, amount))
  const [baseR, baseG, baseB, baseA] = parseColor(base).toInts()
  const [overlayR, overlayG, overlayB, overlayA] = parseColor(overlay).toInts()
  const channel = (from: number, to: number) => Math.round(from + (to - from) * ratio)
  return RGBA.fromInts(
    channel(baseR, overlayR),
    channel(baseG, overlayG),
    channel(baseB, overlayB),
    channel(baseA, overlayA),
  )
}

const DARK_BACKGROUND = "#141414"
const DARK_FOREGROUND = "#D7D4D0"
const DARK_ASSISTANT = "#D58BC8"
const DARK_SYSTEM = "#7FA6D8"

export const DARK: Palette = {
  assistant: DARK_ASSISTANT,
  background: DARK_BACKGROUND,
  border: blend(DARK_BACKGROUND, DARK_FOREGROUND, 0.18),
  borderFocus: blend(DARK_BACKGROUND, DARK_FOREGROUND, 0.43),
  borderStrong: blend(DARK_BACKGROUND, DARK_FOREGROUND, 0.29),
  borderSubtle: blend(DARK_BACKGROUND, DARK_FOREGROUND, 0.1),
  cancelled: "#85817D",
  error: "#DE737A",
  foreground: DARK_FOREGROUND,
  muted: "#85817D",
  panel: blend(DARK_BACKGROUND, DARK_FOREGROUND, 0.04),
  selection: blend(DARK_BACKGROUND, DARK_SYSTEM, 0.3),
  success: "#83B98A",
  system: DARK_SYSTEM,
  userBand: blend(DARK_BACKGROUND, "#E8DDCE", 0.055),
  warning: "#C4A46A",
}

const LIGHT_BACKGROUND = "#F2F1EE"
const LIGHT_FOREGROUND = "#292725"
const LIGHT_ASSISTANT = "#9D3E83"
const LIGHT_SYSTEM = "#376EAB"

export const LIGHT: Palette = {
  assistant: LIGHT_ASSISTANT,
  background: LIGHT_BACKGROUND,
  border: blend(LIGHT_BACKGROUND, LIGHT_FOREGROUND, 0.18),
  borderFocus: blend(LIGHT_BACKGROUND, LIGHT_FOREGROUND, 0.5),
  borderStrong: blend(LIGHT_BACKGROUND, LIGHT_FOREGROUND, 0.3),
  borderSubtle: blend(LIGHT_BACKGROUND, LIGHT_FOREGROUND, 0.1),
  cancelled: "#77716C",
  error: "#B33A43",
  foreground: LIGHT_FOREGROUND,
  muted: "#77716C",
  panel: blend(LIGHT_BACKGROUND, LIGHT_FOREGROUND, 0.035),
  selection: blend(LIGHT_BACKGROUND, LIGHT_SYSTEM, 0.22),
  success: "#347742",
  system: LIGHT_SYSTEM,
  userBand: blend(LIGHT_BACKGROUND, "#8A6F4E", 0.07),
  warning: "#8A640D",
}

export function createMarkdownSyntaxStyle(palette: Palette): SyntaxStyle {
  return SyntaxStyle.fromStyles({
    default: { fg: palette.foreground },
    conceal: { fg: palette.borderStrong, dim: true },
    comment: { fg: palette.muted, italic: true },
    keyword: { fg: palette.assistant },
    number: { fg: palette.warning },
    string: { fg: palette.success },
    type: { fg: palette.system },
    variable: { fg: palette.foreground },
    "markup.heading": { fg: palette.assistant, bold: true },
    "markup.strong": { fg: palette.foreground, bold: true },
    "markup.italic": { fg: palette.foreground, italic: true },
    "markup.strikethrough": { fg: palette.muted, dim: true },
    "markup.raw": {
      bg: blend(palette.background, palette.foreground, 0.08),
      fg: palette.warning,
    },
    "markup.list": { fg: palette.system },
    "markup.quote": { fg: palette.muted, italic: true },
    "markup.link": { fg: palette.borderStrong, dim: true },
    "markup.link.label": { fg: palette.system, underline: true },
    "markup.link.url": { fg: palette.muted, underline: true },
  })
}
