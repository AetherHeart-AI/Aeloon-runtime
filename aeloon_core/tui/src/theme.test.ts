import { expect, test } from "bun:test"
import { parseColor } from "@opentui/core"
import { DARK, blend } from "./theme"

test("dark theme uses a neutral canvas and ordered chrome contrast", () => {
  expect(parseColor(DARK.background).toInts()).toEqual([20, 20, 20, 255])

  const brightness = (color: typeof DARK.background) => {
    const [red, green, blue] = parseColor(color).toInts()
    return red + green + blue
  }

  expect(brightness(DARK.borderSubtle)).toBeLessThan(brightness(DARK.border))
  expect(brightness(DARK.border)).toBeLessThan(brightness(DARK.borderStrong))
  expect(brightness(DARK.borderStrong)).toBeLessThan(brightness(DARK.borderFocus))
})

test("blend clamps its overlay amount", () => {
  expect(blend("#000000", "#FFFFFF", -1).toInts()).toEqual([0, 0, 0, 255])
  expect(blend("#000000", "#FFFFFF", 2).toInts()).toEqual([255, 255, 255, 255])
})
