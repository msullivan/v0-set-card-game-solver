import { test, expect } from "@playwright/test"

// Uses /test-results — a fixture page that renders SetResults with 3 pre-loaded
// mock cards (red/green/purple diamond-solid-1) forming exactly one valid set.

test.describe("Card edit feature", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/test-results")
    await expect(page.getByText("Detected Cards")).toBeVisible()
  })

  test("displays detected cards and valid sets", async ({ page }) => {
    await expect(page.getByText("3 cards detected")).toBeVisible()
    await expect(page.getByText("1 valid set found")).toBeVisible()
    await expect(page.getByText("High confidence")).toBeVisible()
  })

  test("cards in detected grid have edit affordance (cursor-pointer)", async ({ page }) => {
    const grid = page.getByTestId("detected-cards-grid")
    const cards = grid.locator('[class*="cursor-pointer"]')
    await expect(cards).toHaveCount(3)
  })

  test("clicking a card opens the edit dialog", async ({ page }) => {
    const grid = page.getByTestId("detected-cards-grid")
    await grid.locator('[class*="cursor-pointer"]').first().click()

    const dialog = page.getByRole("dialog")
    await expect(dialog).toBeVisible()
    await expect(dialog.getByRole("heading", { name: "Edit Card" })).toBeVisible()
    await expect(dialog.getByText("Color", { exact: true })).toBeVisible()
    await expect(dialog.getByText("Shape", { exact: true })).toBeVisible()
    await expect(dialog.getByText("Shading", { exact: true })).toBeVisible()
    await expect(dialog.getByText("Number", { exact: true })).toBeVisible()
  })

  test("edit dialog shows all attribute options", async ({ page }) => {
    const grid = page.getByTestId("detected-cards-grid")
    await grid.locator('[class*="cursor-pointer"]').first().click()

    const dialog = page.getByRole("dialog")
    await expect(dialog.getByRole("button", { name: /red/i })).toBeVisible()
    await expect(dialog.getByRole("button", { name: /green/i })).toBeVisible()
    await expect(dialog.getByRole("button", { name: /purple/i })).toBeVisible()
    await expect(dialog.getByRole("button", { name: /diamond/i })).toBeVisible()
    await expect(dialog.getByRole("button", { name: /oval/i })).toBeVisible()
    await expect(dialog.getByRole("button", { name: /squiggle/i })).toBeVisible()
    await expect(dialog.getByRole("button", { name: /solid/i })).toBeVisible()
    await expect(dialog.getByRole("button", { name: /striped/i })).toBeVisible()
    await expect(dialog.getByRole("button", { name: /empty/i })).toBeVisible()
    await expect(dialog.getByRole("button", { name: "1" })).toBeVisible()
    await expect(dialog.getByRole("button", { name: "2" })).toBeVisible()
    await expect(dialog.getByRole("button", { name: "3" })).toBeVisible()
  })

  test("cancel closes the dialog without changing valid sets", async ({ page }) => {
    const grid = page.getByTestId("detected-cards-grid")
    await grid.locator('[class*="cursor-pointer"]').first().click()

    await expect(page.getByRole("dialog")).toBeVisible()
    await page.getByRole("button", { name: "Cancel" }).click()
    await expect(page.getByRole("dialog")).not.toBeVisible()
    await expect(page.getByText("1 valid set found")).toBeVisible()
  })

  test("changing a card color breaks the valid set", async ({ page }) => {
    // Cards: red, green, purple → 1 valid set (all different colors)
    // Change red → green: now green, green, purple → no valid set
    const grid = page.getByTestId("detected-cards-grid")
    await grid.locator('[class*="cursor-pointer"]').first().click()

    await page.getByRole("dialog").getByRole("button", { name: /^green$/i }).click()
    await page.getByRole("button", { name: "Save" }).click()

    await expect(page.getByRole("dialog")).not.toBeVisible()
    await expect(page.locator('[class*="bg-destructive"]').filter({ hasText: "No valid sets found" })).toBeVisible()
  })

  test("correcting a card restores the valid set", async ({ page }) => {
    const grid = page.getByTestId("detected-cards-grid")

    // Break: change red → green
    await grid.locator('[class*="cursor-pointer"]').first().click()
    await page.getByRole("dialog").getByRole("button", { name: /^green$/i }).click()
    await page.getByRole("button", { name: "Save" }).click()
    await expect(page.locator('[class*="bg-destructive"]').filter({ hasText: "No valid sets found" })).toBeVisible()

    // Fix: change it back to red
    await grid.locator('[class*="cursor-pointer"]').first().click()
    await page.getByRole("dialog").getByRole("button", { name: /^red$/i }).click()
    await page.getByRole("button", { name: "Save" }).click()

    await expect(page.getByText("1 valid set found")).toBeVisible()
  })

  test("changing number attribute breaks the valid set", async ({ page }) => {
    // Change third card from number 1 to number 2 → no longer all-same number, no set
    const grid = page.getByTestId("detected-cards-grid")
    await grid.locator('[class*="cursor-pointer"]').nth(2).click()

    await page.getByRole("dialog").getByRole("button", { name: "2" }).click()
    await page.getByRole("button", { name: "Save" }).click()

    await expect(page.locator('[class*="bg-destructive"]').filter({ hasText: "No valid sets found" })).toBeVisible()
  })
})

test.describe("Demo page", () => {
  test("renders all 81 cards", async ({ page }) => {
    await page.goto("/demo")
    await expect(page.getByRole("heading", { name: "All 81 Set Cards" })).toBeVisible()
    const cards = page.locator('[class*="aspect-\\[5\\/7\\]"]')
    await expect(cards).toHaveCount(81)
  })

  test("renders all three color sections", async ({ page }) => {
    await page.goto("/demo")
    await expect(page.getByRole("heading", { name: "red", exact: true })).toBeVisible()
    await expect(page.getByRole("heading", { name: "green", exact: true })).toBeVisible()
    await expect(page.getByRole("heading", { name: "purple", exact: true })).toBeVisible()
  })
})
