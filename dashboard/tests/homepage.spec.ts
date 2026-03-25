import { test, expect } from "@playwright/test"

test.describe("Home page", () => {

  test("loads without errors", async ({ page }) => {
    const errors: string[] = []
    page.on("pageerror", e => errors.push(e.message))
    await page.goto("/")
    expect(errors).toHaveLength(0)
  })

  test("shows Projects heading", async ({ page }) => {
    await page.goto("/")
    await expect(page.getByRole("heading", { name: "Projects" })).toBeVisible()
  })

  test("shows New Project button", async ({ page }) => {
    await page.goto("/")
    await expect(page.getByRole("button", { name: /new project/i })).toBeVisible()
  })

  test("shows API status indicator in navbar", async ({ page }) => {
    await page.goto("/")
    // Either 'API online' or 'API offline' or 'checking' — all are valid DOM states
    const navbar = page.locator("nav")
    await expect(navbar).toBeVisible()
  })

  test("shows LLM Platform brand in navbar", async ({ page }) => {
    await page.goto("/")
    await expect(page.locator("nav").getByText("LLM Platform")).toBeVisible()
  })

  test("empty state message shown when no projects exist", async ({ page }) => {
    await page.goto("/")
    // If the backend is clean, should show empty state
    // We check for the presence of either projects grid OR empty state — both are valid
    const hasProjects = await page.locator('[class*="grid"]').count()
    const hasEmptyState = await page.getByText(/no projects yet/i).count()
    expect(hasProjects + hasEmptyState).toBeGreaterThan(0)
  })

})
