import { test, expect } from "@playwright/test"

async function backendIsUp(request: any): Promise<boolean> {
  try {
    const resp = await request.get("http://localhost:8010/health", { timeout: 3000 })
    return resp.ok()
  } catch {
    return false
  }
}

test.describe("Navigation — UI only (no backend needed)", () => {

  test("LLM Platform logo links to home", async ({ page }) => {
    await page.goto("/")
    await page.locator("nav a").filter({ hasText: "LLM Platform" }).click()
    await expect(page).toHaveURL("/")
  })

  test("Projects nav link goes to home", async ({ page }) => {
    await page.goto("/")
    await page.locator("nav").getByRole("link", { name: "Projects" }).click()
    await expect(page).toHaveURL("/")
  })

  test("theme toggle is present and clickable without errors", async ({ page }) => {
    const errors: string[] = []
    page.on("pageerror", e => errors.push(e.message))
    await page.goto("/")
    const toggle = page.locator("nav button[aria-label='Toggle theme']")
    await expect(toggle).toBeVisible()
    await toggle.click()
    await toggle.click()
    expect(errors).toHaveLength(0)
  })

})

test.describe("Navigation — requires backend at :8010", () => {

  test("project card links to project page", async ({ page, request }) => {
    test.skip(!(await backendIsUp(request)), "Backend not running — skipping API test")

    await page.goto("/")
    await page.getByRole("button", { name: /new project/i }).first().click()
    await page.getByPlaceholder(/project name/i).fill("Nav Test Project")
    await Promise.all([
      page.waitForResponse(r => r.url().includes("/projects") && r.status() === 201),
      page.getByRole("button", { name: /^create/i }).click(),
    ])
    await page.getByText("Nav Test Project").click()
    await expect(page).toHaveURL(/\/projects\/[a-f0-9]+/)
  })

  test("project page shows sidebar navigation links", async ({ page, request }) => {
    test.skip(!(await backendIsUp(request)), "Backend not running — skipping API test")

    await page.goto("/")
    await page.getByRole("button", { name: /new project/i }).first().click()
    await page.getByPlaceholder(/project name/i).fill("Sidebar Test")
    await Promise.all([
      page.waitForResponse(r => r.url().includes("/projects") && r.status() === 201),
      page.getByRole("button", { name: /^create/i }).click(),
    ])
    await page.getByText("Sidebar Test").click()
    await expect(page).toHaveURL(/\/projects\/[a-f0-9]+/)

    await expect(page.getByRole("link", { name: /datasets/i })).toBeVisible()
    await expect(page.getByRole("link", { name: /jobs/i })).toBeVisible()
    await expect(page.getByRole("link", { name: /evaluate/i })).toBeVisible()
    await expect(page.getByRole("link", { name: /inference/i })).toBeVisible()
    await expect(page.getByRole("link", { name: /rag/i })).toBeVisible()
  })

})
