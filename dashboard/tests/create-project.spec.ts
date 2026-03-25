import { test, expect } from "@playwright/test"

// Helper: check if backend is reachable before running API-dependent tests
async function backendIsUp(request: any): Promise<boolean> {
  try {
    const resp = await request.get("http://localhost:8010/health", { timeout: 3000 })
    return resp.ok()
  } catch {
    return false
  }
}

test.describe("Create project form — UI only (no backend needed)", () => {

  test("New Project button reveals the create form", async ({ page }) => {
    await page.goto("/")
    await page.getByRole("button", { name: /new project/i }).first().click()
    await expect(page.getByPlaceholder(/project name/i)).toBeVisible()
  })

  test("form has name and description inputs", async ({ page }) => {
    await page.goto("/")
    await page.getByRole("button", { name: /new project/i }).first().click()
    await expect(page.getByPlaceholder(/project name/i)).toBeVisible()
    await expect(page.getByPlaceholder(/description/i)).toBeVisible()
  })

  test("form has a submit button", async ({ page }) => {
    await page.goto("/")
    await page.getByRole("button", { name: /new project/i }).first().click()
    await expect(page.getByRole("button", { name: /create/i })).toBeVisible()
  })

  test("cancel button closes the form", async ({ page }) => {
    await page.goto("/")
    await page.getByRole("button", { name: /new project/i }).first().click()
    await expect(page.getByPlaceholder(/project name/i)).toBeVisible()
    await page.getByRole("button", { name: /cancel/i }).click()
    await expect(page.getByPlaceholder(/project name/i)).not.toBeVisible()
  })

  test("submit button is disabled when name is empty", async ({ page }) => {
    await page.goto("/")
    await page.getByRole("button", { name: /new project/i }).first().click()
    const submitBtn = page.getByRole("button", { name: /^create/i })
    // Name field is empty — submit should be disabled
    await expect(submitBtn).toBeDisabled()
  })

  test("submit button enables once name is typed", async ({ page }) => {
    await page.goto("/")
    await page.getByRole("button", { name: /new project/i }).first().click()
    await page.getByPlaceholder(/project name/i).fill("My Project")
    const submitBtn = page.getByRole("button", { name: /^create/i })
    await expect(submitBtn).toBeEnabled()
  })

})

test.describe("Create project flow — requires backend at :8010", () => {

  test("submitting a project name closes the form", async ({ page, request }) => {
    test.skip(!(await backendIsUp(request)), "Backend not running — skipping API test")

    await page.goto("/")
    await page.getByRole("button", { name: /new project/i }).first().click()
    await page.getByPlaceholder(/project name/i).fill("E2E Test Project")

    // Wait for the API response AND click simultaneously
    await Promise.all([
      page.waitForResponse(r => r.url().includes("/projects") && r.status() === 201),
      page.getByRole("button", { name: /^create/i }).click(),
    ])

    await expect(page.getByPlaceholder(/project name/i)).not.toBeVisible()
  })

})
