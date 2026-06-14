import { expect, test } from "@playwright/test";
import path from "node:path";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.getByRole("heading", { name: "Cascadia Lab" })).toBeVisible();
});

test("desktop renders the playable board and advances through draft selection", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop");
  await expect(
    page.getByRole("application", { name: /Player 1 Cascadia board/ }),
  ).toBeVisible();
  await expect(page.getByRole("heading", { name: "Choose a pair" })).toBeVisible();

  const firstTile = page.getByTitle("Choose habitat tile 1");
  await expect(firstTile).toBeEnabled();
  await firstTile.click();

  await expect(
    page.getByRole("heading", { name: "Place the habitat" }),
  ).toBeVisible();
  const placement = page.getByRole("button", { name: /Place tile at/ }).first();
  await expect(placement).toBeVisible();
  await placement.click();
  await expect(
    page.getByRole("heading", { name: "Place the wildlife" }),
  ).toBeVisible();
  await page.screenshot({
    path: path.resolve(
      process.cwd(),
      "../../docs/v2/reports/web-desktop-play.png",
    ),
    fullPage: true,
  });
});

test("mobile navigation exposes board, market, scores, and analysis", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "mobile");
  await expect(page.getByRole("navigation", { name: "Mobile views" })).toBeVisible();
  await page.getByRole("button", { name: "Scores", exact: true }).click();
  await expect(page.getByLabel("Scores and scoring cards")).toBeVisible();
  await page.getByRole("button", { name: "Market", exact: true }).click();
  await expect(page.getByLabel("Turn workbench")).toBeVisible();
  await page.screenshot({
    path: path.resolve(
      process.cwd(),
      "../../docs/v2/reports/web-mobile-market.png",
    ),
    fullPage: true,
  });
});

test("research analysis exposes terminal search values", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop");
  await page.getByRole("button", { name: /Move analysis/ }).click();
  await expect(page.getByText("Confidence-gated search")).toBeVisible();
  await page.getByRole("button", { name: "Refresh" }).click();
  await expect(page.locator(".candidate-row")).toHaveCount(8);
  await expect(page.locator(".candidate-row").first()).toContainText(/\d+\.\d/);
  await page.screenshot({
    path: path.resolve(
      process.cwd(),
      "../../docs/v2/reports/web-desktop-analysis.png",
    ),
    fullPage: true,
  });
});

test("cluster dashboard reports all configured nodes and active work", async ({
  page,
}, testInfo) => {
  await page.goto("/cluster");
  await expect(page.getByText("Cascadia Compute", { exact: true })).toBeVisible();
  await expect(page.locator(".cluster-node")).toHaveCount(3, { timeout: 15_000 });
  await expect(page.getByRole("heading", { name: "John 1" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "John 2" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "John 3" })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Active workloads" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Utilization history" }),
  ).toBeVisible();
  await expect(page.locator(".history-chart")).toHaveCount(2);
  await expect(page.getByRole("img", { name: "CPU utilization over 24 hours" })).toBeVisible();
  await expect(
    page.getByRole("img", { name: "Memory utilization over 24 hours" }),
  ).toBeVisible();
  const oneDay = page.getByRole("button", { name: "1D" });
  const sevenDays = page.getByRole("button", { name: "7D" });
  await expect(oneDay).toHaveAttribute("aria-pressed", "true");
  await sevenDays.click();
  await expect(sevenDays).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("img", { name: "CPU utilization over 7 days" })).toBeVisible();
  await expect(
    page.getByRole("img", { name: "Memory utilization over 7 days" }),
  ).toBeVisible();
  await oneDay.click();
  await expect(page.getByRole("img", { name: "CPU utilization over 24 hours" })).toBeVisible();
  await page.screenshot({
    path: path.resolve(
      process.cwd(),
      testInfo.project.name === "mobile"
        ? "../../docs/v2/reports/web-cluster-dashboard-mobile.png"
        : "../../docs/v2/reports/web-cluster-dashboard.png",
    ),
    fullPage: true,
  });
});
