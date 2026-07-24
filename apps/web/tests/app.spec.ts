import { expect, test, type TestInfo } from "@playwright/test";
import path from "node:path";

function screenshotPath(testInfo: TestInfo, name: string): string {
  const reportDirectory = process.env.CASCADIA_VISUAL_REPORT_DIR;
  if (reportDirectory) {
    return path.resolve(process.cwd(), reportDirectory, name);
  }
  return testInfo.outputPath(name);
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.getByRole("heading", { name: "CASCADIA" })).toBeVisible();
});

test("desktop renders the playable board and advances through draft selection", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop");
  await expect(
    page.getByRole("application", { name: /Player 1 Cascadia board/ }),
  ).toBeVisible();
  await expect(page.locator(".status-bar")).toContainText(
    "click a market pair",
  );

  const firstTile = page.getByTitle("Draft this pair").first();
  await expect(firstTile).toBeEnabled();
  await firstTile.click();

  await expect(page.locator(".status-bar")).toContainText(
    "highlighted frontier hex",
  );
  const placement = page.getByRole("button", { name: /Place tile at/ }).first();
  await expect(placement).toBeVisible();
  await placement.click();
  await expect(page.locator(".status-bar")).toContainText(
    "place the wildlife token",
  );
  await page.screenshot({
    path: screenshotPath(testInfo, "web-desktop-play.png"),
    fullPage: true,
  });
});

test("mobile navigation exposes board, market, scores, and analysis", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "mobile");
  await expect(page.getByRole("navigation", { name: "Mobile views" })).toBeVisible();
  await page.getByRole("button", { name: "Scores", exact: true }).click();
  await expect(page.getByRole("button", { name: "Scores", exact: true })).toHaveClass(
    /is-active/,
  );
  await expect(page.getByLabel("Scores and scoring cards")).toBeVisible();
  await page.getByRole("button", { name: "Market", exact: true }).click();
  await expect(page.getByRole("button", { name: "Market", exact: true })).toHaveClass(
    /is-active/,
  );
  await expect(page.getByRole("heading", { name: "Market" })).toBeVisible();
  await page.screenshot({
    path: screenshotPath(testInfo, "web-mobile-market.png"),
    fullPage: true,
  });
});

test("research analysis exposes terminal search values", async ({
  page,
}, testInfo) => {
  test.skip(testInfo.project.name !== "desktop");
  await page.locator(".strength-select").selectOption("instant");
  await page.getByRole("button", { name: "Suggest", exact: true }).click();
  await expect(page.locator(".candidate-row")).toHaveCount(8, { timeout: 30_000 });
  await expect(page.locator(".candidate-row").first()).toContainText(/\d+\.\d/);
  await page.screenshot({
    path: screenshotPath(testInfo, "web-desktop-analysis.png"),
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
    path: screenshotPath(
      testInfo,
      testInfo.project.name === "mobile"
        ? "web-cluster-dashboard-mobile.png"
        : "web-cluster-dashboard.png",
    ),
    fullPage: true,
  });
});

test("wildlife atlas browses and explains catalog boards", async ({
  page,
}, testInfo) => {
  await page.goto("/wildlife-catalog?rules=AAAAA");
  await expect(
    page.getByRole("heading", { name: "Cascadia Wildlife Atlas" }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: /AAAAA best-known wildlife board with 20 animals/ }),
  ).toBeVisible();
  await expect(page.locator(".atlas-score-number strong")).toHaveText("68");
  await expect(page.getByText("Optimality proven")).toBeVisible();

  await page.getByRole("button", { name: "Hawk card C" }).click();
  await expect(page).toHaveURL(/rules=AAACA/);
  await expect(page.locator(".atlas-score-number strong")).toHaveText("76");
  await expect(page.getByText("Validated incumbent")).toBeVisible();

  await page.locator(".atlas-breakdown button").filter({ hasText: "Fox" }).click();
  await expect(page.locator(".atlas-token.is-muted")).not.toHaveCount(0);

  await page.screenshot({
    path: screenshotPath(
      testInfo,
      testInfo.project.name === "mobile"
        ? "web-wildlife-atlas-mobile.png"
        : "web-wildlife-atlas.png",
    ),
    fullPage: true,
  });
});
