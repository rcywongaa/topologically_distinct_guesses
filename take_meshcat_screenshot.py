from playwright.sync_api import sync_playwright


def take_screenshot(url, filename):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        # opens a new browser page
        page = browser.new_page()
        # navigate to the website
        page.goto(url)
        # take a full-page screenshot
        page.screenshot(path=filename, full_page=True)
        # always close the browser
        browser.close()
