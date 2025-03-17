from playwright.sync_api import sync_playwright
import pytest

def test_ui_self_healing():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("https://example.com")
        assert page.title() is not None, "Page title is missing"
        browser.close()
