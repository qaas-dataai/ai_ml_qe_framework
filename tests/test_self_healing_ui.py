from playwright.sync_api import sync_playwright
import pytest
from healing.healing_utils import heal_and_interact

def test_ui_self_healing():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("https://www.saucedemo.com/")
        assert page.title() is not None, "Page title is missing"
        browser.close()


def test_login_self_healing_generic():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://www.saucedemo.com/")

        heal_and_interact(page, "input", {"data-test": "user"}, "fill", "standard_user")  # Broken on purpose
        heal_and_interact(page, "input", {"data-test": "password"}, "fill", "secret_sauce")
        heal_and_interact(page, "input", {"data-test": "login-btn"}, "click")             # Broken on purpose

        assert "inventory" in page.url
        browser.close()


def test_login_success():
    with sync_playwright() as p:
        #browser = p.chromium.launch()
        browser = p.chromium.launch(headless=False)

        page = browser.new_page()
        page.goto("https://www.saucedemo.com/")

        # Correct locators
        page.fill('input[data-test="username"]', "standard_user")
        page.fill('input[data-test="password"]', "secret_sauce")
        page.click('input[data-test="login-button"]')

        assert "inventory" in page.url, "Login failed"
        browser.close()
