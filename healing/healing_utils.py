"""
healing_utils.py

This module provides self-healing UI interaction utilities for Playwright-based test automation.
It is designed to help recover from broken or outdated selectors by finding similar DOM elements
based on fuzzy attribute matching.

Functions:
- similarity: Computes a similarity score between two strings.
- find_similar_element: Searches the DOM for elements that most closely match the expected attributes.
- heal_and_interact: Performs an action on a healed element (fill, click, check, select).

Usage Example:
    from healing.healing_utils import heal_and_interact

    heal_and_interact(page, "input", {"data-test": "username"}, "fill", "standard_user")
"""

from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a or "", b or "").ratio()

def find_similar_element(page, tag, expected_attrs):
    candidates = page.query_selector_all(tag)
    best_match = None
    best_score = 0

    for el in candidates:
        try:
            attrs = page.evaluate(
                "(el) => Object.fromEntries(el.getAttributeNames().map(k => [k, el.getAttribute(k)]))", el
            )
            score = sum(similarity(attrs.get(k, ""), v) for k, v in expected_attrs.items())
            if score > best_score:
                best_score = score
                best_match = el
        except:
            continue

    if not best_match:
        raise Exception("Could not self-heal: No matching element found.")
    return best_match

def heal_and_interact(page, tag, expected_attrs, action, value=None):
    el = find_similar_element(page, tag, expected_attrs)
    if action == "fill":
        el.fill(value)
    elif action == "click":
        el.click()
    elif action == "check":
        el.check()
    elif action == "select":
        el.select_option(value)
    else:
        raise ValueError(f"Unsupported action: {action}")
    print(f"✅ Healed and performed '{action}' on {tag} with {expected_attrs}")
