"""Erzeugt die Screenshots für die README (braucht: pip install playwright, Google Chrome, laufende App).

    streamlit run app.py            # in einem Terminal
    python tools/make_screenshots.py --dataset <name>   # im anderen

Es wird ein kurzes Training (300 Schritte) gestartet, damit das Dashboard gefüllt ist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"


def settle(page: Page, ms: int = 2500) -> None:
    page.wait_for_selector('[data-testid="stApp"]')
    page.wait_for_timeout(800)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=30000)
    except Exception:
        pass
    page.wait_for_timeout(ms)


def shot(page: Page, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
    print("gespeichert:", name)


def goto(page: Page, base: str, path: str) -> None:
    page.goto(f"{base}/{path}")
    settle(page)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 2300})

        for slug, name in [("", "01-start"), ("daten", "02-daten"), ("tokenizer", "03-tokenizer"),
                           ("modell", "04-modell")]:
            goto(page, args.url, slug)
            shot(page, name)

        # --- kurzes Training starten
        goto(page, args.url, "training")
        slider = page.get_by_role("slider", name="Trainingsschritte")
        if not slider.is_visible():      # nach einem früheren Lauf ist der Einstellungs-Bereich eingeklappt
            page.get_by_text("Trainings-Einstellungen").click()
            page.wait_for_timeout(800)
        slider.focus()
        page.keyboard.press("Home")
        for _ in range(args.steps // 100 - 1):
            page.keyboard.press("ArrowRight")
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="Training starten").click()
        page.get_by_text("läuft", exact=False).first.wait_for(timeout=60_000)
        page.wait_for_function("() => !document.body.innerText.includes('läuft')", timeout=600_000)
        assert page.get_by_text("300 / 300").count() >= 1, "Training wurde nicht bis zum Ende ausgeführt"
        page.wait_for_timeout(3000)
        page.get_by_text("Trainings-Einstellungen").click()                             # einklappen
        page.wait_for_timeout(800)
        views = [("Kurven", "05-training-kurven"), ("Beispieltexte", "06-training-beispiele"),
                 ("Aufmerksamkeit", "07-training-aufmerksamkeit"), ("Embeddings", "08-training-embeddings"),
                 ("Gewichte", "09-training-gewichte")]
        for label, name in views:
            page.get_by_text(label, exact=False).first.click()
            page.wait_for_timeout(2500)
            shot(page, name)

        goto(page, args.url, "generieren")
        page.get_by_role("button", name="Erzeugen").click()
        page.wait_for_timeout(9000)
        shot(page, "10-generieren")
        goto(page, args.url, "laeufe")
        shot(page, "11-laeufe")
        browser.close()


if __name__ == "__main__":
    main()
