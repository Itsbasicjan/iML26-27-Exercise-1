"""Ensure human-readable yes/no questions preserve branch direction."""

from scripts.tree_labels import question


def test_one_hot_question_has_no_on_left_and_yes_on_right():
    text, yes_left = question("categorical__Kontostatus_A14", 0.5, ["Kontostatus"])
    assert text == "Kein Girokonto vorhanden?"
    # left branch = indicator 0 = category not present
    assert not yes_left


def test_numeric_question_keeps_yes_on_left():
    text, yes_left = question("numeric__Kreditbetrag_DM", 10841.5, [])
    assert text == "Kreditbetrag (DM) ≤ 10841,5?"
    assert yes_left


def test_unsupported_category_keeps_its_original_label():
    text, yes_left = question("categorical__Beruf_A999", 0.5, ["Beruf"])
    assert "A999" in text
    assert not yes_left
