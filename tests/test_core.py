import random
from pathlib import Path

import numpy as np
import pytest

from trhw.corpus import chunk_into_lines, good_sentence, split_for, split_sentences
from trhw.damage import erase_patches, remove_small_marks
from trhw.metrics import cer, correction_report, edit_distance, restoration_report, wer
from trhw.restore import ContextRestorer, HybridRestorer, IdentityRestorer, LexiconRestorer
from trhw.text import apply_case, clean_text, to_base, tr_lower, tr_upper

SAMPLE = Path(__file__).parent.parent / "data" / "sample" / "sample_tr.txt"
FONTS = Path(__file__).parent.parent / "fonts"


# --- text -------------------------------------------------------------------

def test_to_base_strips_all_marks_and_keeps_length():
    s = "Çiğdem Işık İzmir'de şöyle güldü"
    assert to_base(s) == "Cigdem Isik Izmir'de soyle guldu"
    assert len(to_base(s)) == len(s)


def test_turkish_casing():
    assert tr_lower("IŞIK İZMİR") == "ışık izmir"
    assert tr_upper("ışık izmir") == "IŞIK İZMİR"
    assert len(tr_lower("İİİ")) == 3  # str.lower() would give 6 characters


@pytest.mark.parametrize("char,like,expected", [("i", "I", "İ"), ("ı", "I", "I"), ("ş", "S", "Ş"), ("ş", "s", "ş")])
def test_apply_case(char, like, expected):
    assert apply_case(char, like) == expected


def test_clean_text():
    assert clean_text("  Hâlâ “güzel” —  değil mi…") == 'Hala "güzel" - değil mi...'
    assert clean_text("i\u0307stanbul") == "istanbul"


# --- metrics ----------------------------------------------------------------

def test_edit_distance_and_rates():
    assert edit_distance("kitap", "kitab") == 1
    assert cer(["abcd"], ["abcf"]) == 0.25
    assert wer(["bir iki üç"], ["bir iki uc"]) == pytest.approx(1 / 3)


def test_correction_report_counts_fixes_and_breakage():
    r = correction_report(["çocuk okula"], ["cocuk okula"], ["çocuk öküla"])
    assert r["fixed"] == 1 and r["broken"] == 2 and r["net_gain"] == -1


def test_restoration_report():
    r = restoration_report(["çocuk okula gitti"], ["cocuk okula gitti"], seen_words=set())
    assert r["ambiguous_word_acc"] == pytest.approx(2 / 3)
    assert r["unseen_word_share"] == 1.0
    with pytest.raises(ValueError):
        restoration_report(["abc"], ["ab"])


# --- restorers --------------------------------------------------------------

@pytest.fixture(scope="module")
def sentences():
    return list(split_sentences(SAMPLE.read_text(encoding="utf-8")))


def test_identity_is_a_noop():
    assert IdentityRestorer().restore("Cocuk") == "Cocuk"


def test_restorers_preserve_length(sentences):
    lex = LexiconRestorer().fit(sentences)
    ctx = ContextRestorer().fit(sentences)
    for model in (lex, ctx, HybridRestorer(lex, ctx, min_word_count=1)):
        for s in sentences:
            assert len(model.restore(to_base(s))) == len(s)


def test_lexicon_restores_known_words_with_case(sentences):
    lex = LexiconRestorer().fit(sentences)
    assert lex.restore("COCUKLAR ogretmen") == "ÇOCUKLAR öğretmen"


def test_context_model_memorizes_training_text(sentences):
    # sanity check that fit() and restore() agree: with min_count=1 the widest
    # windows are unique, so training sentences must come back (almost) perfectly
    ctx = ContextRestorer(min_count=1).fit(sentences)
    preds = ctx.restore_many([to_base(s) for s in sentences])
    floor = [to_base(s) for s in sentences]
    assert restoration_report(sentences, preds)["ambiguous_char_acc"] > 0.99
    assert restoration_report(sentences, floor)["ambiguous_char_acc"] < 0.8


def test_restorer_round_trip_save_load(tmp_path, sentences):
    ctx = ContextRestorer().fit(sentences)
    ctx.save(tmp_path / "m.pkl.gz")
    loaded = ContextRestorer.load(tmp_path / "m.pkl.gz")
    assert loaded.restore("kucuk kopek") == ctx.restore("kucuk kopek")


# --- corpus -----------------------------------------------------------------

def test_sentence_filtering_and_splits():
    assert good_sentence("Bu cümle yeterince uzun bir cümledir.")
    assert not good_sentence("Kısa.")
    assert not good_sentence("Bu cümlede 日本 karakterleri var var.")
    assert split_for("doc-42") == split_for("doc-42")  # deterministic per document


def test_chunk_into_lines_keeps_all_words():
    s = "Türkçe sondan eklemeli bir dildir ve bir kelimeye birçok ek getirilebilir."
    lines = chunk_into_lines(s, random.Random(0), 10, 25)
    assert " ".join(lines) == s
    assert len(lines) > 1


# --- images -----------------------------------------------------------------

needs_fonts = pytest.mark.skipif(not (FONTS / "manifest.json").exists(),
                                 reason="run scripts/download_fonts.py first")


@needs_fonts
def test_synthetic_sample_has_labels_and_fixed_height():
    from trhw.fonts import load_manifest
    from trhw.synth import make_sample

    rng = random.Random(0)
    img, meta = make_sample("Çiğdem ışıklı odada", load_manifest(FONTS), rng)
    assert img.dtype == np.uint8 and img.shape[0] == 64 and img.shape[1] > 100
    assert meta["base"] == "Cigdem isikli odada"
    assert img.mean() > 120  # light paper, dark ink


def test_damage_functions_keep_shape():
    img = np.full((64, 300), 240, np.uint8)
    img[20:44, 20:60] = 20      # a "letter"
    img[5:10, 35:40] = 20       # a small "dot" above it
    img[20:44, 100:140] = 20
    img[20:44, 180:220] = 20
    cleaned = remove_small_marks(img)
    assert cleaned.shape == img.shape
    assert cleaned[7, 37] > 200 and cleaned[30, 40] < 50  # dot gone, letter kept
    assert erase_patches(img, random.Random(0)).shape == img.shape
