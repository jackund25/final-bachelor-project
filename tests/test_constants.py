"""Ambang glikemik: satu sumber kebenaran, dua fungsi klasifikasi."""

import pytest

from src.constants import (
    CLARKE_HIGH,
    CLARKE_LOW,
    CLASS_HYPER,
    CLASS_HYPO,
    CLASS_NORMAL,
    CONDITION_CLASSES,
    GLUCOSE_CRITICAL_HIGH,
    GLUCOSE_CRITICAL_LOW,
    GLUCOSE_HIGH,
    GLUCOSE_LOW,
    RISK_CRITICAL_HYPER,
    RISK_CRITICAL_HYPO,
    RISK_HYPER,
    RISK_HYPO,
    RISK_NORMAL,
    base_condition,
    classify_glucose_3class,
    classify_glucose_5zone,
    is_critical,
    risk_from_condition_class,
    risk_label_id,
)


@pytest.mark.parametrize(
    "glucose,expected",
    [
        (40.0, RISK_CRITICAL_HYPO),
        (53.9, RISK_CRITICAL_HYPO),
        (54.0, RISK_HYPO),          # batas: 54 sudah TIDAK kritis
        (69.9, RISK_HYPO),
        (70.0, RISK_NORMAL),        # batas: 70 sudah normal
        (120.0, RISK_NORMAL),
        (180.0, RISK_NORMAL),       # batas: 180 masih normal
        (180.1, RISK_HYPER),
        (250.0, RISK_HYPER),        # batas: 250 belum kritis
        (250.1, RISK_CRITICAL_HYPER),
    ],
)
def test_classify_5zone_boundaries(glucose, expected):
    assert classify_glucose_5zone(glucose) == expected


@pytest.mark.parametrize(
    "glucose,expected",
    [
        (40.0, CLASS_HYPO),
        (69.9, CLASS_HYPO),
        (70.0, CLASS_NORMAL),
        (180.0, CLASS_NORMAL),
        (180.1, CLASS_HYPER),
        (300.0, CLASS_HYPER),
    ],
)
def test_classify_3class_boundaries(glucose, expected):
    assert classify_glucose_3class(glucose) == expected


def test_3class_is_5zone_flattened():
    """Kedua fungsi harus sepakat pada ambang 70/180 di seluruh rentang."""
    for g in range(20, 601):
        lima = base_condition(classify_glucose_5zone(float(g)))
        tiga = classify_glucose_3class(float(g))
        cocok = {
            RISK_HYPO: CLASS_HYPO,
            RISK_NORMAL: CLASS_NORMAL,
            RISK_HYPER: CLASS_HYPER,
        }[lima]
        assert cocok == tiga, f"tidak sepakat pada {g} mg/dL: 5zona={lima}, 3kelas={tiga}"


def test_condition_classes_order_is_stable():
    """Urutan kelas menentukan urutan baris confusion matrix pada laporan."""
    assert CONDITION_CLASSES == [CLASS_HYPO, CLASS_NORMAL, CLASS_HYPER]


def test_risk_labels_cover_every_zone():
    for zona in (RISK_CRITICAL_HYPO, RISK_HYPO, RISK_NORMAL, RISK_HYPER, RISK_CRITICAL_HYPER):
        assert risk_label_id(zona), f"label kosong untuk {zona}"
    assert risk_label_id(RISK_NORMAL) == "AMAN"


def test_is_critical_only_for_extremes():
    assert is_critical(RISK_CRITICAL_HYPO)
    assert is_critical(RISK_CRITICAL_HYPER)
    assert not is_critical(RISK_HYPO)
    assert not is_critical(RISK_HYPER)
    assert not is_critical(RISK_NORMAL)


def test_base_condition_strips_critical_prefix():
    assert base_condition(RISK_CRITICAL_HYPO) == RISK_HYPO
    assert base_condition(RISK_CRITICAL_HYPER) == RISK_HYPER
    assert base_condition(RISK_NORMAL) == RISK_NORMAL


def test_risk_from_condition_class_maps_indonesian_to_english():
    assert risk_from_condition_class(CLASS_HYPO) == RISK_HYPO
    assert risk_from_condition_class(CLASS_HYPER) == RISK_HYPER
    assert risk_from_condition_class(CLASS_NORMAL) == RISK_NORMAL


def test_thresholds_are_ordered():
    assert GLUCOSE_CRITICAL_LOW < GLUCOSE_LOW < GLUCOSE_HIGH < GLUCOSE_CRITICAL_HIGH


def test_clarke_thresholds_are_independent_of_risk_policy():
    """Clarke Error Grid adalah metrik terbitan dengan batas zona baku.

    Nilainya kebetulan sama dengan ambang klinis, tetapi keduanya SENGAJA dipisah:
    mengubah kebijakan risiko tidak boleh diam-diam mengubah metrik yang
    dibandingkan dengan literatur. Tes ini mendokumentasikan pemisahan itu.
    """
    assert CLARKE_LOW == 70.0
    assert CLARKE_HIGH == 180.0
