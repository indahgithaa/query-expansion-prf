from scripts.run_stage10 import table_like


def test_table_like_detects_markdown_table():
    assert table_like("| Kolom | Nilai |\n|---|---|\n| A | 1 |")


def test_table_like_rejects_plain_clinical_prose():
    assert not table_like("Status gizi pasien dijelaskan dalam paragraf biasa.")
