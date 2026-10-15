from scripts.prepare_paraphrases import PARAPHRASES, validate
from src.data import QUESTION_TYPES
from src.scenarios import deterministic_paraphrase, scenario_collection


def test_all_paraphrases_are_valid_and_disjoint():
    templates = {value: key for key, value in QUESTION_TYPES.items()}
    assert validate(PARAPHRASES, templates)["status"] == "PASS"


def test_paraphrase_assignment_is_deterministic():
    variants = PARAPHRASES["tujuan_diet"]["dev"]
    assert deterministic_paraphrase("Q0001", 7, variants) == deterministic_paraphrase("Q0001", 7, variants)


def test_new_scenarios_exclude_same_type():
    rows = [{"qid": "a", "question_type": "pes"}, {"qid": "b", "question_type": "biokimia"}]
    assert scenario_collection(rows, "pes", "NEW") == [rows[1]]
    assert scenario_collection(rows, "pes", "PNEW") == [rows[1]]
    assert scenario_collection(rows, "pes", "TPL") == rows
