"""The sample key: mix<i> names to MoTrPAC vial labels (docs/audit/remediation/STEP3_PREREG.md section 2.5).

Every stage after the deconvolution joins a pseudo-cell to its animal by the position of its sample in
bulk_samples.tsv. The helper must return that list unchanged, refuse a list that disagrees with what Stage 8
wrote, and refuse a sample name it cannot place (which used to drop the pseudo-cell silently).
"""
import numpy as np
import pytest

from sample_join import SampleJoinError, load_viallabels, mix_index

VIALS = ["90217016803", "90218016803", "90222016803", "90229016803"]


def _tree(tmp_path, vials=VIALS, key=None, fraction_rows=None, tissue="LIVER"):
    """A bulk root and a results root for one tissue; key = rows of mixture_samples.tsv (with header),
    fraction_rows = number of sample rows in estimated_fractions.csv."""
    bulk, res = tmp_path / "bulk", tmp_path / "results" / "motrpac"
    (bulk / tissue).mkdir(parents=True)
    (bulk / tissue / "bulk_samples.tsv").write_text("\n".join(vials) + "\n")
    if key is not None or fraction_rows is not None:
        (res / tissue).mkdir(parents=True)
    if key is not None:
        (res / tissue / "mixture_samples.tsv").write_text("\n".join("\t".join(r) for r in key) + "\n")
    if fraction_rows is not None:
        lines = ['"","Hepatocytes","Kupffer cells"'] + [f'"mix{i + 1}",0.9,0.1' for i in range(fraction_rows)]
        (res / tissue / "estimated_fractions.csv").write_text("\n".join(lines) + "\n")
    return bulk, res


def _key(vials):
    return [("mix", "viallabel")] + [(f"mix{i + 1}", v) for i, v in enumerate(vials)]


def test_vial_labels_come_back_in_file_order_without_a_results_tree(tmp_path):
    bulk, res = _tree(tmp_path)
    assert load_viallabels(bulk, "liver") == VIALS                      # the folder is the upper-case tissue
    assert load_viallabels(bulk, "LIVER", res) == VIALS                 # no Stage 8 output for it: nothing to check


def test_a_matching_key_file_passes(tmp_path):
    bulk, res = _tree(tmp_path, key=_key(VIALS))
    assert load_viallabels(bulk, "liver", res) == VIALS


def test_a_key_file_in_another_order_is_an_error(tmp_path):
    swapped = [VIALS[1], VIALS[0]] + VIALS[2:]
    bulk, res = _tree(tmp_path, key=_key(swapped))
    with pytest.raises(SampleJoinError, match="first difference at row 1"):
        load_viallabels(bulk, "liver", res)


def test_a_key_file_with_another_length_or_header_is_an_error(tmp_path):
    bulk, res = _tree(tmp_path, key=_key(VIALS[:3]))
    with pytest.raises(SampleJoinError, match="3 rows against 4"):
        load_viallabels(bulk, "liver", res)
    (res / "LIVER" / "mixture_samples.tsv").write_text("sample\tvial\nmix1\t" + VIALS[0] + "\n")
    with pytest.raises(SampleJoinError, match="header"):
        load_viallabels(bulk, "liver", res)


def test_without_a_key_file_the_fraction_rows_are_counted(tmp_path):
    bulk, res = _tree(tmp_path, fraction_rows=4)
    assert load_viallabels(bulk, "liver", res) == VIALS
    (res / "LIVER" / "estimated_fractions.csv").write_text('"","Hepatocytes"\n"mix1",1\n"mix2",1\n"mix3",1\n')
    with pytest.raises(SampleJoinError, match="3 samples .* 4 vial"):
        load_viallabels(bulk, "liver", res)


def test_the_key_file_is_the_check_when_both_files_exist(tmp_path):
    # registered order of the checks: the key file if it exists, else the row count of the fractions
    bulk, res = _tree(tmp_path, key=_key(VIALS), fraction_rows=3)
    assert load_viallabels(bulk, "liver", res) == VIALS


def test_an_empty_sample_list_is_an_error(tmp_path):
    bulk, _ = _tree(tmp_path, vials=[])
    with pytest.raises(SampleJoinError):
        load_viallabels(bulk, "liver")


def test_mix_names_map_to_positions():
    idx = mix_index(np.array(["mix1", "mix4", "mix2", "mix4"]), 4)
    assert idx.tolist() == [0, 3, 1, 3]
    assert np.array(VIALS)[idx].tolist() == [VIALS[0], VIALS[3], VIALS[1], VIALS[3]]


@pytest.mark.parametrize("name", ["mix0", "mix5", "mix51", "sample3", "3", "mix", ""])
def test_a_sample_outside_the_list_is_an_error_not_a_dropped_cell(name):
    with pytest.raises(SampleJoinError, match="no place among the 4 vial labels"):
        mix_index(["mix1", name], 4, context="liver")
