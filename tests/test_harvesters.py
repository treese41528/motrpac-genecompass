"""Stage 1 harvesters: which supplementary files get downloaded, and the path/URL builders.

Offline tests exercise the real geo_harvester / arrayexpress_harvester functions. The live API
checks (pipeline/01_data_harvesting/*_api_checks.py) run only with  pytest -m network.
"""
import os

import pytest

import arrayexpress_harvester as ae

SC_FILES = ["GSM123_matrix.mtx.gz", "GSM123_barcodes.tsv.gz", "GSM123_features.tsv.gz",
            "GSM123_genes.tsv.gz", "atlas.h5ad", "sample.loom"]
NEVER_FILES = ["filelist.txt", "index.html", "protocol.pdf", "README", "matrix.mtx.gz.md5"]


@pytest.fixture(params=["geo", "ae"])
def harvester(request, geo_harvester):
    return geo_harvester if request.param == "geo" else ae


def _decide(h, name, mode="single-cell", selective=True, discovery=False, archives=None):
    return h.filter_file(name, h.compile_mode_patterns(mode), h.compile_skip_patterns(),
                         selective, discovery, archives).action


# ------------------------------------------------------------------ file filter
@pytest.mark.parametrize("name", SC_FILES)
def test_single_cell_files_are_downloaded(harvester, name):
    assert _decide(harvester, name) == "download"


@pytest.mark.parametrize("name", NEVER_FILES)
def test_skip_patterns_win_in_every_mode(harvester, name):
    # skip patterns apply before discovery / non-selective modes
    assert _decide(harvester, name) == "skip_pattern"
    assert _decide(harvester, name, discovery=True) == "skip_pattern"
    assert _decide(harvester, name, selective=False) == "skip_pattern"


def test_unmatched_file_is_skipped_only_when_selective(harvester):
    name = "slides_from_talk.pptx"
    assert _decide(harvester, name) == "skip_no_match"
    assert _decide(harvester, name, selective=False) == "download"
    assert _decide(harvester, name, discovery=True) == "download"


def test_archive_listed_in_manifest_is_downloaded(harvester):
    name = "GSE1_RAW.tar"
    assert _decide(harvester, name, archives={name}) == "download"


def test_bulk_count_table_matches_bulk_mode(harvester):
    assert _decide(harvester, "GSE1_gene_counts.tsv.gz", mode="bulk") == "download"


# ------------------------------------------------------------------ GEO helpers
def test_geo_suppl_path(geo_harvester):
    assert geo_harvester.construct_suppl_path("GSE123456") == "/geo/series/GSE123nnn/GSE123456/suppl/"
    assert geo_harvester.construct_suppl_path("GSE1234") == "/geo/series/GSE1nnn/GSE1234/suppl/"


def test_geo_filelist_parsing_groups_files_under_archives(geo_harvester):
    content = "\n".join([
        "#Archive/File\tName\tTime\tSize\tType",
        "Archive\tGSE1_RAW.tar\t01/01/2024\t3000\tTAR",
        "File\tGSM1_matrix.mtx.gz\t01/01/2024\t2000\tMTX",
        "File\tGSM1_barcodes.tsv.gz\t01/01/2024\tbad\tTSV",
        "too\tshort",
    ])
    entries, archives = geo_harvester.parse_filelist_txt(content)
    assert [e.name for e in entries] == ["GSE1_RAW.tar", "GSM1_matrix.mtx.gz", "GSM1_barcodes.tsv.gz"]
    assert [e.name for e in archives["GSE1_RAW.tar"]] == ["GSM1_matrix.mtx.gz", "GSM1_barcodes.tsv.gz"]
    assert entries[1].parent_archive == "GSE1_RAW.tar"
    assert entries[2].size == 0                       # unparseable size -> 0, not a crash


# ------------------------------------------------------------------ ArrayExpress helpers
@pytest.mark.parametrize("raw, expected", [
    ("E-MTAB-5920", "E-MTAB-5920"),
    ("S-ECPF-MTAB-1994", "E-MTAB-1994"),
    ("S-ECPF-GEOD-58135", "E-GEOD-58135"),
    ("S-EPMC123456", None),                           # literature record
    ("S-BSST100", None),
    ("", None),
])
def test_arrayexpress_accession_extraction(raw, expected):
    assert ae.extract_arrayexpress_accession(raw) == expected


def test_arrayexpress_ftp_paths():
    assert ae.construct_ftp_path("E-MTAB-6798") == "/biostudies/fire/E-MTAB-/798/E-MTAB-6798/Files"
    assert ae.construct_ftp_path("E-MTAB-12") == "/biostudies/fire/E-MTAB-/012/E-MTAB-12/Files"
    assert ae.construct_legacy_ftp_path("E-GEOD-509") == "/pub/databases/microarray/data/experiment/GEOD/E-GEOD-509"


@pytest.mark.parametrize("run, path", [
    ("ERR000001", "vol1/fastq/ERR000/ERR000001"),
    ("ERR1234567", "vol1/fastq/ERR123/007/ERR1234567"),
    ("SRR0000001", "vol1/fastq/SRR000/001/SRR0000001"),
    ("SRR12345678", "vol1/fastq/SRR123/078/SRR12345678"),
])
def test_ena_fastq_url(run, path):
    assert ae.construct_ena_fastq_url(run) == f"ftp://ftp.sra.ebi.ac.uk/{path}"


def test_mage_tab_parsing():
    idf = ae.parse_idf("Investigation Title\tRat liver\t\nPerson Last Name\tA\tB\nno tabs here\n")
    assert idf == {"Investigation Title": ["Rat liver"], "Person Last Name": ["A", "B"]}
    headers, samples = ae.parse_sdrf("Source Name\tCharacteristics[organism]\nS1\tRattus norvegicus\n\nS2\tRattus norvegicus\n")
    assert headers == ["Source Name", "Characteristics[organism]"]
    assert [s["Source Name"] for s in samples] == ["S1", "S2"]
    assert ae.parse_sdrf("header only") == ([], [])


# ------------------------------------------------------------------ live APIs (pytest -m network)
@pytest.mark.network
def test_geo_ftp_reachable():
    import geo_api_checks as g
    assert g.check_ftp_connection().success


@pytest.mark.network
def test_geo_entrez_reachable():
    email = os.environ.get("NCBI_EMAIL")
    if not email:
        pytest.skip("set NCBI_EMAIL to run the Entrez check")
    import geo_api_checks as g
    assert g.check_entrez_connection(email).success


@pytest.mark.network
@pytest.mark.parametrize("check", ["check_network_connectivity", "check_direct_api_json",
                                   "check_ena_portal_api", "check_ftp_connection"])
def test_arrayexpress_services_reachable(check):
    import arrayexpress_api_checks as a
    result = getattr(a, check)()
    assert result.success, result.message


@pytest.mark.network
def test_arrayexpress_study_metadata():
    import arrayexpress_api_checks as a
    result = a.check_study_metadata("E-MTAB-5920")
    assert result.success, result.message
