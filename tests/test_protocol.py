"""Protocol parsing: the seen/unseen split derives from these files, so a
parsing bug would silently invalidate the entire experiment."""
from pathlib import Path

from src.data import parse_protocol, SEEN_ATTACKS, UNSEEN_ATTACKS

FIXTURE = """\
LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1271820 - A01 spoof
LA_0080 LA_T_9999999 - A06 spoof
LA_0081 LA_E_1234567 - A17 spoof
"""


def test_parse_protocol(tmp_path):
    f = tmp_path / "proto.txt"
    f.write_text(FIXTURE)
    items = parse_protocol(f, tmp_path / "flac")

    assert len(items) == 4
    assert items[0].label == 1 and items[0].attack == "-"          # bona fide
    assert items[1].label == 0 and items[1].attack == "A01"
    assert items[1].path.endswith("LA_T_1271820.flac")
    assert items[2].attack in SEEN_ATTACKS
    assert items[3].attack in UNSEEN_ATTACKS


def test_attack_sets_disjoint_and_complete():
    assert SEEN_ATTACKS.isdisjoint(UNSEEN_ATTACKS)
    assert len(SEEN_ATTACKS) == 6 and len(UNSEEN_ATTACKS) == 13


def test_parse_skips_blank_lines(tmp_path):
    f = tmp_path / "proto.txt"
    f.write_text(FIXTURE + "\n\n")
    assert len(parse_protocol(f, tmp_path)) == 4
