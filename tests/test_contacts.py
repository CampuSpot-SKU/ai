from __future__ import annotations

import pytest

from ai import contacts as ct

CONTACTS = [
    {
        "부서": "교무처",
        "성명": "김가나",
        "직위": "처장",
        "담당업무": "교무처 총괄",
        "내선번호": "",
        "이메일": "a@x.kr",
    },
    {
        "부서": "교무처",
        "성명": "이다라",
        "직위": "팀장",
        "담당업무": "졸업\n수료",
        "내선번호": "7071",
        "이메일": "",
    },
    {
        "부서": "홍보실",
        "성명": "박마바",
        "직위": "",
        "담당업무": "",
        "내선번호": "7660",
        "이메일": "",
    },
]
ORG = [
    {"대분류": "교학부총장", "중분류": "처", "소분류": "교무처", "세분류": "", "홈페이지": ""},
    {
        "대분류": "교학부총장",
        "중분류": "처",
        "소분류": "교무처",
        "세분류": "홍보실",
        "홈페이지": "https://x.kr/",
    },
    {
        "대분류": "교학부총장",
        "중분류": "학과",
        "소분류": "컴퓨터공학과",
        "세분류": "",
        "홈페이지": "",
    },
]


def test_snapshot_has_no_personal_data() -> None:
    snap = ct.build_snapshot(CONTACTS, ORG)
    text = str(snap)
    assert "김가나" not in text and "a@x.kr" not in text
    assert snap["departments"]["교무처"]["rows"][1]["담당업무"] == "졸업 / 수료"
    assert snap["no_contact"] == ["컴퓨터공학과"]


def test_render_phone_and_parent() -> None:
    snap = ct.build_snapshot(CONTACTS, ORG)
    md = ct.render_markdown(snap, "2026-10-05")
    assert "- 처장: 내선 미공개 — 교무처 총괄" in md
    assert "- 팀장: 02-940-7071 — 졸업 / 수료" in md
    assert "### 홍보실 (소속: 교무처)" in md
    assert "- 홈페이지: https://x.kr/" in md
    assert "- 담당: 02-940-7660" in md
    assert "02-940-7114" in md


def test_diff_detects_changes() -> None:
    old = ct.build_snapshot(CONTACTS, ORG)
    changed = [dict(r) for r in CONTACTS]
    changed[1]["내선번호"] = "7099"
    changed.append(
        {
            "부서": "신설팀",
            "성명": "x",
            "직위": "팀장",
            "담당업무": "",
            "내선번호": "7001",
            "이메일": "",
        }
    )
    new = ct.build_snapshot(changed, ORG)
    out = ct.diff_snapshots(old, new)
    assert "부서 추가: 신설팀" in out
    assert any("내선 변경: 교무처" in line and "7071" in line and "7099" in line for line in out)
    assert ct.diff_snapshots(old, old) == []


def test_check_snapshot_rejects_small_result() -> None:
    with pytest.raises(ct.ContactsError):
        ct.check_snapshot(ct.build_snapshot(CONTACTS, ORG))


def test_sites_links_and_missing_homepage() -> None:
    org = [
        {
            "대분류": "교학부총장",
            "중분류": "대학",
            "소분류": "이공대학",
            "세분류": "소프트웨어학과",
            "홈페이지": "https://cs.skuniv.ac.kr/cs_professor",
        },
        {
            "대분류": "교학부총장",
            "중분류": "대학",
            "소분류": "창의인재대학",
            "세분류": "뷰티디자인학과",
            "홈페이지": "",
        },
        {
            "대분류": "교학부총장",
            "중분류": "대학원",
            "소분류": "대학원",
            "세분류": "경영학과",
            "홈페이지": "https://grad.skuniv.ac.kr/x#y",
        },
    ]
    sites = ct.build_sites(org)
    assert sites[0]["home"] == "https://cs.skuniv.ac.kr/"
    assert sites[0]["professors"] == "https://cs.skuniv.ac.kr/cs_professor"
    assert sites[1]["home"] == ""
    assert len(sites) == 2  # 대학원은 학과 사이트 목록에 넣지 않음
    snap = {"sites": sites, "grad_ext": ["7063"]}
    md = ct.render_sites_markdown(snap, "2026-10-05")
    assert "교수진 소개 페이지 https://cs.skuniv.ac.kr/cs_professor" in md
    assert "뷰티디자인학과: 학과 홈페이지가 조직도에 등록되어 있지 않음" in md
    assert "https://grad.skuniv.ac.kr/" in md and "02-940-7063" in md
