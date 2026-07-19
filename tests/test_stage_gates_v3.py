from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ecgcert import lineage, security_scan
from ecgcert.paper_evidence import FIGURE_ARTIFACTS, direct_artifact_hashes
from ecgcert.stage_gates import (
    json_artifact_bytes,
    json_artifact_sha256,
    make_review,
    merge_review,
    validate_review,
    validate_reviewed_gate,
)
from experiments.stage_gates_v3 import (
    SECURITY_FLAGS,
    _stage15_reviewed_paper_decision,
    build_stage5_gate,
    build_stage9_gate,
    build_stage20_gate,
    static_paper_check,
)
from scripts.claim_sync_v3 import _review_is_valid
from scripts.wait_for_stage_review import wait_for_review


CREATED = datetime(2026, 7, 19, 8, 0, tzinfo=timezone.utc)


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_artifact_bytes(value))
    return path


def _remote_legacy_archive(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema_version": "ecgcert-remote-legacy-archive-v1",
            "created_at": "2026-07-19T17:45:55+08:00",
            "source": {
                "repository_path": "/root/autodl-tmp/ecg-recoverability",
                "head": "2" * 40,
                "working_tree_preserved_in_place": True,
                "reset_or_overwrite_performed": False,
                "status_entries": 24,
                "result_files": 37,
            },
            "archive": {
                "path": "/root/autodl-fs/ecg-legacy-archive/frozen",
                "read_only": True,
                "data_directory_excluded": True,
                "source_and_results_tar_sha256": "a" * 64,
                "archive_manifest_sha256": "b" * 64,
                "results_manifest_sha256": "c" * 64,
                "status_sha256": "d" * 64,
            },
        },
    )


def _repository_secret_scan(path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]

    def git_value(expression: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", expression],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=True,
        ).stdout.strip()

    findings: list[dict] = []
    return _write_json(
        path,
        {
            "schema_version": "ecgcert-repository-secret-scan-v1",
            "status": "complete",
            "scanned_at": "2026-07-19T08:00:00+00:00",
            "repository": {
                "commit": git_value("HEAD"),
                "tree": git_value("HEAD^{tree}"),
                "clean": True,
            },
            "scanner": {
                "version": "1",
                "module": "src/ecgcert/security_scan.py",
                "sha256": lineage.artifact_sha256(Path(security_scan.__file__)),
                "patterns_sha256": "e" * 64,
            },
            "scope": {
                "tracked": True,
                "untracked": True,
                "ignored": True,
                "symlinks_followed": False,
                "excluded_prefixes": [],
                "max_text_bytes": 5 * 1024 * 1024,
                "inventory_sha256": "f" * 64,
                "inventory_entries": 1,
                "scanned_files": 1,
                "excluded_files": 0,
                "skipped_binary": 0,
                "skipped_large": 0,
                "skipped_symlink": 0,
            },
            "findings_count": 0,
            "findings_sha256": lineage.canonical_sha256(findings),
            "findings": findings,
        },
    )


def _registry(
    path: Path,
    *,
    status: str = "verified_primary",
    numeric_status: str = "verified_artifact",
    stage15_sha256: str | None = None,
    stage15_status: str | None = None,
    values: dict[str, str] | None = None,
    claim_macros_sha256: str | None = None,
    figures_summary_sha256: str | None = None,
    figure_artifacts_sha256: dict[str, str] | None = None,
) -> Path:
    values = values or {
        "ResultHeadline": "headline",
        "ResultOne": "effect",
        "ResultConclusion": "conclusion",
    }
    numeric = {}
    for key, value in values.items():
        entry = {"status": numeric_status, "artifact": "artifacts/value.json"}
        if stage15_sha256 is not None:
            entry.update({
                "stage15_sha256": stage15_sha256,
                "stage15_status": stage15_status,
                "value_sha256": lineage.canonical_sha256(value),
            })
        if claim_macros_sha256 is not None:
            entry["claim_macros_sha256"] = claim_macros_sha256
        numeric[key] = entry
    registry = {
        "schema_version": "verified-registry-v1",
        "citations": {
            "source": {"status": status, "source": "https://example.test/paper"}
        },
        "required_literature_coverage": {
            topic: {
                "citation_key": "source",
                "status": status,
                "source": "https://example.test/paper",
            }
            for topic in (
                "full_configuration_benchmark",
                "imputeecg",
                "ecgrecover",
            )
        },
        "numeric_claims": numeric,
    }
    if stage15_sha256 is not None:
        registry.update({
            "stage15_sha256": stage15_sha256,
            "stage15_status": stage15_status,
            "claim_values_sha256": lineage.canonical_sha256(values),
        })
    if claim_macros_sha256 is not None:
        registry["claim_macros_sha256"] = claim_macros_sha256
    if figures_summary_sha256 is not None:
        registry["figures_summary_sha256"] = figures_summary_sha256
    if figure_artifacts_sha256 is not None:
        registry["figure_artifacts_sha256"] = figure_artifacts_sha256
    return _write_json(path, registry)


def _manuscript(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join((
            r"\input{auto/robust_map_placeholders}",
            r"\begin{abstract}\ResultHeadline\end{abstract}",
            r"Claim \cite{source}; \ResultOne.",
            r"\section{Conclusion}\ResultConclusion\clearpage",
        )),
        encoding="utf-8",
    )
    return path


def test_stage5_registers_pending_but_blocks_proceed(tmp_path: Path, reviewer_keys) -> None:
    manuscript = _manuscript(tmp_path / "paper.tex")
    pending_registry = _registry(
        tmp_path / "pending.json", status="registered_pending_stage5"
    )
    gate = build_stage5_gate(
        manuscript=manuscript, registry=pending_registry, created_at=CREATED
    )
    assert gate["status"] == "PENDING_USER_REVIEW"
    assert gate["evidence"]["registry_check"]["all_registered"] is True
    assert gate["eligible_for_proceed"] is False
    with pytest.raises(ValueError, match="do not permit PROCEED"):
        make_review(
            gate,
            gate_sha256=json_artifact_sha256(gate),
            reviewer="author",
            decision="PROCEED",
            private_key_path=reviewer_keys.private,
            public_key_path=reviewer_keys.public,
            reviewed_at=CREATED + timedelta(hours=1),
        )

    verified = build_stage5_gate(
        manuscript=manuscript,
        registry=_registry(tmp_path / "verified.json"),
        created_at=CREATED,
    )
    assert verified["eligible_for_proceed"] is True
    review = make_review(
        verified,
        gate_sha256=json_artifact_sha256(verified),
        reviewer="author",
        decision="PROCEED",
        private_key_path=reviewer_keys.private,
        public_key_path=reviewer_keys.public,
        reviewed_at=CREATED + timedelta(hours=1),
    )
    validate_review(
        review,
        verified,
        gate_sha256=json_artifact_sha256(verified),
        public_key_path=reviewer_keys.public,
    )


def test_stage9_freezes_all_four_inputs_and_requires_secure_status(tmp_path: Path) -> None:
    for name in ("protocol.md", "manifest.yaml", "arc.yaml"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    security = {
        "schema_version": "ecgcert-security-status-v2",
        **{field: True for field in SECURITY_FLAGS},
        "known_hosts_sha256": "c" * 64,
        "public_key_sha256": "d" * 64,
        "host_key_fingerprint": "SHA256:" + "A" * 43,
        "host_key_verification_method": "strict-known-hosts-key-only-probe",
        "risk_acceptance": {
            "decision": "ACCEPT_UNROTATED_PASSWORD_RISK",
            "accepted_at": "2026-07-19T07:00:00+00:00",
            "accepted_by": "author",
            "rationale": "explicit residual-risk acceptance",
        },
        "verified_at": "2026-07-19T07:00:00+00:00",
        "verified_by": "author",
    }
    security_path = _write_json(tmp_path / "security.json", security)
    ecgrecover_integration = _write_json(
        tmp_path / "ecgrecover-integration.json",
        {
            "schema_version": "ecgrecover-integration-v3",
            "upstream_commit": "ed49dddf8e5e599b8af702e871a1f66b1d628518",
            "license_spdx": "NOASSERTION",
            "redistribution": False,
            "permission_basis": "author_permission_reported_by_project_owner",
            "adaptation_disclosure": ["U-Net", "loss", "fixed scaling", "single-input"],
        },
    )
    ecgrecover_upstream = _write_json(
        tmp_path / "ecgrecover-upstream.json",
        {
            "schema_version": "upstream-pin-v1",
            "commit": "ed49dddf8e5e599b8af702e871a1f66b1d628518",
            "license": {
                "spdx": "NOASSERTION",
                "redistribution_by_this_repository": False,
                "permission_basis": "author_permission_reported_by_project_owner",
            },
        },
    )
    ecgrecover_permission = _write_json(
        tmp_path / "ecgrecover-permission.json",
        {
            "schema_version": "ecgrecover-permission-attestation-v1",
            "upstream_commit": "ed49dddf8e5e599b8af702e871a1f66b1d628518",
            "repository_license_status_acknowledged": "NOASSERTION",
            "source_or_weight_redistribution_by_this_repository": False,
            "attested_by": "project_owner",
            "author_permission_received": True,
            "external_permission_record_review_required": True,
        },
    )
    ecgrecover_inputs = {
        "repository_secret_scan": _repository_secret_scan(
            tmp_path / "repository-secret-scan.json"
        ),
        "ecgrecover_integration": ecgrecover_integration,
        "ecgrecover_upstream": ecgrecover_upstream,
        "ecgrecover_permission": ecgrecover_permission,
        "remote_legacy_archive": _remote_legacy_archive(
            tmp_path / "remote-legacy-archive.json"
        ),
    }
    gate = build_stage9_gate(
        protocol=tmp_path / "protocol.md",
        experiment_manifest=tmp_path / "manifest.yaml",
        arc_config=tmp_path / "arc.yaml",
        security_status=security_path,
        **ecgrecover_inputs,
        created_at=CREATED,
    )
    assert gate["eligible_for_proceed"] is True
    hashes = gate["evidence"]["input_sha256"]
    assert set(hashes) == {
        "research_protocol",
        "experiment_manifest",
        "arc_config",
        "security_status",
        "repository_secret_scan",
        "ecgrecover_integration",
        "ecgrecover_upstream",
        "ecgrecover_permission",
        "remote_legacy_archive",
    }
    assert gate["evidence"]["ecgrecover"]["all_controls_satisfied"] is True
    assert all(len(value) == 64 for value in hashes.values())

    security["exposed_password_rotated"] = False
    accepted_path = _write_json(tmp_path / "accepted-risk.json", security)
    accepted = build_stage9_gate(
        protocol=tmp_path / "protocol.md",
        experiment_manifest=tmp_path / "manifest.yaml",
        arc_config=tmp_path / "arc.yaml",
        security_status=accepted_path,
        **ecgrecover_inputs,
        created_at=CREATED,
    )
    assert accepted["eligible_for_proceed"] is True
    assert accepted["evidence"]["security"]["residual_risks"]

    failed_scan_value = json.loads(
        ecgrecover_inputs["repository_secret_scan"].read_text(encoding="utf-8")
    )
    failed_scan_value["status"] = "failed"
    failed_scan_value["findings_count"] = 1
    failed_scan_path = _write_json(tmp_path / "failed-secret-scan.json", failed_scan_value)
    failed_scan_inputs = dict(ecgrecover_inputs)
    failed_scan_inputs["repository_secret_scan"] = failed_scan_path
    failed_scan_gate = build_stage9_gate(
        protocol=tmp_path / "protocol.md",
        experiment_manifest=tmp_path / "manifest.yaml",
        arc_config=tmp_path / "arc.yaml",
        security_status=accepted_path,
        **failed_scan_inputs,
        created_at=CREATED,
    )
    assert failed_scan_gate["eligible_for_proceed"] is False
    assert not failed_scan_gate["evidence"]["repository_secret_scan"][
        "all_controls_satisfied"
    ]

    security["exposed_password_risk_accepted"] = False
    insecure_path = _write_json(tmp_path / "insecure.json", security)
    insecure = build_stage9_gate(
        protocol=tmp_path / "protocol.md",
        experiment_manifest=tmp_path / "manifest.yaml",
        arc_config=tmp_path / "arc.yaml",
        security_status=insecure_path,
        **ecgrecover_inputs,
        created_at=CREATED,
    )
    assert insecure["eligible_for_proceed"] is False
    assert not insecure["evidence"]["security"]["all_controls_satisfied"]

    denied_permission = json.loads(ecgrecover_permission.read_text(encoding="utf-8"))
    denied_permission["author_permission_received"] = False
    denied_permission_path = _write_json(
        tmp_path / "ecgrecover-permission-denied.json", denied_permission
    )
    denied = build_stage9_gate(
        protocol=tmp_path / "protocol.md",
        experiment_manifest=tmp_path / "manifest.yaml",
        arc_config=tmp_path / "arc.yaml",
        security_status=accepted_path,
        repository_secret_scan=ecgrecover_inputs["repository_secret_scan"],
        ecgrecover_integration=ecgrecover_integration,
        ecgrecover_upstream=ecgrecover_upstream,
        ecgrecover_permission=denied_permission_path,
        remote_legacy_archive=ecgrecover_inputs["remote_legacy_archive"],
        created_at=CREATED,
    )
    assert denied["eligible_for_proceed"] is False
    assert not denied["evidence"]["ecgrecover"]["all_controls_satisfied"]

    invalid_archive = json.loads(
        ecgrecover_inputs["remote_legacy_archive"].read_text(encoding="utf-8")
    )
    invalid_archive["archive"]["read_only"] = False
    invalid_archive_path = _write_json(tmp_path / "mutable-archive.json", invalid_archive)
    mutable_archive_inputs = dict(ecgrecover_inputs)
    mutable_archive_inputs["remote_legacy_archive"] = invalid_archive_path
    mutable_archive_gate = build_stage9_gate(
        protocol=tmp_path / "protocol.md",
        experiment_manifest=tmp_path / "manifest.yaml",
        arc_config=tmp_path / "arc.yaml",
        security_status=accepted_path,
        **mutable_archive_inputs,
        created_at=CREATED,
    )
    assert mutable_archive_gate["eligible_for_proceed"] is False
    assert not mutable_archive_gate["evidence"]["remote_legacy_archive"][
        "all_controls_satisfied"
    ]


def test_generic_review_wait_is_hash_bound_and_24_hour_fail_closed(
    tmp_path: Path, reviewer_keys
) -> None:
    manuscript = _manuscript(tmp_path / "paper.tex")
    gate = build_stage5_gate(
        manuscript=manuscript,
        registry=_registry(tmp_path / "registry.json"),
        # Use a live timestamp so the wait helper's wall-clock deadline is open.
        created_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    gate_path = _write_json(tmp_path / "gate.json", gate)
    gate_sha = lineage.artifact_sha256(gate_path)
    review = make_review(
        gate,
        gate_sha256=gate_sha,
        reviewer="author",
        decision="PROCEED",
        private_key_path=reviewer_keys.private,
        public_key_path=reviewer_keys.public,
        reviewed_at=datetime.now(timezone.utc),
    )
    approval = _write_json(tmp_path / "approval.json", review)
    combined, loaded = wait_for_review(
        gate_path=gate_path,
        approval_path=approval,
        timeout_hours=1,
        poll_seconds=0.5,
        public_key_path=reviewer_keys.public,
    )
    assert loaded == review
    assert combined["status"] == "PROCEED"
    assert combined["approval_sha256"] == lineage.artifact_sha256(approval)
    assert combined["review_signature_ed25519"] == review["review_signature_ed25519"]
    assert combined["reviewer_public_key_sha256"] == review[
        "reviewer_public_key_sha256"
    ]
    assert combined["review_gate_sha256"] == gate_sha
    validate_reviewed_gate(
        combined,
        require_proceed=True,
        public_key_path=reviewer_keys.public,
    )

    tampered = dict(review)
    tampered["evidence_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="evidence|signature"):
        validate_review(
            tampered,
            gate,
            gate_sha256=gate_sha,
            public_key_path=reviewer_keys.public,
        )
    with pytest.raises(TimeoutError, match="deadline"):
        make_review(
            gate,
            gate_sha256=gate_sha,
            reviewer="late",
            decision="PROCEED",
            private_key_path=reviewer_keys.private,
            public_key_path=reviewer_keys.public,
            reviewed_at=(
                datetime.fromisoformat(gate["created_at"]) + timedelta(hours=25)
            ),
        )

    expired_gate = build_stage5_gate(
        manuscript=manuscript,
        registry=_registry(tmp_path / "expired-registry.json"),
        created_at=datetime.now(timezone.utc) - timedelta(hours=25),
    )
    expired_path = _write_json(tmp_path / "expired-gate.json", expired_gate)
    with pytest.raises(TimeoutError, match="within 24 hours"):
        wait_for_review(
            gate_path=expired_path,
            approval_path=tmp_path / "never-created-approval.json",
            timeout_hours=24,
            poll_seconds=0.5,
            public_key_path=reviewer_keys.public,
        )


def test_ed25519_review_rejects_forgery_tampering_and_wrong_key(
    tmp_path: Path, reviewer_keys, wrong_reviewer_keys
) -> None:
    gate = build_stage5_gate(
        manuscript=_manuscript(tmp_path / "paper.tex"),
        registry=_registry(tmp_path / "registry.json"),
        created_at=CREATED,
    )
    gate_sha = json_artifact_sha256(gate)
    review = make_review(
        gate,
        gate_sha256=gate_sha,
        reviewer="author",
        decision="PROCEED",
        private_key_path=reviewer_keys.private,
        public_key_path=reviewer_keys.public,
        reviewed_at=CREATED + timedelta(hours=1),
    )

    tampered = dict(review)
    tampered["decision"] = "REFINE"
    with pytest.raises(ValueError, match="signature"):
        validate_review(
            tampered,
            gate,
            gate_sha256=gate_sha,
            public_key_path=reviewer_keys.public,
        )

    forged = dict(review)
    forged["review_signature_ed25519"] = "A" * 86 + "=="
    with pytest.raises(ValueError, match="signature"):
        validate_review(
            forged,
            gate,
            gate_sha256=gate_sha,
            public_key_path=reviewer_keys.public,
        )

    with pytest.raises(ValueError, match="different reviewer public key"):
        validate_review(
            review,
            gate,
            gate_sha256=gate_sha,
            public_key_path=wrong_reviewer_keys.public,
        )
    with pytest.raises(ValueError, match="does not match the pinned public key"):
        make_review(
            gate,
            gate_sha256=gate_sha,
            reviewer="author",
            decision="PROCEED",
            private_key_path=wrong_reviewer_keys.private,
            public_key_path=reviewer_keys.public,
            reviewed_at=CREATED + timedelta(hours=1),
        )

    approval = _write_json(tmp_path / "approval.json", review)
    combined = merge_review(
        gate,
        review,
        gate_sha256=gate_sha,
        approval_sha256=lineage.artifact_sha256(approval),
        public_key_path=reviewer_keys.public,
    )
    combined["reviewed_by"] = "attacker"
    with pytest.raises(ValueError, match="approval_sha256|signature"):
        validate_reviewed_gate(combined, public_key_path=reviewer_keys.public)


def test_review_private_key_cannot_be_loaded_from_repository(
    tmp_path: Path, reviewer_keys
) -> None:
    gate = build_stage5_gate(
        manuscript=_manuscript(tmp_path / "paper.tex"),
        registry=_registry(tmp_path / "registry.json"),
        created_at=CREATED,
    )
    repository_file = Path(__file__).resolve()
    with pytest.raises(ValueError, match="outside the repository"):
        make_review(
            gate,
            gate_sha256=json_artifact_sha256(gate),
            reviewer="author",
            decision="PROCEED",
            private_key_path=repository_file,
            public_key_path=reviewer_keys.public,
            reviewed_at=CREATED + timedelta(hours=1),
        )


def _reviewed_stage15(tmp_path: Path, reviewer_keys, status: str = "PROCEED") -> Path:
    evidence = {"decision_fixture": status}
    gate = {
        "schema_version": "arc-stage15-v3",
        "stage": 15,
        "status": "PENDING_USER_REVIEW",
        "eligible_for_proceed": status == "PROCEED",
        "human_review_required": True,
        "review_deadline_hours": 24,
        "created_at": CREATED.isoformat(timespec="seconds"),
        "automatic_decision": status,
        "automatic_reasons": [] if status == "PROCEED" else ["hard criterion failed"],
        "meta_analysis_sha256": "a" * 64,
        "evidence_sha256": lineage.canonical_sha256(evidence),
        "evidence": evidence,
    }
    automatic_path = _write_json(tmp_path / "stage15-automatic.json", gate)
    gate_sha = lineage.artifact_sha256(automatic_path)
    review = make_review(
        gate,
        gate_sha256=gate_sha,
        reviewer="author",
        decision=status,
        private_key_path=reviewer_keys.private,
        public_key_path=reviewer_keys.public,
        reviewed_at=CREATED + timedelta(hours=1),
    )
    assert "meta_analysis_sha256" in review and "evidence_sha256" in review
    approval = _write_json(tmp_path / "stage15-approval.json", review)
    combined = merge_review(
        gate,
        review,
        gate_sha256=gate_sha,
        approval_sha256=lineage.artifact_sha256(approval),
        public_key_path=reviewer_keys.public,
    )
    assert _review_is_valid(combined, reviewer_keys.public) is (
        status in {"PROCEED", "PIVOT"}
    )
    return _write_json(tmp_path / f"stage15-{status}" / "decision.v3.json", combined)


def _five_page_pdf(
    path: Path,
    *,
    first_page_text: str | None = None,
    last_page_text: str | None = None,
) -> Path:
    writer = PdfWriter()
    for index in range(5):
        page = writer.add_blank_page(width=612, height=792)
        page_text = (
            first_page_text
            if index == 0
            else last_page_text if index == 4 else None
        )
        if page_text:
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): writer._add_object(font)}
                    )
                }
            )
            stream = DecodedStreamObject()
            escaped = page_text.replace("(", r"\(").replace(")", r"\)")
            stream.set_data(
                f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
            )
            page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def _stage20_inputs(
    tmp_path: Path,
    *,
    decision: str,
    reviewer_keys,
    citation_status: str = "verified_primary",
    numeric_status: str = "verified_artifact",
) -> tuple[Path, Path, Path, Path, Path]:
    paper_dir = tmp_path / "paper"
    manuscript = _manuscript(paper_dir / "main_v2.tex")
    (paper_dir / "compliance.tex").write_text("compliance fixture\n", encoding="utf-8")
    (paper_dir / "refs.bib").write_text("refs fixture\n", encoding="utf-8")
    (paper_dir / "spconf.sty").write_text("style fixture\n", encoding="utf-8")
    (paper_dir / "IEEEbib.bst").write_text("bst fixture\n", encoding="utf-8")
    stage15 = _reviewed_stage15(tmp_path, reviewer_keys, decision)
    stage15_sha = lineage.artifact_sha256(stage15)
    values = {
        "ResultHeadline": "reviewed headline",
        "ResultOne": "delta R-squared is resolved",
        "ResultConclusion": "reviewed conclusion",
    }
    claims_dir = tmp_path / "claims"
    claims_dir.mkdir(parents=True)
    macros = claims_dir / "robust_map_placeholders.tex"
    macros.write_text("% reviewed claim macros\n", encoding="utf-8")
    claim_macros_sha256 = lineage.artifact_sha256(macros)
    figures = tmp_path / "figures"
    figures.mkdir(parents=True)
    for index, name in enumerate(FIGURE_ARTIFACTS):
        (figures / name).write_bytes(f"stage20-figure-{index}".encode())
    figure_artifacts_sha256 = direct_artifact_hashes(figures)
    figure_summary = _write_json(
        figures / "summary.v3.json",
        {
            "schema_version": "paper-figures-v3",
            "input_sha256": {"stage15": stage15_sha},
            "artifacts_sha256": figure_artifacts_sha256,
        },
    )
    figures_summary_sha256 = lineage.artifact_sha256(figure_summary)
    registry = _registry(
        tmp_path / "registry.json",
        status=citation_status,
        numeric_status=numeric_status,
        stage15_sha256=stage15_sha,
        stage15_status=decision,
        values=values,
        claim_macros_sha256=claim_macros_sha256,
        figures_summary_sha256=figures_summary_sha256,
        figure_artifacts_sha256=figure_artifacts_sha256,
    )
    claims = _write_json(
        claims_dir / "claims.v3.json",
        {
            "schema_version": "paper-claims-v3",
            "status": decision,
            "submission_ready": True,
            "stage15_sha256": stage15_sha,
            "verified_registry_sha256": lineage.artifact_sha256(registry),
            "claim_macros_sha256": claim_macros_sha256,
            "figures_sha256": figures_summary_sha256,
            "figures_summary_sha256": figures_summary_sha256,
            "figure_artifacts_sha256": figure_artifacts_sha256,
            "claim_values_sha256": lineage.canonical_sha256(values),
            "values": values,
        },
    )
    build_dir = tmp_path / "review-draft"
    build_dir.mkdir()
    draft_marker = "REVIEW DRAFT--NOT FOR SUBMISSION"
    pdf = _five_page_pdf(
        build_dir / "stage20_review_draft.pdf",
        first_page_text=draft_marker,
        last_page_text=draft_marker,
    )
    compiled_root = build_dir / "build"
    (compiled_root / "auto").mkdir(parents=True)
    (compiled_root / "figures_v3").mkdir(parents=True)
    for name in ("main_v2.tex", "compliance.tex", "refs.bib", "spconf.sty", "IEEEbib.bst"):
        shutil.copy2(paper_dir / name, compiled_root / name)
    shutil.copy2(macros, compiled_root / "auto" / macros.name)
    for name, text in {
        "author_declaration.tex": "% no author facts asserted\n",
        "tool_provenance.tex": "% no final tool provenance asserted\n",
        "venue_policy.tex": "% venue policy unresolved\n",
    }.items():
        (compiled_root / "auto" / name).write_text(text, encoding="utf-8")
    shutil.copy2(figure_summary, compiled_root / "figures_v3" / figure_summary.name)
    for name in FIGURE_ARTIFACTS:
        shutil.copy2(figures / name, compiled_root / "figures_v3" / name)
    compiled_input_sha256 = {
        **{
            f"source/{name}": lineage.artifact_sha256(compiled_root / name)
            for name in (
                "main_v2.tex",
                "compliance.tex",
                "refs.bib",
                "spconf.sty",
                "IEEEbib.bst",
            )
        },
        "auto/robust_map_placeholders.tex": lineage.artifact_sha256(
            compiled_root / "auto" / macros.name
        ),
        **{
            f"auto/{name}": lineage.artifact_sha256(compiled_root / "auto" / name)
            for name in (
                "author_declaration.tex",
                "tool_provenance.tex",
                "venue_policy.tex",
            )
        },
        "figures_v3/summary.v3.json": lineage.artifact_sha256(
            compiled_root / "figures_v3" / figure_summary.name
        ),
        **{
            f"figures_v3/{name}": lineage.artifact_sha256(
                compiled_root / "figures_v3" / name
            )
            for name in FIGURE_ARTIFACTS
        },
    }
    paper_source_sha256 = {
        f"source/{name}": lineage.artifact_sha256(paper_dir / name)
        for name in (
            "main_v2.tex",
            "compliance.tex",
            "refs.bib",
            "spconf.sty",
            "IEEEbib.bst",
        )
    }
    draft_macro_sha256 = {
        f"auto/{name}": lineage.artifact_sha256(compiled_root / "auto" / name)
        for name in (
            "author_declaration.tex",
            "tool_provenance.tex",
            "venue_policy.tex",
        )
    }
    reviewed_scientific_input_sha256 = {
        "paper/main_v2.tex": lineage.artifact_sha256(manuscript),
        "paper/compliance.tex": lineage.artifact_sha256(paper_dir / "compliance.tex"),
        "paper/refs.bib": lineage.artifact_sha256(paper_dir / "refs.bib"),
        "claims/claims.v3.json": lineage.artifact_sha256(claims),
        "claims/robust_map_placeholders.tex": claim_macros_sha256,
        "claims/verified_registry.v1.json": lineage.artifact_sha256(registry),
        "figures/summary.v3.json": figures_summary_sha256,
        **{
            f"figures/{name}": digest
            for name, digest in figure_artifacts_sha256.items()
        },
    }
    _write_json(
        build_dir / "review_draft_report.v3.json",
        {
            "schema_version": "stage20-review-draft-v3",
            "status": "complete",
            "review_ready": True,
            "submission_ready": False,
            "release_eligible": False,
            "not_for_submission": True,
            "identity_claimed": False,
            "funding_claimed": False,
            "final_ai_provenance_claimed": False,
            "official_author_kit_claimed": False,
            "stage15_status": decision,
            "pages": 5,
            "overfull_boxes": 0,
            "claims_sha256": lineage.artifact_sha256(claims),
            "claim_macros_sha256": claim_macros_sha256,
            "verified_registry_sha256": lineage.artifact_sha256(registry),
            "figures_summary_sha256": figures_summary_sha256,
            "figure_artifacts_sha256": figure_artifacts_sha256,
            "paper_source_sha256": paper_source_sha256,
            "draft_macro_sha256": draft_macro_sha256,
            "reviewed_scientific_input_sha256": (
                reviewed_scientific_input_sha256
            ),
            "reviewed_scientific_input_bundle_sha256": lineage.canonical_sha256(
                reviewed_scientific_input_sha256
            ),
            "compiled_input_sha256": compiled_input_sha256,
            "pdf_sha256": lineage.artifact_sha256(pdf),
        },
    )
    return stage15, build_dir, claims, manuscript, registry


@pytest.mark.parametrize("decision", ["PROCEED", "PIVOT"])
def test_stage20_accepts_reviewed_proceed_or_transparent_pivot(
    tmp_path: Path, decision: str, reviewer_keys
) -> None:
    stage15, build_dir, claims, manuscript, registry = _stage20_inputs(
        tmp_path, decision=decision, reviewer_keys=reviewer_keys
    )
    gate = build_stage20_gate(
        stage15=stage15,
        review_draft=build_dir,
        claims=claims,
        manuscript=manuscript,
        registry=registry,
        reviewer_public_key=reviewer_keys.public,
        created_at=CREATED,
    )
    assert gate["eligible_for_proceed"] is True
    assert gate["status"] == "PENDING_USER_REVIEW"
    assert all(gate["evidence"]["checks"].values())
    assert gate["evidence"]["build_checks"]["exactly_five_pages"] is True


def test_stage20_refuses_pending_and_reviewed_refine_stage15(
    tmp_path: Path, reviewer_keys
) -> None:
    reviewed_refine = _reviewed_stage15(tmp_path, reviewer_keys, "REFINE")
    refine = json.loads(reviewed_refine.read_text(encoding="utf-8"))
    assert _stage15_reviewed_paper_decision(refine, reviewer_keys.public) is None

    pending = dict(refine)
    pending["status"] = "PENDING_USER_REVIEW"
    for field in (
        "reviewed_by",
        "reviewed_at",
        "reviewed_from_status",
        "signature_algorithm",
        "reviewer_public_key_sha256",
        "review_signature_ed25519",
        "review_gate_sha256",
        "review_gate_content_sha256",
        "approval_sha256",
    ):
        pending.pop(field, None)
    assert _stage15_reviewed_paper_decision(pending, reviewer_keys.public) is None


def test_stage20_keeps_citation_and_numeric_verification_fail_closed(
    tmp_path: Path, reviewer_keys
) -> None:
    stage15, build_dir, claims, manuscript, pending = _stage20_inputs(
        tmp_path / "citation",
        decision="PROCEED",
        reviewer_keys=reviewer_keys,
        citation_status="registered_pending_stage5",
    )
    blocked = build_stage20_gate(
        stage15=stage15,
        review_draft=build_dir,
        claims=claims,
        manuscript=manuscript,
        registry=pending,
        reviewer_public_key=reviewer_keys.public,
        created_at=CREATED,
    )
    assert blocked["eligible_for_proceed"] is False
    assert blocked["evidence"]["checks"]["verified_registry"] is False

    stage15, build_dir, claims, manuscript, numeric_blocked = _stage20_inputs(
        tmp_path / "numeric",
        decision="PROCEED",
        reviewer_keys=reviewer_keys,
        numeric_status="blocked_until_stage15_review",
    )
    numeric_gate = build_stage20_gate(
        stage15=stage15,
        review_draft=build_dir,
        claims=claims,
        manuscript=manuscript,
        registry=numeric_blocked,
        reviewer_public_key=reviewer_keys.public,
        created_at=CREATED,
    )
    assert numeric_gate["eligible_for_proceed"] is False
    assert numeric_gate["evidence"]["registry_check"][
        "pending_registry_numeric_claims"
    ] == ["ResultConclusion", "ResultHeadline", "ResultOne"]


def test_static_paper_check_rejects_pending_pdf_and_legacy_headline(tmp_path: Path) -> None:
    manuscript = _manuscript(tmp_path / "paper.tex")
    manuscript.write_text(
        manuscript.read_text(encoding="utf-8") + " reconstructor-independent guarantee",
        encoding="utf-8",
    )
    pdf = _five_page_pdf(
        tmp_path / "pending.pdf", first_page_text="PENDING--STAGE 15"
    )
    report = static_paper_check(manuscript, pdf)
    assert report["passed"] is False
    assert report["checks"]["no_pending_text_in_pdf"] is False
    assert "reconstructor_independent" in report["legacy_pattern_hits"]


def test_stage20_fails_on_tampered_claim_binding(tmp_path: Path, reviewer_keys) -> None:
    # Integrity bindings are fail-closed before quality eligibility is considered.
    stage15, build, claims, manuscript, registry = _stage20_inputs(
        tmp_path, decision="PROCEED", reviewer_keys=reviewer_keys
    )
    report_path = build / "review_draft_report.v3.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["claims_sha256"] = "0" * 64
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="claim-sync"):
        build_stage20_gate(
            stage15=stage15,
            review_draft=build,
            claims=claims,
            manuscript=manuscript,
            registry=registry,
            reviewer_public_key=reviewer_keys.public,
            created_at=CREATED,
        )


@pytest.mark.parametrize(
    "target",
    ("robust_map_placeholders.tex", "summary.v3.json", *FIGURE_ARTIFACTS),
)
def test_stage20_fails_closed_if_macro_figure_pdf_source_or_summary_is_tampered(
    tmp_path: Path, target: str, reviewer_keys
) -> None:
    stage15, build, claims, manuscript, registry = _stage20_inputs(
        tmp_path, decision="PIVOT", reviewer_keys=reviewer_keys
    )
    if target == "robust_map_placeholders.tex":
        path = claims.parent / target
    else:
        path = claims.parent.parent / "figures" / target
    if target == "summary.v3.json":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["tampered"] = True
        _write_json(path, value)
    else:
        with path.open("ab") as stream:
            stream.write(b"tampered")
    with pytest.raises(ValueError):
        build_stage20_gate(
            stage15=stage15,
            review_draft=build,
            claims=claims,
            manuscript=manuscript,
            registry=registry,
            reviewer_public_key=reviewer_keys.public,
            created_at=CREATED,
        )


@pytest.mark.parametrize(
    "target",
    ("auto/robust_map_placeholders.tex", "figures_v3/summary.v3.json")
    + tuple(f"figures_v3/{name}" for name in FIGURE_ARTIFACTS),
)
def test_stage20_fails_closed_if_compiled_evidence_copy_is_tampered(
    tmp_path: Path, target: str, reviewer_keys
) -> None:
    stage15, build, claims, manuscript, registry = _stage20_inputs(
        tmp_path, decision="PROCEED", reviewer_keys=reviewer_keys
    )
    with (build / "build" / target).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="compiled review-draft input"):
        build_stage20_gate(
            stage15=stage15,
            review_draft=build,
            claims=claims,
            manuscript=manuscript,
            registry=registry,
            reviewer_public_key=reviewer_keys.public,
            created_at=CREATED,
        )
