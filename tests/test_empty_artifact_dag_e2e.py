from pathlib import Path
import json
import shutil
import subprocess

from pypdf import PdfReader

from ecgcert.execution import DAGRunner, ExperimentManifest, ResultEnvelope


ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    )


def test_empty_artifact_dag_rebuilds_bound_figures_macros_and_five_page_pdf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "ecgcert.execution.runner.lineage.environment_sha256", lambda: "a" * 64,
    )
    monkeypatch.setattr(
        "ecgcert.execution.runner.lineage.hardware_fingerprint",
        lambda: {"cpu_count": 1, "gpu": [], "cuda_runtime": "unavailable"},
    )
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "paper" / "auto").mkdir(parents=True)
    (repo / "environments").mkdir()
    shutil.copytree(ROOT / "src" / "ecgcert", repo / "src" / "ecgcert")
    shutil.copy2(
        ROOT / "scripts" / "compile_submission_v3.py",
        repo / "scripts" / "compile_submission_v3.py",
    )
    for name in ("main_v2.tex", "refs.bib", "spconf.sty", "IEEEbib.bst"):
        shutil.copy2(ROOT / "paper" / name, repo / "paper" / name)
    shutil.copy2(
        ROOT / "paper" / "auto" / "robust_map_placeholders.tex",
        repo / "paper" / "auto" / "robust_map_placeholders.tex",
    )
    for name in ("cpu.lock.txt", "gpu.lock.txt"):
        shutil.copy2(ROOT / "environments" / name, repo / "environments" / name)
    (repo / "make_inputs.py").write_text(
        """from pathlib import Path
import json
from pypdf import PdfWriter
from ecgcert import lineage
from ecgcert.paper_evidence import direct_artifact_hashes

figures = Path('artifacts/paper/figures')
claims = Path('artifacts/paper/claims')
figures.mkdir(parents=True)
claims.mkdir(parents=True)
for name in ('figure1_robust_map.pdf', 'figure2_prediction_gain.pdf'):
    writer = PdfWriter()
    writer.add_blank_page(width=240, height=150)
    with (figures / name).open('wb') as stream:
        writer.write(stream)
for name in ('figure1_source.parquet', 'figure2_source.parquet'):
    (figures / name).write_bytes(b'tiny-e2e-source-table')
artifacts = direct_artifact_hashes(figures)
stage15_sha = 'b' * 64
summary_path = figures / 'summary.v3.json'
summary_path.write_text(json.dumps({
    'schema_version': 'paper-figures-v3',
    'artifacts_sha256': artifacts,
    'input_sha256': {'stage15': stage15_sha},
}, sort_keys=True) + '\\n')
macros = claims / 'robust_map_placeholders.tex'
macros.write_bytes(Path('paper/auto/robust_map_placeholders.tex').read_bytes())
macro_sha = lineage.artifact_sha256(macros)
summary_sha = lineage.artifact_sha256(summary_path)
registry_path = claims / 'verified_registry.v1.json'
registry_path.write_text(json.dumps({
    'schema_version': 'verified-registry-v1',
    'claim_macros_sha256': macro_sha,
    'figures_summary_sha256': summary_sha,
    'figure_artifacts_sha256': artifacts,
}, sort_keys=True) + '\\n')
(claims / 'claims.v3.json').write_text(json.dumps({
    'schema_version': 'paper-claims-v3',
    'submission_ready': False,
    'status': 'PENDING_USER_REVIEW',
    'stage15_sha256': stage15_sha,
    'claim_macros_sha256': macro_sha,
    'figures_sha256': summary_sha,
    'figures_summary_sha256': summary_sha,
    'figure_artifacts_sha256': artifacts,
    'verified_registry_sha256': lineage.artifact_sha256(registry_path),
}, sort_keys=True) + '\\n')
""",
        encoding="utf-8",
    )
    manifest = ExperimentManifest.from_dict({
        "schema_version": 1,
        "nodes": [
            {
                "id": "tiny_evidence",
                "profile": ["icassp", "extended", "legacy"],
                "command": ["{python}", "make_inputs.py"],
                "resource": {"kind": "cpu", "cpus": 1, "memory_gb": 2, "gpus": 0},
                "deps": [],
                "inputs": [],
                "outputs": ["artifacts/paper/figures", "artifacts/paper/claims"],
                "timeout": 30,
                "seed": 0,
            },
            {
                "id": "tiny_compile",
                "profile": ["icassp", "extended", "legacy"],
                "command": [
                    "{python}", "scripts/compile_submission_v3.py",
                    "--claims", "artifacts/paper/claims",
                    "--figures", "artifacts/paper/figures",
                    "--output-dir", "artifacts/paper/submission",
                ],
                "resource": {"kind": "paper", "cpus": 1, "memory_gb": 2, "gpus": 0},
                "deps": ["tiny_evidence"],
                "inputs": ["artifacts/paper/figures", "artifacts/paper/claims"],
                "outputs": ["artifacts/paper/submission"],
                "timeout": 120,
                "seed": 0,
            },
        ],
    })
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "tiny evidence DAG")
    run_root = tmp_path / "runs"
    assert not (run_root / "tiny" / "workspace" / "artifacts").exists()
    run_dir = DAGRunner(
        repo=repo,
        manifest=manifest,
        profile="icassp",
        run_root=run_root,
        run_id="tiny",
    ).run()

    submission = run_dir / "workspace" / "artifacts" / "paper" / "submission"
    report = json.loads((submission / "build_report.v3.json").read_text(encoding="utf-8"))
    assert report["pages"] == 5
    assert report["overfull_boxes"] == 0
    assert len(PdfReader(str(submission / "main_v2.pdf")).pages) == 5
    envelopes = [
        ResultEnvelope.read(run_dir / "envelopes" / f"{node_id}.json")
        for node_id in ("tiny_evidence", "tiny_compile")
    ]
    assert envelopes[0].source_sha256 == envelopes[1].source_sha256
    assert all(envelope.environment_lock_sha256 for envelope in envelopes)
