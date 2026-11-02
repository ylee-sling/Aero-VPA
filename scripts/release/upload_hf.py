# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""
scripts/release/upload_hf.py -- Upload the release staged by scripts/release/package_release.py to the Hugging
Face Hub: the model and the dataset repository are created private, uploaded, checked against their
MANIFEST.json, optionally tagged, and made public only with --public.

    hf auth login                                                  # once: a token with write access (or HF_TOKEN)
    python3 scripts/release/upload_hf.py --namespace ORG --dry_run  # what would go where
    python3 scripts/release/upload_hf.py --namespace ORG            # private repositories, uploaded and verified
    python3 scripts/release/upload_hf.py --namespace ORG --tag v1.1.0 --public     # on release day
    python3 scripts/release/upload_hf.py --self-test

Uploads are resumable: rerunning skips the files already on the Hub. Needs huggingface_hub >= 0.32.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "scripts", "release", "release_manifest.json")
ALWAYS = ("README.md", "MANIFEST.json", "SHA256SUMS")


def staged_repos(out: str, manifest_path: str = MANIFEST) -> List[Dict[str, Any]]:
    """The staged repositories: directory, repo id and type, files from their MANIFEST.json."""
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    repos = []
    for kind, name in (("models", m["model_repo"]), ("datasets", m["dataset_repo"])):
        d = os.path.join(out, kind, name)
        path = os.path.join(d, "MANIFEST.json")
        if not os.path.isfile(path):
            raise SystemExit("%s not found: run scripts/release/package_release.py first" % path)
        with open(path, encoding="utf-8") as f:
            staged = json.load(f)
        repos.append({"dir": d, "repo_id": staged["repo_id"], "repo_type": staged["repo_type"],
                      "files": [e["path"] for e in staged["files"]], "bytes": sum(e["bytes"] for e in staged["files"]),
                      "generated": staged.get("generated", "")})
    return repos


class RepoMissing(Exception):
    """Raised by stand-ins of HfApi (self-test) for a repository that does not exist."""


def _not_found_errors() -> tuple:
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        try:                                 # huggingface_hub < 0.25
            from huggingface_hub.utils import RepositoryNotFoundError
        except ImportError:                  # no huggingface_hub (self-test only)
            return (RepoMissing,)
    return (RepositoryNotFoundError, RepoMissing)


def is_private(api: Any, repo_id: str, repo_type: str) -> Optional[bool]:
    """True/False for an existing repository, None when it does not exist yet."""
    try:
        return bool(api.repo_info(repo_id, repo_type=repo_type).private)
    except _not_found_errors():
        return None


def set_public(api: Any, repo_id: str, repo_type: str) -> None:
    if hasattr(api, "update_repo_settings"):
        api.update_repo_settings(repo_id, private=False, repo_type=repo_type)
    else:                                    # older huggingface_hub
        api.update_repo_visibility(repo_id, private=False, repo_type=repo_type)


def create_tag(api: Any, repo_id: str, repo_type: str, tag: str) -> None:
    try:
        api.create_tag(repo_id, tag=tag, repo_type=repo_type, exist_ok=True)
    except TypeError:                        # no exist_ok in older huggingface_hub
        api.create_tag(repo_id, tag=tag, repo_type=repo_type)


def upload(api: Any, repos: Sequence[Dict[str, Any]], namespace: str, tag: Optional[str] = None,
           public: bool = False, allow_public_update: bool = False) -> None:
    who = api.whoami()
    names = [who.get("name")] + [o.get("name") for o in who.get("orgs", []) if isinstance(o, dict)]
    if namespace not in names:
        raise SystemExit("the token belongs to %s (orgs: %s), not to %s" % (
            who.get("name"), ", ".join(n for n in names[1:] if n) or "none", namespace))
    for r in repos:
        if not r["repo_id"].startswith(namespace + "/"):
            raise SystemExit("%s was staged for another namespace: rerun package_release.py --namespace %s"
                             % (r["repo_id"], namespace))
        private = is_private(api, r["repo_id"], r["repo_type"])
        if private is False and not (public or allow_public_update):
            raise SystemExit("%s is already public: uploading would publish at once (pass --public or "
                             "--allow_public_update)" % r["repo_id"])
        if private is None:
            print("creating private %s %s" % (r["repo_type"], r["repo_id"]))
            api.create_repo(r["repo_id"], repo_type=r["repo_type"], private=True, exist_ok=True)
        print("uploading %d files to %s ..." % (len(r["files"]) + len(ALWAYS), r["repo_id"]))
        api.upload_folder(repo_id=r["repo_id"], folder_path=r["dir"], repo_type=r["repo_type"],
                          commit_message="Aero-VPA release (%s)" % r["generated"] if r["generated"] else
                          "Aero-VPA release")
        remote = set(api.list_repo_files(r["repo_id"], repo_type=r["repo_type"]))
        missing = sorted((set(r["files"]) | set(ALWAYS)) - remote)
        if missing:
            raise SystemExit("%s is missing %d files after the upload: %s (rerun to resume)" % (
                r["repo_id"], len(missing), ", ".join(missing[:10])))
        print("  verified: all %d staged files are on the Hub" % (len(r["files"]) + len(ALWAYS)))
        if tag:
            create_tag(api, r["repo_id"], r["repo_type"], tag)
            print("  tagged %s" % tag)
        if public:
            set_public(api, r["repo_id"], r["repo_type"])
            print("  public: https://huggingface.co/%s%s" % ("datasets/" if r["repo_type"] == "dataset" else "",
                                                           r["repo_id"]))
        else:
            print("  private: https://huggingface.co/%s%s (rerun with --public to publish)" % (
                "datasets/" if r["repo_type"] == "dataset" else "", r["repo_id"]))


# ---------------------------------------------------------------------------------------------
# Self-test (python3 scripts/release/upload_hf.py --self-test): a stand-in for HfApi, no network
# ---------------------------------------------------------------------------------------------
class _FakeApi:
    def __init__(self, existing: Optional[Dict[str, bool]] = None, drop: Sequence[str] = ()) -> None:
        self.calls: List[tuple] = []
        self.existing = dict(existing or {})
        self.files: Dict[str, List[str]] = {}
        self.drop = set(drop)

    def whoami(self) -> Dict[str, Any]:
        return {"name": "someone", "orgs": [{"name": "test-org"}]}

    def repo_info(self, repo_id: str, repo_type: str) -> Any:
        if repo_id not in self.existing:
            raise RepoMissing(repo_id)
        return type("Info", (), {"private": self.existing[repo_id]})()

    def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool) -> None:
        self.calls.append(("create", repo_id, repo_type, private))
        self.existing[repo_id] = private

    def upload_folder(self, repo_id: str, folder_path: str, repo_type: str, commit_message: str) -> None:
        self.calls.append(("upload", repo_id, repo_type))
        names = []
        for dirpath, _, files in os.walk(folder_path):
            names += [os.path.relpath(os.path.join(dirpath, n), folder_path) for n in files]
        self.files[repo_id] = [n for n in names if n not in self.drop] + [".gitattributes"]

    def list_repo_files(self, repo_id: str, repo_type: str) -> List[str]:
        return self.files.get(repo_id, [])

    def create_tag(self, repo_id: str, tag: str, repo_type: str, exist_ok: bool = False) -> None:
        self.calls.append(("tag", repo_id, tag))

    def update_repo_settings(self, repo_id: str, private: bool, repo_type: str) -> None:
        self.calls.append(("public", repo_id, private))
        self.existing[repo_id] = private


def _self_test() -> None:
    import contextlib
    import io
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        manifest = os.path.join(tmp, "release_manifest.json")
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump({"model_repo": "m", "dataset_repo": "d"}, f)
        for kind, name, rtype, files in (("models", "m", "model", ["checkpoints/a.pt", "media/v.mp4"]),
                                         ("datasets", "d", "dataset", ["missions/missions.parquet"])):
            d = os.path.join(tmp, "release", kind, name)
            for rel in files + ["README.md", "SHA256SUMS"]:
                os.makedirs(os.path.dirname(os.path.join(d, rel)) or d, exist_ok=True)
                with open(os.path.join(d, rel), "w", encoding="utf-8") as f:
                    f.write("x")
            with open(os.path.join(d, "MANIFEST.json"), "w", encoding="utf-8") as f:
                json.dump({"repo_id": "test-org/" + name, "repo_type": rtype, "generated": "today",
                           "files": [{"path": p, "bytes": 1} for p in files + ["README.md"]]}, f)
        repos = staged_repos(os.path.join(tmp, "release"), manifest)
        assert [r["repo_id"] for r in repos] == ["test-org/m", "test-org/d"]
        quiet = contextlib.redirect_stdout(io.StringIO())

        api = _FakeApi()
        with quiet:
            upload(api, repos, "test-org")
        assert api.calls == [("create", "test-org/m", "model", True), ("upload", "test-org/m", "model"),
                             ("create", "test-org/d", "dataset", True), ("upload", "test-org/d", "dataset")]
        api2 = _FakeApi(existing={"test-org/m": True, "test-org/d": True})
        with contextlib.redirect_stdout(io.StringIO()):
            upload(api2, repos, "test-org", tag="v1.1.0", public=True)
        assert ("tag", "test-org/d", "v1.1.0") in api2.calls and ("public", "test-org/m", False) in api2.calls
        assert not any(c[0] == "create" for c in api2.calls)

        for bad, expect in ((lambda: upload(_FakeApi(), repos, "other-org"), "not to other-org"),
                            (lambda: upload(_FakeApi(existing={"test-org/m": False}), repos, "test-org"),
                             "already public"),
                            (lambda: upload(_FakeApi(drop=["media/v.mp4"]), repos, "test-org"), "media/v.mp4")):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    bad()
                raise AssertionError("expected SystemExit: " + expect)
            except SystemExit as exc:
                assert expect in str(exc), str(exc)
    print("upload_hf self-test passed: private creation, upload, verification, tag, publication, refusals")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--namespace", help="Hugging Face user or organisation the release was staged for")
    p.add_argument("--out", default="release", help="staging directory of package_release.py [release]")
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--tag", default=None, help="also tag both repositories (e.g. v1.1.0, the GitHub tag)")
    p.add_argument("--public", action="store_true", help="make both repositories public after the upload")
    p.add_argument("--allow_public_update", action="store_true", help="upload to a repository that is already public")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    repos = staged_repos(args.out, args.manifest)
    for r in repos:
        print("%-8s %-40s %4d files  %8.1f MB  from %s" % (r["repo_type"], r["repo_id"], len(r["files"]),
                                                          r["bytes"] / 2**20, os.path.relpath(r["dir"])))
    if args.dry_run:
        return 0
    if not args.namespace:
        raise SystemExit("--namespace is required")
    from huggingface_hub import HfApi

    upload(HfApi(), repos, args.namespace, tag=args.tag, public=args.public,
           allow_public_update=args.allow_public_update)
    return 0


if __name__ == "__main__":
    sys.exit(main())
