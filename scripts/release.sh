#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
#
# Prepare, tag and (optionally) publish an Aero-VPA release.
#
# Usage:
#   scripts/release.sh                        checks + self-tests, git init/commit, annotated tag, set origin
#   scripts/release.sh --push                 ... then push the branch and the tag to origin
#   scripts/release.sh --push --gh-release    ... and create the GitHub Release (needs an authenticated gh)
#
# Options:
#   --version X.Y.Z   release version; must equal the version in CITATION.cff (default: that version)
#   --skip-tests      run the static checks only, not ruff and the six self-tests
#   --sign            GPG-sign the commit and the tag
#   --remote URL      expected origin (default: git@github.com:ylee-sling/Aero-VPA.git)
#   --python PATH     interpreter for the checks (default: $PYTHON or python3; Python >= 3.10)
#   -h, --help        show this help
#
# Nothing leaves the machine without --push. Before the first push, create the repository
# github.com/ylee-sling/Aero-VPA on GitHub *empty* (no README, licence or .gitignore), otherwise the push of
# the first commit is rejected. The script is idempotent: re-running it after a successful tag only re-checks.
set -euo pipefail

TITLE="Aero-VPA: Few-Step Flow-Matching Control over Non-Reconstructive JEPA Latents for Agile Flight on Edge Hardware"
DOI="10.5281/zenodo.23633375"
GITHUB_REPO="ylee-sling/Aero-VPA"
REMOTE_URL="git@github.com:${GITHUB_REPO}.git"
BRANCH="main"
PY="${PYTHON:-python3}"
PUSH=0
GH_RELEASE=0
RUN_TESTS=1
SIGN=0
VERSION=""

die() { printf 'release.sh: error: %s\n' "$*" >&2; exit 1; }
info() { printf '==> %s\n' "$*"; }
usage() { sed -n '5,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --version) [ $# -ge 2 ] || die "--version needs a value"; VERSION="${2#v}"; shift 2 ;;
        --remote)  [ $# -ge 2 ] || die "--remote needs a value"; REMOTE_URL="$2"; shift 2 ;;
        --python)  [ $# -ge 2 ] || die "--python needs a value"; PY="$2"; shift 2 ;;
        --push) PUSH=1; shift ;;
        --gh-release) GH_RELEASE=1; shift ;;
        --skip-tests) RUN_TESTS=0; shift ;;
        --sign) SIGN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option '$1' (see --help)" ;;
    esac
done
[ "$GH_RELEASE" -eq 0 ] || [ "$PUSH" -eq 1 ] || die "--gh-release requires --push"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- preflight -------------------------------------------------------------------------------------------
command -v git >/dev/null 2>&1 || die "git not found"
command -v "$PY" >/dev/null 2>&1 || die "Python interpreter '$PY' not found (use --python)"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || die "Python >= 3.10 required ($PY)"
[ "$GH_RELEASE" -eq 0 ] || command -v gh >/dev/null 2>&1 || die "--gh-release needs the GitHub CLI (gh)"

CFF_VERSION="$(sed -n 's/^version:[[:space:]]*"\{0,1\}\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)"\{0,1\}[[:space:]]*$/\1/p' CITATION.cff)"
[ -n "$CFF_VERSION" ] || die "CITATION.cff has no 'version: \"X.Y.Z\"' line"
VERSION="${VERSION:-$CFF_VERSION}"
[ "$VERSION" = "$CFF_VERSION" ] || die "version $VERSION differs from CITATION.cff ($CFF_VERSION); update CITATION.cff first"
TAG="v${VERSION}"

if [ ! -d .git ]; then
    info "initialising git repository on branch '$BRANCH'"
    git init -q
    git symbolic-ref HEAD "refs/heads/$BRANCH"          # instead of 'git init -b' (needs git >= 2.28)
fi
CURRENT_BRANCH="$(git symbolic-ref --short -q HEAD || true)"
[ "$CURRENT_BRANCH" = "$BRANCH" ] || die "on branch '${CURRENT_BRANCH:-detached HEAD}', expected '$BRANCH'"

# Run after 'git init' so that the checks see exactly the files 'git add -A' will commit.
info "release checks ($("$PY" -c 'import platform; print("Python", platform.python_version())'))"
if [ "$RUN_TESTS" -eq 1 ]; then
    "$PY" -B scripts/check_release.py --run-tests
else
    "$PY" -B scripts/check_release.py
fi

# --- commit and tag ----------------------------------------------------------------------------------------
GIT_ID=()
if [ -z "$(git config user.email || true)" ]; then
    GIT_ID=(-c "user.name=Yeonseok Lee" -c "user.email=ylee@sling.ai.kr")
    info "git identity not configured; committing as Yeonseok Lee <ylee@sling.ai.kr>"
fi
COMMIT_SIGN=()
TAG_MODE="-a"
if [ "$SIGN" -eq 1 ]; then
    COMMIT_SIGN=(-S)
    TAG_MODE="-s"
fi
git_id() { git ${GIT_ID[@]+"${GIT_ID[@]}"} "$@"; }      # empty-array safe under 'set -u' on bash 3.2 (macOS)

git add -A
git add --chmod=+x scripts/release.sh                     # keep the executable bit, also from a zip on Windows
if ! git rev-parse -q --verify HEAD >/dev/null; then
    info "creating the initial commit"
    git_id commit -q ${COMMIT_SIGN[@]+"${COMMIT_SIGN[@]}"} \
        -m "Aero-VPA ${TAG}: reference implementation" \
        -m "Reference PyTorch implementation of the preprint
\"${TITLE}\",
https://doi.org/${DOI}

Code: GPL-3.0-only. The preprint PDF in docs/ is CC BY 4.0 (Copyright (C) 2026 Yeonseok Lee)."
elif ! git diff --cached --quiet; then
    git rev-parse -q --verify "refs/tags/$TAG" >/dev/null && die "tag $TAG exists but the tree has changed; bump the version in CITATION.cff"
    info "committing changes for $TAG"
    git_id commit -q ${COMMIT_SIGN[@]+"${COMMIT_SIGN[@]}"} -m "Release ${TAG}"
else
    info "working tree clean; nothing to commit"
fi

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    [ "$(git rev-list -n 1 "$TAG")" = "$(git rev-parse HEAD)" ] || die "tag $TAG exists and does not point at HEAD"
    info "tag $TAG already points at HEAD"
else
    info "creating annotated tag $TAG"
    git_id tag "$TAG_MODE" "$TAG" -m "Aero-VPA ${TAG}" -m "${TITLE}" -m "Preprint: https://doi.org/${DOI}"
fi

# --- remote ----------------------------------------------------------------------------------------------
if CURRENT_URL="$(git remote get-url origin 2>/dev/null)"; then
    [ "$CURRENT_URL" = "$REMOTE_URL" ] || die "origin is '$CURRENT_URL', expected '$REMOTE_URL' (see --remote)"
else
    info "adding origin $REMOTE_URL"
    git remote add origin "$REMOTE_URL"
fi

# --- publish (only with --push) ----------------------------------------------------------------------------
if [ "$PUSH" -eq 0 ]; then
    info "release $TAG prepared locally at $(git rev-parse --short HEAD); nothing was pushed"
    printf '\nTo publish, create the empty repository https://github.com/%s and run:\n' "$GITHUB_REPO"
    printf '    scripts/release.sh --push               # or: git push -u origin %s && git push origin %s\n' "$BRANCH" "$TAG"
    printf '    scripts/release.sh --push --gh-release  # also create the GitHub Release\n'
    exit 0
fi

info "pushing $BRANCH and $TAG to origin"
git push -u origin "$BRANCH"
git push origin "refs/tags/$TAG"

if [ "$GH_RELEASE" -eq 1 ]; then
    if gh release view "$TAG" --repo "$GITHUB_REPO" >/dev/null 2>&1; then
        info "GitHub Release $TAG already exists"
    else
        NOTES="$(mktemp)"
        trap 'rm -f "$NOTES"' EXIT
        cat > "$NOTES" <<EOF
Reference PyTorch implementation of the preprint **${TITLE}** (https://doi.org/${DOI}).

* All networks of Table 1 (18.13M parameters), the training objectives, the uncertainty-gated safety layer and the
  closed-loop runtime of Algorithm 1 (decision loop and 100 Hz executor).
* Six offline self-tests: \`python scripts/check_release.py --run-tests\`.
* This release ships no trained weights and no flight results.

**License.** Code: GPL-3.0-only. The preprint PDF in \`docs/\` is not covered by the GPL: Copyright (C) 2026
Yeonseok Lee, CC BY 4.0. For commercial integration, proprietary licensing, or defense adaptation, contact ylee@sling.ai.kr.
EOF
        info "creating GitHub Release $TAG"
        gh release create "$TAG" --repo "$GITHUB_REPO" --verify-tag --title "Aero-VPA ${TAG}" --notes-file "$NOTES"
    fi
fi
info "done: https://github.com/${GITHUB_REPO}/releases/tag/${TAG}"
