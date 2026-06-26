#!/usr/bin/env bash
#
# git_commit_each.sh — commit every changed/new file as its OWN commit, with a
# meaningful, conventional-commit-style message, then optionally push.
#
# WHY would you want one commit per file?
#   * A clean, readable history where each change is isolated.
#   * Easy `git revert <one file>` without untangling unrelated edits.
#   * A great way to *learn* what `git add` / `git commit` / `git push` actually
#     do, one step at a time.
#
# THE THREE GIT VERBS THIS SCRIPT USES
#   git add <file>      Move a file's changes into the "staging area" (a.k.a. the
#                       "index") — a holding pen for what the NEXT commit will
#                       contain. Nothing is recorded in history yet.
#   git commit -m "..." Take everything currently staged and record it as a
#                       permanent snapshot in your LOCAL repository, with a
#                       message describing it.
#   git push            Send your local commits to the REMOTE (here, GitHub).
#                       This is the only step that leaves your machine.
#
# USAGE
#   scripts/git_commit_each.sh            # commit each file locally (no push)
#   scripts/git_commit_each.sh --dry-run  # show what WOULD happen, change nothing
#   scripts/git_commit_each.sh --push     # commit each file, then push to origin
#   scripts/git_commit_each.sh -h         # help
#
# SAFETY: by default this script does NOT push. Pushing is the irreversible,
# outward-facing step, so you must opt in with --push. Start with --dry-run.

# ---------------------------------------------------------------------------
# 'set' makes the script fail loudly instead of limping on after an error:
#   -e          exit immediately if any command returns non-zero
#   -u          treat use of an unset variable as an error
#   -o pipefail a pipeline fails if ANY command in it fails (not just the last)
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Option parsing -------------------------------------------------------
DRY_RUN=false
PUSH=false

usage() {
  # 'sed' here just prints the comment header above as help text. Simpler than
  # duplicating it. (Lines starting with '#!' or this function are skipped.)
  cat <<'EOF'
Commit each changed/new file as its own commit.

Options:
  --dry-run   Print the git commands without running them.
  --push      After committing, push the current branch to 'origin'.
  -h, --help  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --push)    PUSH=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

# ---- Work from the repository root ----------------------------------------
# 'git rev-parse --show-toplevel' prints the absolute path of the repo root, so
# the script behaves the same no matter which sub-directory you run it from.
# (It also errors out if you're not inside a git repo, which is exactly what we
# want — there is nothing to commit otherwise.)
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# message_for <path> — derive a commit message from a file path.
#
# This encodes the "Conventional Commits" convention (type(scope): summary),
# which keeps history searchable and even drives automated changelogs. Pattern
# matching on the path lets one rule cover many files, while still allowing a
# hand-written message for the important modules.
# ---------------------------------------------------------------------------
message_for() {
  local path="$1"
  local base
  base="$(basename "$path")"

  case "$path" in
    moe/__init__.py)  echo "feat(moe): expose the public package API" ;;
    moe/config.py)    echo "feat(config): add the validated MoEConfig dataclass" ;;
    moe/router.py)    echo "feat(router): add TopK, Switch and ExpertChoice routers" ;;
    moe/losses.py)    echo "feat(losses): add aux load-balancing and router z-loss" ;;
    moe/experts.py)   echo "feat(experts): add Expert FFN and ExpertBank dispatch" ;;
    moe/layer.py)     echo "feat(layer): add the composable MoELayer" ;;
    moe/utils.py)     echo "feat(utils): add routing monitoring and diagnostics" ;;
    moe/bench.py)     echo "perf(bench): add MoE-vs-dense forward benchmark" ;;
    moe/*.py)         echo "feat(moe): add ${base}" ;;

    tests/conftest.py) echo "test: add shared pytest fixtures" ;;
    tests/*.py)        echo "test: add ${base%.py} tests" ;;

    docs/course/*)    echo "docs(course): add ${base}" ;;
    docs/*)           echo "docs: add ${base}" ;;

    scripts/*)        echo "chore(scripts): add ${base}" ;;

    pyproject.toml)   echo "build: add packaging and tool configuration" ;;
    Makefile)         echo "build: add make targets (test, lint, bench, docs)" ;;
    .gitignore)       echo "chore: add .gitignore" ;;
    README.md)        echo "docs: add the project README" ;;

    # Fallback for anything not explicitly named above.
    *)                echo "chore: add ${path}" ;;
  esac
}

# ---------------------------------------------------------------------------
# Build the list of files to commit, EACH listed individually.
#
# Why not 'git status --porcelain'? Because for a brand-new directory git
# collapses it to a single line (e.g. '?? moe/') instead of listing its files.
# We want per-FILE granularity, so we gather three precise sources and dedupe:
#
#   git ls-files --others --exclude-standard
#       → untracked files, expanded to individual paths, respecting .gitignore
#   git diff --name-only
#       → tracked files with UNSTAGED modifications
#   git diff --name-only --cached
#       → files already staged (in case you staged something earlier)
#
# (Filenames with spaces/newlines would need -z and a NUL-delimited read; this
#  repo has none, so we keep it readable. The caveat is noted for your learning.)
# ---------------------------------------------------------------------------
mapfile -t FILES < <(
  {
    git ls-files --others --exclude-standard
    git diff --name-only
    git diff --name-only --cached
  } | sort -u
)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "Nothing to commit — working tree is clean. ✔"
  exit 0
fi

echo "Found ${#FILES[@]} file(s) to commit individually."
$DRY_RUN && echo "(dry-run: no changes will be made)"
echo

# ---- Commit loop: one file → one commit -----------------------------------
committed=0
for file in "${FILES[@]}"; do
  message="$(message_for "$file")"

  if $DRY_RUN; then
    # Show the exact commands a real run would execute.
    printf 'git add -- %q\n' "$file"
    printf 'git commit -m %q\n\n' "$message"
    continue
  fi

  # 1) Stage just this one file.
  git add -- "$file"

  # 2) Guard against an empty commit: if staging produced no difference (the
  #    file is byte-identical to what's already committed), skip it. The
  #    '--quiet' form of 'git diff --cached' exits 0 when there is nothing
  #    staged for this path, non-zero when there is.
  if git diff --cached --quiet -- "$file"; then
    echo "skip (no change): $file"
    continue
  fi

  # 3) Record the snapshot. We pass the pathspec so the commit contains ONLY
  #    this file even if something else were somehow staged.
  git commit --quiet -m "$message" -- "$file"
  echo "committed: $file   →   $message"
  committed=$((committed + 1))
done

echo
echo "Done. Created ${committed} commit(s)."

# ---- Optional push --------------------------------------------------------
if $PUSH && ! $DRY_RUN; then
  branch="$(git rev-parse --abbrev-ref HEAD)"
  if git remote get-url origin >/dev/null 2>&1; then
    echo "Pushing branch '${branch}' to origin..."
    # '-u' sets the upstream so future plain 'git push' / 'git pull' just work.
    git push -u origin "$branch"
    echo "Pushed. ✔"
  else
    echo "No 'origin' remote configured; skipping push." >&2
    echo "Add one with:  git remote add origin <url>" >&2
  fi
elif $PUSH && $DRY_RUN; then
  echo "(dry-run: would push the current branch to origin)"
else
  echo "Not pushing (re-run with --push to publish to GitHub)."
fi
