# Component release policy

This repository follows [zylon-base-template](https://github.com/zylon-ai/zylon-base-template).
The common workflow is `.github/workflows/release-please.yml`; customize follow-up tasks
in the reusable `.github/workflows/release-hooks.yml`, preserving its input contract.
Release-please config and manifest files live at the repository root. All common
release jobs use Ubuntu 24.04 and the same pinned release-please-action v5.0.0.
Configure `CI_GITHUB_TOKEN`; existing `PAT` secrets are accepted as an alias. Neither
workflow falls back to GITHUB_TOKEN, so generated PRs/events can trigger normal CI.

`main` publishes new minor versions with release-please (`always-bump-minor`).
Each stable component line has exactly one maintenance publisher: `release/X.Y`
uses `always-bump-patch`. Product/BOM `zylon/*` branches no longer publish releases.
Conventional commits still decide which changes warrant a release PR. Breaking
changes require an explicitly reviewed major-version policy change; this policy
always increments the minor on trunk.

Use the zylon-bom CLI to prepare branches, including old tags:

```sh
python cli/maintenance_branches.py zylon-ai/private-gpt vX.Y.Z --dry-run
python cli/maintenance_branches.py zylon-ai/private-gpt vX.Y.Z
```

BOM `create` and `update` perform the same preparation for managed services.
Preparation copies release automation onto a child of the latest published tag
in that component line before creating the branch ref. Existing branches are
validated and reused, never reset. A BOM keeps its exact artifact pin; sharing a
maintenance branch does not change what existing BOM releases contain.

Merge this repository's policy PR before enabling BOM automation. Existing
`zylon/*` branches retain their old workflows until migrated: preparation opens
retirement PRs for publishers of the selected component line and stops until
those PRs are merged. It never deletes branches or copies unreleased fixes.
Review outstanding fixes and release PRs before retiring a line. Close obsolete
whole-branch backport PRs; do not merge their version/changelog changes.

Prefer fixing `main` first and cherry-picking into maintenance. A merged hotfix
PR into `release/X.Y` also opens an independent code-only forward-port PR onto
`main`; it never merges release-branch ancestry. Release PRs are skipped.
Use squash merges or merge commits, not multi-commit rebase merges, for hotfix PRs.
Direct pushes do not create forward-port PRs: use PRs for maintenance fixes.
Conflicts and edits to files mixing code/dependencies with version metadata stop
for manual forward-porting instead of silently dropping part of a fix. Check the
Forward-port maintenance fixes workflow for failures. Existing forward-port PRs
are not overwritten. Normal CI and review apply before merging.

The Release policy check should be required by branch protection. Do not use
`Release-As` commit trailers or version overrides to bypass version ownership.
Manual tags, manual publication and existing queued workflows are outside this
policy; drain old release runs during rollout. Artifact deployment/channel
workflows remain repo-specific and must be reviewed when enabling maintenance.
