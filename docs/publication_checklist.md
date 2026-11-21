# Manual GitHub publication checklist

The owner will handle Git and GitHub manually. This phase does not authenticate
to GitHub, create a repository, stage files, commit, set remotes or push.

## Local preparation

- [x] Verify the repository root is `Project_TWO`, not the Windows user directory.
- [x] Record all 35 original relative paths, lengths and SHA-256 hashes before edits.
- [x] Preserve original bytes of all three repaired source files and record exact changes.
- [x] Retain all existing numerical results and scientific algorithms.
- [x] Add README, experiment/dependency documentation and a future layout proposal.
- [x] Add tests and distinguish passed checks from missing optional components.
- [x] Exclude environments, caches, credentials and new scratch/run outputs.
- [ ] Owner reviews the complete publication file list and staged diff before committing.

## Personal and sensitive material

- [ ] Review the two PDFs and C-grid photograph. They contain personal identifiers
  and are currently ignored by exact path; keep originals local or explicitly
  approve reviewed sharing copies.
- [ ] If creating redacted PDFs, remove the underlying text/metadata as well as
  visible markings and verify text extraction afterward. No redaction was done
  in this phase.
- [ ] Review `FLOWCHART.jpg`, code comments, future logs and environment exports
  for material the owner does not intend to share.
- [ ] Keep `.env`, credentials, private keys and machine-specific configuration
  out of the selected publication files. Prior pattern scans found no credential
  matches in inspected source, CSV and extracted PDF text; this is not an
  exhaustive guarantee about future additions or image content.
- [ ] Keep `Project_TWO.` excluded: it is an empty archive with an unsafe Windows
  checkout filename. Its original remains locally preserved.

Ignoring a file does not untrack an already committed file. At the inspected
starting state there were no tracked files or commits, so exact ignore rules
can prevent initial inclusion. Before committing manually, inspect both ignored
and candidate files; do not force-add sensitive originals.

## Scientific reproducibility and ownership

- [x] Repair and smoke-test the `main.py` import and timing-key wiring without
  changing numerical calculations.
- [x] Validate JAX and PhiFlow small-grid checks in the recorded optional environment.
- [ ] Review known reporting/timing discrepancies and retain their limits in
  project documentation.
- [ ] Reproduce full scientific experiments if the repository will claim their
  regeneration. Record commands, parameters, seeds, versions, hardware and
  source revision, and keep original figures separate from new runs.
- [ ] Review authorship, institutional/course sharing requirements and the licence
  decision. No licence or permission to redistribute third-party material is
  inferred from this preparation work.

## Owner's manual Git checks

From inside `Project_TWO`, the following are read-only checks:

```console
git rev-parse --show-toplevel
git status --short --branch
git ls-files --others --exclude-standard
git ls-files --others --ignored --exclude-standard
python scripts/verify_preservation.py
python -m pytest -q
```

After selecting/staging files manually, review `git diff --cached --stat` and
`git diff --cached`, including the absence of personal documents and temporary
test logs. Make a deliberate decision about author identity before the initial
commit. The user-directory parent repository must not be used for these actions.

## Private repository and post-publication verification

- [ ] Owner creates the intended repository and explicitly selects **Private**.
- [ ] Owner verifies the destination account, repository URL, visibility and
  collaborator access before adding a remote or publishing.
- [ ] Owner reviews the exact staged content, creates the initial commit and
  performs any push manually.
- [ ] In a clean checkout, install the documented environment and run
  `python scripts/verify_preservation.py --allow-excluded-missing` and pytest.
- [ ] Confirm that original reference CSVs/figures and preservation records are
  present, while the four deliberately excluded originals and local caches are
  absent. Confirm the remote remains private.

These final publication tasks remain with the owner. Nothing in this checklist
authorises the assistant to perform authentication, commits or pushes.
