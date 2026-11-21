# Original-file preservation record

`original-files.json` records all 35 files present before Phase 2 additions,
their exact relative paths, byte lengths and SHA-256 hashes. Git metadata is
excluded. The original total is 6,327,784 bytes. No original was changed while
this snapshot was created. The trailing dot in `Project_TWO.` is intentional.

SHA-256 of the manifest when created:

```text
3b751ee8802801586cc3b796fbf68ab962bfbbb4c6733547529ce18d11b3f322
```

From the repository root, using Python 3.11 or newer:

```console
python scripts/verify_preservation.py
```

After an approved publication that omits the four specifically excluded local
files, use `--allow-excluded-missing`. This option still checks every file that
is present, and does not permit other files to be missing or changed.

The manifest is an integrity record, not a backup or a proof of original run
provenance. `originals/` now retains byte-identical copies of the three source
files that received import/reporting repairs. `repairs.json` records their original and revised
hashes and the reason for each change. The verifier checks both the original
backup and the recorded repaired version; all other originals must match the
snapshot directly. This is not a general exception for source changes.

Keep an independent backup before future source edits. Do not
replace the original hashes to conceal an approved repair: retain this snapshot
and document the exact diff and subsequent hashes separately. New documentation,
tests, configuration and future outputs are outside the original-file snapshot.

The two reports and C-grid photograph contain personal identifiers; their
filenames and hashes are recorded without copying those identifiers into this
manifest. They are ignored by Git, as is the empty trailing-dot archive. They
remain on disk.
