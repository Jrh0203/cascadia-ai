# ADR 0199: Manifest-Driven Source Archives

Status: accepted before the replacement W0 source freeze

## Decision

Every R2-MAP immutable source archive is built from the authorized source
manifest by `tools/r2_map_source_archive.py`. The builder writes only strict
USTAR regular-file members in manifest order and normalizes every header to:

- the exact manifest path, byte count, SHA-256, and mode (`0400` or `0500`);
- uid and gid zero, empty owner/group names, and mtime zero;
- no links, devices, directory entries, USTAR prefixes, PAX/GNU control
  records, AppleDouble files, extended attributes, or nonzero padding; and
- two terminal zero blocks with deterministic 10-KiB record padding.

The self-contained verifier is copied beside `source.tar` and
`source-manifest.json` in the immutable transaction. Every gate verifies the
raw archive before extraction, requires a new absent private workspace,
extracts once with `COPYFILE_DISABLE=1`, and verifies the extracted file and
directory sets, modes, sizes, hashes, link counts, ownership safety, and xattr
policy against the external manifest. A gate never reuses an old workspace.

macOS injects `com.apple.provenance` on the fresh root, auto-created
directories, and extracted files even when the strict USTAR contains no xattr
records. Direct libc experiments show that the kernel immediately re-presents
that attribute after successful removal. The extracted-tree verifier therefore
allows exactly that one host-generated name, treats its value as
non-authoritative, rejects any other or mixed xattr set, and records the count
and path-set digest. The raw USTAR contract remains completely xattr-free, and
the manifest-driven builder never reads host xattrs when constructing it.

## Rejected v40 evidence

The otherwise hash-consistent v40 transaction is permanently rejected. Its
manifest named 699 files, while its BSD-tar stream contained 887 regular
members: 188 unmanifested `._*` AppleDouble records. The same 188 expected
members carried `com.apple.provenance` PAX xattrs, and all 699 expected file
modes differed from the manifest contract. There were no missing expected
files or expected-content mismatches, but an outer archive digest cannot make
unmanifested members acceptable.

The immutable v40 transaction remains evidence and is never rewritten or
deleted. Its exact forensic summary is frozen in
`tests/fixtures/r2_map/rejected-v40-source-archive-characteristics.json`.
Replacement source begins at v41 and must pass both raw archive and
post-extraction verification before any Rust, Python, P1, W7, or protected-seed
gate is admitted.

## Why flags alone are insufficient

`COPYFILE_DISABLE=1` prevents the observed macOS metadata emission, but a flag
does not prove an archive's contents. The manifest-driven builder makes the
authorized member set and metadata constructive; the independent verifier
then fails closed on extras, missing files, duplicates, reordering, unsafe
paths, non-regular types, PAX/GNU extensions, mode drift, byte drift, or
noncanonical trailing data.
