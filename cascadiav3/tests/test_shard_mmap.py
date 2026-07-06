"""Memory-mapped shard loading must be bit-identical to np.load."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

FIXTURE = Path("cascadiav3/fixtures/gumbel_tiny_tensor.npz")


class ShardMmapTest(unittest.TestCase):
    def _require(self):  # type: ignore[no-untyped-def]
        try:
            import numpy as np  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("numpy unavailable")
        if not FIXTURE.exists():
            self.skipTest("gumbel tiny tensor fixture has not been generated")

    def test_mmap_arrays_match_np_load_exactly(self) -> None:
        self._require()
        import numpy as np

        from cascadiav3.expert_tensor_shards import _MmapNpz

        mapped = _MmapNpz(FIXTURE)
        loaded = np.load(FIXTURE, allow_pickle=False)
        try:
            self.assertEqual(sorted(mapped.files), sorted(loaded.files))
            for name in loaded.files:
                reference = loaded[name]
                candidate = mapped[name]
                self.assertEqual(reference.dtype, candidate.dtype, name)
                self.assertEqual(reference.shape, candidate.shape, name)
                self.assertTrue(np.array_equal(reference, np.asarray(candidate)), name)
        finally:
            mapped.close()
            loaded.close()

    def test_shard_examples_identical_across_loaders(self) -> None:
        self._require()
        import numpy as np

        from cascadiav3.expert_tensor_shards import ExpertTensorShard

        previous = os.environ.get("CASCADIA_SHARD_MMAP")
        try:
            os.environ["CASCADIA_SHARD_MMAP"] = "1"
            mapped_shard = ExpertTensorShard(FIXTURE)
            os.environ["CASCADIA_SHARD_MMAP"] = "0"
            eager_shard = ExpertTensorShard(FIXTURE)
            self.assertEqual(len(mapped_shard), len(eager_shard))
            for index in range(len(eager_shard)):
                mapped_example = mapped_shard.example(index)
                eager_example = eager_shard.example(index)
                self.assertEqual(sorted(mapped_example), sorted(eager_example))
                for key, reference in eager_example.items():
                    candidate = mapped_example[key]
                    if isinstance(reference, np.ndarray):
                        self.assertTrue(
                            np.array_equal(reference, np.asarray(candidate)),
                            f"{key}[{index}]",
                        )
                    else:
                        self.assertEqual(reference, candidate, f"{key}[{index}]")
            mapped_shard.close()
            eager_shard.close()
        finally:
            if previous is None:
                os.environ.pop("CASCADIA_SHARD_MMAP", None)
            else:
                os.environ["CASCADIA_SHARD_MMAP"] = previous

    def test_compressed_shard_falls_back_to_np_load(self) -> None:
        self._require()
        import numpy as np

        from cascadiav3.expert_tensor_shards import ExpertTensorShard, _MmapNpz

        with tempfile.TemporaryDirectory() as tmp:
            compressed = Path(tmp) / "compressed.npz"
            with zipfile.ZipFile(FIXTURE) as src, zipfile.ZipFile(
                compressed, "w", compression=zipfile.ZIP_DEFLATED
            ) as dst:
                for info in src.infolist():
                    dst.writestr(info.filename, src.read(info.filename))
            with self.assertRaises(ValueError):
                _MmapNpz(compressed)
            shard = ExpertTensorShard(compressed)
            reference = ExpertTensorShard(FIXTURE)
            try:
                self.assertEqual(len(shard), len(reference))
                self.assertTrue(
                    np.array_equal(
                        np.asarray(reference.tokens), np.asarray(shard.tokens)
                    )
                )
            finally:
                shard.close()
                reference.close()

    def test_shards_survive_source_file_intact(self) -> None:
        """Opening via mmap must not mutate the shard file."""
        self._require()
        from cascadiav3.expert_tensor_shards import ExpertTensorShard, _sha256

        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "copy.npz"
            shutil.copyfile(FIXTURE, copy)
            before = _sha256(copy)
            shard = ExpertTensorShard(copy)
            _ = shard.example(0)
            shard.close()
            self.assertEqual(before, _sha256(copy))


if __name__ == "__main__":
    unittest.main()
