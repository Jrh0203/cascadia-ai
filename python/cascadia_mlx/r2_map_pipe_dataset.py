"""Direct, backpressured R2-MAP packed-batch pipe for native MLX training.

Rust owns replay reconstruction and exact legal-action encoding.  Python owns
only the current packed batch.  No expanded R2-MAP window is written to disk.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import blake3

from cascadia_mlx.r2_map_dataset import (
    R2MapDatasetError,
    R2MapStreamReader,
    _accumulate_packing_statistics,
    _canonical_blake3,
    _empty_packing_statistics,
    _sampler_hash,
    _training_dataset_contract,
    validate_compact_index,
)
from cascadia_mlx.r2_map_model import R2MapBatch
from cascadia_mlx.r2_map_training_contract import (
    R2MapAdapterStep,
    R2MapSupervisedBatch,
    R2MapTrainingAdapter,
)

PIPE_PROTOCOL_ID = "r2-map-packed-batch-pipe-v1"
ADAPTER_PROTOCOL_ID = "r2-map-focal-seat-bootstrap-value-pipe-adapter-v2"
SAMPLER_PROTOCOL_ID = "source-game-focal-turn-packed-pipe-v1"
FOCAL_SEAT_RULE = "global-game-index-mod-4"


class _PipeSession:
    def __init__(
        self,
        command: list[str],
        *,
        expected_source: dict[str, Any],
        expected_mode: str,
        expected_epoch: int,
        expected_seed: int,
        expected_start: tuple[int, int, int],
        expected_bootstrap_value_only: bool,
    ):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise R2MapDatasetError("packed producer pipe creation failed")
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout
        self.bootstrap_value_only = bool(expected_bootstrap_value_only)
        ready = self._read_control()
        if (
            ready.get("type") != "ready"
            or ready.get("protocol_id") != PIPE_PROTOCOL_ID
            or ready.get("source") != expected_source
            or ready.get("mode") != expected_mode
            or ready.get("epoch") != expected_epoch
            or ready.get("sampler_seed") != expected_seed
            or (
                ready.get("start_game_offset"),
                ready.get("start_turn_offset"),
                ready.get("start_batch_index"),
            )
            != expected_start
            or ready.get("focal_seat_rule") != FOCAL_SEAT_RULE
            or ready.get("bootstrap_value_only") is not expected_bootstrap_value_only
        ):
            self.close()
            raise R2MapDatasetError("packed producer ready identity differs")
        self.ready = ready
        self.manifest = ready["manifest"]

    def _read_control(self) -> dict[str, Any]:
        line = self.stdout.readline()
        if not line:
            detail = b""
            if self.process.stderr is not None:
                detail = self.process.stderr.read()
            raise R2MapDatasetError(
                f"packed producer closed unexpectedly: {detail.decode(errors='replace')}"
            )
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise R2MapDatasetError("packed producer control frame is invalid") from error
        if not isinstance(value, dict):
            raise R2MapDatasetError("packed producer control frame must be an object")
        return value

    def _read_exact(self, size: int) -> bytearray:
        if not 0 < size < 1 << 30:
            raise R2MapDatasetError("packed producer payload exceeds the 1 GiB hard gate")
        payload = bytearray(size)
        view = memoryview(payload)
        offset = 0
        while offset < size:
            observed = self.stdout.readinto(view[offset:])
            if not observed:
                raise R2MapDatasetError("packed producer payload is truncated")
            offset += observed
        view.release()
        return payload

    def next_batch(
        self,
    ) -> tuple[R2MapSupervisedBatch, R2MapBatch, dict[str, Any]] | None:
        control = self._read_control()
        if control.get("type") == "done":
            return None
        required = {
            "type",
            "protocol_id",
            "producer_identity",
            "batch_identity",
            "batch_index",
            "first_game_offset",
            "first_turn_offset",
            "next_game_offset",
            "next_turn_offset",
            "groups",
            "padded_width",
            "draft_candidates",
            "game_indices",
            "payload_bytes",
            "payload_blake3",
        }
        if set(control) != required or control["protocol_id"] != PIPE_PROTOCOL_ID:
            raise R2MapDatasetError("packed producer batch control schema differs")
        payload = self._read_exact(int(control["payload_bytes"]))
        if blake3.blake3(payload).hexdigest() != control["payload_blake3"]:
            raise R2MapDatasetError("packed producer payload checksum differs")
        game_indices = tuple(int(value) for value in control["game_indices"])
        with R2MapStreamReader(
            self.manifest,
            payload,
            game_indices=game_indices,
            ordered_game_indices=True,
            bootstrap_value_only=self.bootstrap_value_only,
        ) as reader:
            indices = list(range(len(reader.refs)))
            batch = reader.batch(indices)
            panel = reader.fixed_selected_batch(indices)
        groups, width = batch.validate()
        if groups != control["groups"] or width != control["padded_width"]:
            raise R2MapDatasetError("packed producer decoded shape differs")
        acknowledgement = (
            json.dumps({"ack": control["batch_identity"]}, separators=(",", ":")).encode() + b"\n"
        )
        self.stdin.write(acknowledgement)
        self.stdin.flush()
        return batch, panel, control

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()


class R2MapPackedPipeDatasetAdapter(R2MapTrainingAdapter):
    """Exactly resumable one-focal-seat adapter over persistent Rust pipes."""

    protocol_id = ADAPTER_PROTOCOL_ID

    def __init__(
        self,
        *,
        index: str | Path,
        shard_root: str | Path,
        exporter: str | Path,
        validated_aggregate_receipt: str | Path,
        validated_packing_receipt: str | Path,
        group_batch_size: int = 128,
        maximum_candidates_per_batch: int = 16_384,
        sampler_seed: int = 20260618,
        bootstrap_value_only: bool = True,
    ):
        self.index_path = Path(index).resolve(strict=True)
        self.shard_root = Path(shard_root).resolve(strict=True)
        self.exporter = Path(exporter).resolve(strict=True)
        self.aggregate_receipt = Path(validated_aggregate_receipt).resolve(strict=True)
        self.packing_receipt = Path(validated_packing_receipt).resolve(strict=True)
        self.index = validate_compact_index(self.index_path, shard_root=self.shard_root)
        self.dataset_blake3 = self.index["dataset_manifest"]["dataset_blake3"]
        self.group_batch_size = group_batch_size
        self.maximum_candidates_per_batch = maximum_candidates_per_batch
        self.sampler_seed = sampler_seed
        self.bootstrap_value_only = bootstrap_value_only
        self._sources = {
            source["file_name"]: source for source in self.index["dataset_manifest"]["sources"]
        }
        self._games_by_source: dict[str, list[dict[str, Any]]] = {}
        for game in self.index["games"]:
            self._games_by_source.setdefault(game["source_file_name"], []).append(game)
        self.one_epoch_plan = self._packing_plan()
        self.dataset_contract = _training_dataset_contract(self.index["dataset_manifest"])
        self.dataset_contract.update(
            {
                "adapter_contract_schema_version": 2,
                "adapter_protocol_id": self.protocol_id,
                "pipe_protocol_id": PIPE_PROTOCOL_ID,
                "focal_seat_rule": FOCAL_SEAT_RULE,
                "bootstrap_games": self.index["dataset_manifest"]["game_count"],
                "bootstrap_focal_examples": self.index["dataset_manifest"]["game_count"] * 20,
                "one_epoch_steps": self.one_epoch_plan["steps"],
                "one_epoch_plan_blake3": self.one_epoch_plan["plan_blake3"],
                "expanded_window_files": False,
                "bootstrap_objective": (
                    "selected-value-only-v1"
                    if self.bootstrap_value_only
                    else "value-plus-greedy-imitation-v1"
                ),
                "bootstrap_policy_loss_weight": 0.0 if self.bootstrap_value_only else 0.1,
            }
        )
        self._session: _PipeSession | None = None
        self._session_key: tuple[Any, ...] | None = None
        self._cached_key: tuple[Any, ...] | None = None
        self._cached_step: R2MapAdapterStep | None = None
        self._closed = False

    def __enter__(self) -> R2MapPackedPipeDatasetAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_session()

    def _packing_plan(self) -> dict[str, Any]:
        stats = _empty_packing_statistics(0)
        for source in self._source_order(0, self.sampler_seed, split="train"):
            widths = [
                width
                for game in self._game_order(source, 0, self.sampler_seed, split="train")
                for width in ((1,) * 20 if self.bootstrap_value_only else self._focal_widths(game))
            ]
            _accumulate_packing_statistics(
                stats,
                widths,
                group_batch_size=self.group_batch_size,
                maximum_candidates_per_batch=self.maximum_candidates_per_batch,
            )
        stats.update(
            {
                "schema_id": "r2-map-focal-seat-one-epoch-packing-v1",
                "dataset_blake3": self.dataset_blake3,
                "seed": self.sampler_seed,
                "epochs": 1,
                "focal_seat_rule": FOCAL_SEAT_RULE,
                "configured_group_batch_size": self.group_batch_size,
                "maximum_candidates_per_batch": self.maximum_candidates_per_batch,
            }
        )
        stats["plan_blake3"] = _canonical_blake3(stats)
        return stats

    @staticmethod
    def _focal_widths(game: dict[str, Any]) -> tuple[int, ...]:
        widths = game.get("candidate_widths")
        if not isinstance(widths, list) or len(widths) != 80:
            raise R2MapDatasetError("packed focal game requires 80 indexed widths")
        focal = int(game["global_game_index"]) % 4
        selected = tuple(int(widths[focal + 4 * turn]) for turn in range(20))
        if any(width <= 0 for width in selected):
            raise R2MapDatasetError("packed focal game has an invalid width")
        return selected

    def _source_order(self, epoch: int, seed: int, *, split: str) -> tuple[str, ...]:
        eligible = [
            source
            for source, games in self._games_by_source.items()
            if any(game["split"] == split for game in games)
        ]
        if split == "validation":
            return tuple(
                sorted(eligible, key=lambda source: self._sources[source]["first_game_index"])
            )
        return tuple(sorted(eligible, key=lambda source: _sampler_hash(seed, epoch, source)))

    def _game_order(
        self, source: str, epoch: int, seed: int, *, split: str
    ) -> tuple[dict[str, Any], ...]:
        games = [game for game in self._games_by_source[source] if game["split"] == split]
        if split == "validation":
            games.sort(key=lambda game: game["global_game_index"])
        else:
            games.sort(key=lambda game: _sampler_hash(seed, epoch, game["game_id"]))
        return tuple(games)

    def initial_state(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed != self.sampler_seed:
            raise R2MapDatasetError("packed-pipe sampler seed differs")
        return self._cursor(0, 0, 0, 0, 0, seed), {
            "seed": seed,
            "sampler_protocol": SAMPLER_PROTOCOL_ID,
        }

    def _cursor(
        self,
        epoch: int,
        source_offset: int,
        game_offset: int,
        turn_offset: int,
        batch_index: int,
        seed: int,
    ) -> dict[str, Any]:
        order = self._source_order(epoch, seed, split="train")
        source = order[source_offset]
        return {
            "epoch": epoch,
            "source_offset": source_offset,
            "source": source,
            "source_blake3": self._sources[source]["blake3"],
            "game_offset": game_offset,
            "turn_offset": turn_offset,
            "batch_index": batch_index,
        }

    def _validate_state(
        self, cursor: dict[str, Any], sampler: dict[str, Any]
    ) -> tuple[int, int, int, int, int, int]:
        if (
            set(sampler) != {"seed", "sampler_protocol"}
            or sampler["sampler_protocol"] != SAMPLER_PROTOCOL_ID
        ):
            raise R2MapDatasetError("packed-pipe sampler state differs")
        expected = {
            "epoch",
            "source_offset",
            "source",
            "source_blake3",
            "game_offset",
            "turn_offset",
            "batch_index",
        }
        if set(cursor) != expected:
            raise R2MapDatasetError("packed-pipe cursor schema differs")
        epoch = int(cursor["epoch"])
        source_offset = int(cursor["source_offset"])
        game_offset = int(cursor["game_offset"])
        turn_offset = int(cursor["turn_offset"])
        batch_index = int(cursor["batch_index"])
        seed = int(sampler["seed"])
        if min(epoch, source_offset, game_offset, turn_offset, batch_index, seed) < 0:
            raise R2MapDatasetError("packed-pipe cursor is negative")
        order = self._source_order(epoch, seed, split="train")
        if source_offset >= len(order):
            raise R2MapDatasetError("packed-pipe source cursor exceeds its epoch")
        source = order[source_offset]
        games = self._game_order(source, epoch, seed, split="train")
        if (
            cursor["source"] != source
            or cursor["source_blake3"] != self._sources[source]["blake3"]
            or game_offset >= len(games)
            or turn_offset >= 20
        ):
            raise R2MapDatasetError("packed-pipe cursor identity differs")
        return epoch, source_offset, game_offset, turn_offset, batch_index, seed

    def training_batch(
        self, cursor: dict[str, Any], sampler_state: dict[str, Any]
    ) -> R2MapAdapterStep:
        state = self._validate_state(cursor, sampler_state)
        key = tuple(state)
        if key == self._cached_key and self._cached_step is not None:
            return self._cached_step
        epoch, source_offset, game_offset, turn_offset, batch_index, seed = state
        source_order = self._source_order(epoch, seed, split="train")
        source = source_order[source_offset]
        games = self._game_order(source, epoch, seed, split="train")
        session_key = ("train", epoch, seed, source_offset)
        if self._session_key != session_key:
            self._close_session()
            self._session = self._start_session(
                source=source,
                mode="train",
                epoch=epoch,
                seed=seed,
                games=games,
                game_offset=game_offset,
                turn_offset=turn_offset,
                batch_index=batch_index,
            )
            self._session_key = session_key
        assert self._session is not None
        produced = self._session.next_batch()
        if produced is None:
            raise R2MapDatasetError("packed producer ended before the indexed source")
        batch, _, control = produced
        if (
            control["batch_index"] != batch_index
            or control["first_game_offset"] != game_offset
            or control["first_turn_offset"] != turn_offset
        ):
            raise R2MapDatasetError("packed producer cursor differs")
        next_game = int(control["next_game_offset"])
        next_turn = int(control["next_turn_offset"])
        next_source = source_offset
        next_epoch = epoch
        if next_game == len(games):
            next_source += 1
            next_game = next_turn = 0
            self._close_session()
            if next_source == len(source_order):
                next_epoch += 1
                next_source = 0
        step = R2MapAdapterStep(
            batch=batch,
            next_cursor=self._cursor(
                next_epoch,
                next_source,
                next_game,
                next_turn,
                batch_index + 1,
                seed,
            ),
            next_sampler_state=dict(sampler_state),
        )
        self._cached_key = key
        self._cached_step = step
        return step

    def _start_session(
        self,
        *,
        source: str,
        mode: str,
        epoch: int,
        seed: int,
        games: Sequence[dict[str, Any]],
        game_offset: int,
        turn_offset: int,
        batch_index: int,
    ) -> _PipeSession:
        command = [
            str(self.exporter),
            "serve-r2-map-packed-batches",
            "--shard",
            str(self.shard_root / source),
            "--mode",
            mode,
            "--epoch",
            str(epoch),
            "--sampler-seed",
            str(seed),
            "--group-batch-size",
            str(self.group_batch_size),
            "--maximum-candidates-per-batch",
            str(self.maximum_candidates_per_batch),
            "--start-game-offset",
            str(game_offset),
            "--start-turn-offset",
            str(turn_offset),
            "--start-batch-index",
            str(batch_index),
            "--validated-aggregate-receipt",
            str(self.aggregate_receipt),
            "--validated-compact-index",
            str(self.index_path),
            "--validated-packing-receipt",
            str(self.packing_receipt),
        ]
        for game in games:
            command.extend(("--game-index", str(game["global_game_index"])))
        if self.bootstrap_value_only:
            command.append("--bootstrap-value-only")
        return _PipeSession(
            command,
            expected_source=self._sources[source],
            expected_mode=mode,
            expected_epoch=epoch,
            expected_seed=seed,
            expected_start=(game_offset, turn_offset, batch_index),
            expected_bootstrap_value_only=self.bootstrap_value_only,
        )

    def _close_session(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None
        self._session_key = None

    def validation_batches(self) -> Iterator[R2MapSupervisedBatch]:
        def produce() -> Iterator[R2MapSupervisedBatch]:
            for source in self._source_order(0, 0, split="validation"):
                games = self._game_order(source, 0, 0, split="validation")
                session = self._start_session(
                    source=source,
                    mode="validation",
                    epoch=0,
                    seed=0,
                    games=games,
                    game_offset=0,
                    turn_offset=0,
                    batch_index=0,
                )
                try:
                    while True:
                        value = session.next_batch()
                        if value is None:
                            break
                        yield value[0]
                finally:
                    session.close()

        return produce()

    def fixed_prediction_batch(self, panel_id: str) -> R2MapBatch:
        if not panel_id:
            raise R2MapDatasetError("packed-pipe fixed panel must be named")
        source = self._source_order(0, 0, split="validation")[0]
        games = self._game_order(source, 0, 0, split="validation")[:1]
        session = self._start_session(
            source=source,
            mode="validation",
            epoch=0,
            seed=0,
            games=games,
            game_offset=0,
            turn_offset=0,
            batch_index=0,
        )
        try:
            value = session.next_batch()
            if value is None:
                raise R2MapDatasetError("packed-pipe fixed panel is empty")
            return value[1]
        finally:
            session.close()
