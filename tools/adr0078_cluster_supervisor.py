#!/usr/bin/env python3
"""Complete ADR 0078 and conditionally execute ADR 0079 on the local cluster."""

from __future__ import annotations

import argparse
import sys

import adr0078_artifact_handoff as handoff
import adr0078_cluster_runtime as rt
import adr0078_collection as collection
import adr0078_training as training
import adr0079_cluster_handoff as sealed_test


def supervise() -> None:
    descriptor = rt.acquire_lock()
    try:
        rt.log("ADR 0078 cluster supervisor started")
        rt.update_state("starting", supervisor_pid=rt.os.getpid())
        collection.verify_binary_identity(require_remote=False)
        handoff.assert_no_unregistered_local_validation_collector()
        collection.wait_for_collections()
        collection.verify_binary_identity()
        collection.validate_on_producer_hosts()
        collection.sync_validation_to_john1()
        collection.provision_john3_data()
        training.train_on_john3()
        report = training.evaluate_on_john3()
        training.retrieve_run(report)
        rt.log(
            "ADR 0078 validation complete: "
            f"passed={report['passed']} failed_gates={report['failed_gates']}"
        )
        sealed_test.complete_conditional_test(report)
    except Exception as error:
        rt.update_state("failed", error=str(error))
        rt.log(f"FAILED: {error}")
        raise
    finally:
        rt.release_lock(descriptor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify fixed identities and current manifests without waiting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        collection.verify_binary_identity()
        for spec in (rt.TRAIN_SPEC, rt.VALIDATION_SPEC):
            manifest = rt.load_manifest(spec)
            if manifest is not None:
                rt.validate_manifest_contract(manifest, spec, require_complete=False)
                rt.log(
                    f"{spec.label} check: "
                    f"{manifest['completed_games']}/{manifest['requested_games']}"
                )
        return
    supervise()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"adr0078 supervisor failed: {error}", file=sys.stderr)
        raise
