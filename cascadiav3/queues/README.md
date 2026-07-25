# Experiment queues

Queues are optional convenience files for starting ordinary scripts. They do
not allocate seeds, pin source revisions, seal outputs, create receipts, or
decide whether a result may be used.

Each non-comment line is a JSON object:

```json
{"name":"cycle","script":"cascadiav3/scripts/run_gumbel_selfplay_cycle.sh","env":{"MAX_EXAMPLE_PASSES":0}}
```

Run a queue:

```bash
cascadiav3/scripts/run_experiment_queue.sh path/to/queue.jsonl
```

Stages run in order by default. Set `QUEUE_PARALLEL=1` to start every stage
concurrently. Logs are ordinary files under `cascadiav3/logs/`; rerunning a
queue is allowed.
