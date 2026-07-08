use std::{
    collections::{BTreeMap, VecDeque},
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::cluster::{self, ClusterResponse};

const HISTORY_SCHEMA_VERSION: u16 = 1;
const SAMPLE_INTERVAL_SECONDS: u64 = 30;
const MINIMUM_RECORD_INTERVAL_MS: u128 = 25_000;
const RETENTION_MS: u128 = 7 * 24 * 60 * 60 * 1_000;
const MAX_POINTS_PER_SERIES: u128 = 480;

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
pub enum ClusterHistoryRange {
    #[default]
    #[serde(rename = "1d")]
    OneDay,
    #[serde(rename = "7d")]
    SevenDays,
}

impl ClusterHistoryRange {
    fn duration_ms(self) -> u128 {
        match self {
            Self::OneDay => 24 * 60 * 60 * 1_000,
            Self::SevenDays => RETENTION_MS,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ClusterHistoryQuery {
    #[serde(default)]
    pub range: ClusterHistoryRange,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterHistoryResponse {
    pub schema_version: u16,
    pub range: ClusterHistoryRange,
    pub generated_at_unix_ms: u128,
    pub start_unix_ms: u128,
    pub end_unix_ms: u128,
    pub oldest_sample_unix_ms: Option<u128>,
    pub newest_sample_unix_ms: Option<u128>,
    pub source_sample_interval_seconds: u64,
    pub bucket_seconds: u64,
    pub raw_sample_count: usize,
    pub series: Vec<ClusterHistorySeries>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterHistorySeries {
    pub node_id: String,
    pub node_label: String,
    pub summary: ClusterHistorySummary,
    pub points: Vec<ClusterHistoryPoint>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ClusterHistorySummary {
    pub observed_samples: usize,
    pub reachable_percent: f64,
    pub average_cpu_percent: Option<f64>,
    pub peak_cpu_percent: Option<f64>,
    pub average_memory_percent: Option<f64>,
    pub peak_memory_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterHistoryPoint {
    pub timestamp_unix_ms: u128,
    pub reachable_percent: f64,
    pub cpu_percent: Option<f64>,
    pub memory_percent: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct StoredSample {
    schema_version: u16,
    timestamp_unix_ms: u128,
    nodes: Vec<StoredNodeSample>,
}

impl StoredSample {
    fn from_cluster(cluster: &ClusterResponse) -> Self {
        Self {
            schema_version: HISTORY_SCHEMA_VERSION,
            timestamp_unix_ms: cluster.collected_at_unix_ms,
            nodes: cluster
                .nodes
                .iter()
                .map(|node| StoredNodeSample {
                    node_id: node.id.to_string(),
                    reachable: node.reachable,
                    cpu_percent: node.reachable.then_some(node.cpu_percent),
                    memory_percent: node.reachable.then_some(node.memory_used_percent),
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct StoredNodeSample {
    node_id: String,
    reachable: bool,
    cpu_percent: Option<f64>,
    memory_percent: Option<f64>,
}

#[derive(Debug, Default)]
struct HistoryState {
    samples: VecDeque<StoredSample>,
    last_compaction_unix_ms: u128,
}

#[derive(Debug)]
pub struct ClusterHistoryStore {
    path: PathBuf,
    state: Mutex<HistoryState>,
}

impl ClusterHistoryStore {
    pub fn open(path: impl Into<PathBuf>, now_unix_ms: u128) -> io::Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        if !path.exists() {
            File::create(&path)?.sync_all()?;
        }

        let (mut samples, recovered_trailing_record) = load_samples(&path)?;
        let original_len = samples.len();
        prune_samples(&mut samples, now_unix_ms);
        if recovered_trailing_record || samples.len() != original_len {
            rewrite_samples(&path, &samples)?;
        }

        Ok(Self {
            path,
            state: Mutex::new(HistoryState {
                samples,
                last_compaction_unix_ms: now_unix_ms,
            }),
        })
    }

    pub fn record(&self, cluster: &ClusterResponse) -> io::Result<bool> {
        let sample = StoredSample::from_cluster(cluster);
        let mut state = self.lock_state()?;
        if state.samples.back().is_some_and(|last| {
            sample.timestamp_unix_ms <= last.timestamp_unix_ms
                || sample.timestamp_unix_ms - last.timestamp_unix_ms < MINIMUM_RECORD_INTERVAL_MS
        }) {
            return Ok(false);
        }

        append_sample(&self.path, &sample)?;
        state.samples.push_back(sample);
        let removed = prune_samples(&mut state.samples, cluster.collected_at_unix_ms);
        if removed > 0
            && cluster
                .collected_at_unix_ms
                .saturating_sub(state.last_compaction_unix_ms)
                >= 6 * 60 * 60 * 1_000
        {
            rewrite_samples(&self.path, &state.samples)?;
            state.last_compaction_unix_ms = cluster.collected_at_unix_ms;
        }
        Ok(true)
    }

    pub fn query(
        &self,
        range: ClusterHistoryRange,
        now_unix_ms: u128,
    ) -> io::Result<ClusterHistoryResponse> {
        let state = self.lock_state()?;
        Ok(build_response(&state.samples, range, now_unix_ms))
    }

    fn lock_state(&self) -> io::Result<MutexGuard<'_, HistoryState>> {
        self.state
            .lock()
            .map_err(|_| io::Error::other("cluster history lock is poisoned"))
    }
}

pub fn spawn_sampler(store: Arc<ClusterHistoryStore>) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(SAMPLE_INTERVAL_SECONDS));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let store = Arc::clone(&store);
            let result = tokio::task::spawn_blocking(move || {
                let sample = cluster::collect_cluster();
                store.record(&sample)
            })
            .await;
            match result {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => eprintln!("cluster history sample could not be stored: {error}"),
                Err(error) => eprintln!("cluster history sampler failed: {error}"),
            }
        }
    });
}

fn build_response(
    samples: &VecDeque<StoredSample>,
    range: ClusterHistoryRange,
    now_unix_ms: u128,
) -> ClusterHistoryResponse {
    let range_ms = range.duration_ms();
    let start_unix_ms = now_unix_ms.saturating_sub(range_ms);
    let bucket_ms = range_ms.div_ceil(MAX_POINTS_PER_SERIES).max(1);
    let in_range: Vec<_> = samples
        .iter()
        .filter(|sample| sample.timestamp_unix_ms >= start_unix_ms)
        .collect();

    let series = cluster::node_id_labels()
        .into_iter()
        .map(|(node_id, node_label)| {
            build_series(node_id, node_label, &in_range, start_unix_ms, bucket_ms)
        })
        .collect();

    ClusterHistoryResponse {
        schema_version: HISTORY_SCHEMA_VERSION,
        range,
        generated_at_unix_ms: now_unix_ms,
        start_unix_ms,
        end_unix_ms: now_unix_ms,
        oldest_sample_unix_ms: samples.front().map(|sample| sample.timestamp_unix_ms),
        newest_sample_unix_ms: samples.back().map(|sample| sample.timestamp_unix_ms),
        source_sample_interval_seconds: SAMPLE_INTERVAL_SECONDS,
        bucket_seconds: u64::try_from(bucket_ms.div_ceil(1_000)).unwrap_or(u64::MAX),
        raw_sample_count: in_range.len(),
        series,
    }
}

fn build_series(
    node_id: &str,
    node_label: &str,
    samples: &[&StoredSample],
    start_unix_ms: u128,
    bucket_ms: u128,
) -> ClusterHistorySeries {
    let mut buckets = BTreeMap::<u128, Bucket>::new();
    let mut summary = SummaryAccumulator::default();

    for sample in samples {
        let Some(node) = sample.nodes.iter().find(|node| node.node_id == node_id) else {
            continue;
        };
        let bucket_index = (sample.timestamp_unix_ms - start_unix_ms) / bucket_ms;
        buckets
            .entry(bucket_index)
            .or_default()
            .push(sample.timestamp_unix_ms, node);
        summary.push(node);
    }

    ClusterHistorySeries {
        node_id: node_id.to_string(),
        node_label: node_label.to_string(),
        summary: summary.finish(),
        points: buckets.into_values().map(Bucket::finish).collect(),
    }
}

#[derive(Debug, Default)]
struct Bucket {
    timestamp_unix_ms: u128,
    samples: usize,
    reachable: usize,
    cpu_sum: f64,
    cpu_samples: usize,
    memory_sum: f64,
    memory_samples: usize,
}

impl Bucket {
    fn push(&mut self, timestamp_unix_ms: u128, node: &StoredNodeSample) {
        self.timestamp_unix_ms = timestamp_unix_ms;
        self.samples += 1;
        self.reachable += usize::from(node.reachable);
        if let Some(cpu) = node.cpu_percent {
            self.cpu_sum += cpu;
            self.cpu_samples += 1;
        }
        if let Some(memory) = node.memory_percent {
            self.memory_sum += memory;
            self.memory_samples += 1;
        }
    }

    fn finish(self) -> ClusterHistoryPoint {
        ClusterHistoryPoint {
            timestamp_unix_ms: self.timestamp_unix_ms,
            reachable_percent: percent(self.reachable, self.samples),
            cpu_percent: mean(self.cpu_sum, self.cpu_samples),
            memory_percent: mean(self.memory_sum, self.memory_samples),
        }
    }
}

#[derive(Debug, Default)]
struct SummaryAccumulator {
    samples: usize,
    reachable: usize,
    cpu_sum: f64,
    cpu_samples: usize,
    cpu_peak: Option<f64>,
    memory_sum: f64,
    memory_samples: usize,
    memory_peak: Option<f64>,
}

impl SummaryAccumulator {
    fn push(&mut self, node: &StoredNodeSample) {
        self.samples += 1;
        self.reachable += usize::from(node.reachable);
        if let Some(cpu) = node.cpu_percent {
            self.cpu_sum += cpu;
            self.cpu_samples += 1;
            self.cpu_peak = Some(self.cpu_peak.map_or(cpu, |peak| peak.max(cpu)));
        }
        if let Some(memory) = node.memory_percent {
            self.memory_sum += memory;
            self.memory_samples += 1;
            self.memory_peak = Some(self.memory_peak.map_or(memory, |peak| peak.max(memory)));
        }
    }

    fn finish(self) -> ClusterHistorySummary {
        ClusterHistorySummary {
            observed_samples: self.samples,
            reachable_percent: percent(self.reachable, self.samples),
            average_cpu_percent: mean(self.cpu_sum, self.cpu_samples),
            peak_cpu_percent: self.cpu_peak.map(round_one),
            average_memory_percent: mean(self.memory_sum, self.memory_samples),
            peak_memory_percent: self.memory_peak.map(round_one),
        }
    }
}

fn mean(sum: f64, count: usize) -> Option<f64> {
    (count > 0).then(|| round_one(sum / count as f64))
}

fn percent(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        round_one(part as f64 / total as f64 * 100.0)
    }
}

fn round_one(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn prune_samples(samples: &mut VecDeque<StoredSample>, now_unix_ms: u128) -> usize {
    let cutoff = now_unix_ms.saturating_sub(RETENTION_MS);
    let before = samples.len();
    while samples
        .front()
        .is_some_and(|sample| sample.timestamp_unix_ms < cutoff)
    {
        samples.pop_front();
    }
    before - samples.len()
}

fn append_sample(path: &Path, sample: &StoredSample) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, sample).map_err(invalid_data)?;
    file.write_all(b"\n")?;
    file.sync_data()
}

fn rewrite_samples(path: &Path, samples: &VecDeque<StoredSample>) -> io::Result<()> {
    let temp = path.with_extension("jsonl.tmp");
    {
        let mut file = File::create(&temp)?;
        for sample in samples {
            serde_json::to_writer(&mut file, sample).map_err(invalid_data)?;
            file.write_all(b"\n")?;
        }
        file.sync_all()?;
    }
    fs::rename(temp, path)
}

fn load_samples(path: &Path) -> io::Result<(VecDeque<StoredSample>, bool)> {
    let contents = fs::read_to_string(path)?;
    let has_trailing_newline = contents.is_empty() || contents.ends_with('\n');
    let lines: Vec<_> = contents.lines().collect();
    let mut samples = VecDeque::with_capacity(lines.len());
    let mut recovered_trailing_record = false;

    for (index, line) in lines.iter().enumerate() {
        let sample: StoredSample = match serde_json::from_str(line) {
            Ok(sample) => sample,
            Err(_) if index + 1 == lines.len() && !has_trailing_newline => {
                recovered_trailing_record = true;
                break;
            }
            Err(error) => return Err(invalid_data(error)),
        };
        if sample.schema_version != HISTORY_SCHEMA_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported cluster history schema {}",
                    sample.schema_version
                ),
            ));
        }
        if samples.back().is_some_and(|previous: &StoredSample| {
            sample.timestamp_unix_ms <= previous.timestamp_unix_ms
        }) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "cluster history timestamps must be strictly increasing",
            ));
        }
        samples.push_back(sample);
    }
    Ok((samples, recovered_trailing_record))
}

fn invalid_data(error: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    fn temp_history_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "cascadia-cluster-history-{name}-{}-{}.jsonl",
            std::process::id(),
            NEXT_PATH.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn response(timestamp_unix_ms: u128, cpu: f64, memory: f64) -> ClusterResponse {
        cluster::test_cluster_response(timestamp_unix_ms, cpu, memory)
    }

    #[test]
    fn records_deduplicates_and_reloads_samples() {
        let path = temp_history_path("roundtrip");
        let store = ClusterHistoryStore::open(&path, 1_000_000).unwrap();
        assert!(store.record(&response(1_000_000, 10.0, 20.0)).unwrap());
        assert!(!store.record(&response(1_005_000, 90.0, 90.0)).unwrap());
        assert!(store.record(&response(1_030_000, 30.0, 40.0)).unwrap());
        drop(store);

        let reloaded = ClusterHistoryStore::open(&path, 1_030_000).unwrap();
        let history = reloaded
            .query(ClusterHistoryRange::OneDay, 1_030_000)
            .unwrap();
        assert_eq!(history.raw_sample_count, 2);
        assert_eq!(history.series[0].summary.average_cpu_percent, Some(20.0));
        assert_eq!(history.series[0].summary.average_memory_percent, Some(30.0));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn query_downsamples_and_retains_node_outages() {
        let path = temp_history_path("downsample");
        let store = ClusterHistoryStore::open(&path, 100_000_000).unwrap();
        for index in 0..600 {
            let timestamp = 100_000_000 + index * 30_000;
            let mut sample = response(timestamp, index as f64 % 100.0, 50.0);
            if index == 599 {
                sample.nodes[1].reachable = false;
            }
            store.record(&sample).unwrap();
        }

        let history = store
            .query(ClusterHistoryRange::OneDay, 118_000_000)
            .unwrap();
        assert!(
            history
                .series
                .iter()
                .all(|series| series.points.len() <= 480)
        );
        assert!(history.series[1].summary.reachable_percent < 100.0);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn open_recovers_only_an_incomplete_trailing_record() {
        let path = temp_history_path("trailing");
        let valid = serde_json::to_string(&StoredSample::from_cluster(&response(
            2_000_000, 10.0, 20.0,
        )))
        .unwrap();
        fs::write(&path, format!("{valid}\n{{\"schema_version\":")).unwrap();

        let store = ClusterHistoryStore::open(&path, 2_000_000).unwrap();
        assert_eq!(
            store
                .query(ClusterHistoryRange::SevenDays, 2_000_000)
                .unwrap()
                .raw_sample_count,
            1
        );
        assert!(fs::read_to_string(&path).unwrap().ends_with('\n'));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn open_rejects_corruption_before_the_trailing_record() {
        let path = temp_history_path("corrupt");
        fs::write(&path, "{}\n{}\n").unwrap();
        let error = ClusterHistoryStore::open(&path, 2_000_000).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        fs::remove_file(path).unwrap();
    }
}
