"""
Streaming Inference Pipeline — E-Commerce Fraud Detection
=========================================================

Three coordinated services that run as independent processes (or containers):

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Producer              Inference Consumer       Analytics Consumer  │
  │  ───────────        ────────────────      ──────────────── │
  │  Reads raw CSV    →    Consumes 'purchases'  →  Consumes            │
  │  sorted by             Computes rolling            'fraud_scores'   │
  │  purchase_time         features (1h / 24h)     Aggregates hourly    │
  │  Publishes to          Scores with MLflow        alert rates        │
  │  'purchases'           model                   Writes CSV sink      │
  │  topic                 Publishes to                                 │
  │                        'fraud_scores'                               │
  └─────────────────────────────────────────────────────────────────────┘

Usage (run each in a separate terminal or Docker container):

  # 1. Start Kafka & Zookeeper first (via docker-compose)
  #    docker-compose up -d zookeeper kafka

  # 2. Run producer
  python pipelines/streaming_inference_pipeline.py producer

  # 3. Run inference consumer
  python pipelines/streaming_inference_pipeline.py inference

  # 4. Run analytics consumer
  python pipelines/streaming_inference_pipeline.py analytics

  # Or run all three in one process (demo / local dev):
  python pipelines/streaming_inference_pipeline.py all

Environment variables (override via .env or docker-compose):
  KAFKA_BOOTSTRAP_SERVERS   default: localhost:9092
  MLFLOW_TRACKING_URI       default: file:./mlruns
  STREAMING_DELAY_SECS      seconds between producer events (default: 0.05)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import signal
import sqlite3
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Path bootstrap 
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from config import get_mlflow_config

# Optional Kafka import (graceful fallback for local testing)
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import NoBrokersAvailable, KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

#  Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("streaming")

# Configuration 

def _load_streaming_config() -> Dict[str, Any]:
    """Load the `streaming` block from config.yaml with safe defaults."""
    import yaml
    cfg_path = os.path.join(ROOT, "config", "config.yaml")
    try:
        with open(cfg_path) as fh:
            full = yaml.safe_load(fh)
        return full.get("streaming", {})
    except Exception as exc:
        logger.warning(f"Could not read streaming config ({exc}), using defaults.")
        return {}


_STREAMING_CFG = _load_streaming_config()

# Kafka
KAFKA_BOOTSTRAP = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    _STREAMING_CFG.get("kafka_bootstrap_servers", "localhost:9092"),
)
TOPIC_PURCHASES  = _STREAMING_CFG.get("topic_purchases",  "purchases")
TOPIC_SCORES     = _STREAMING_CFG.get("topic_fraud_scores", "fraud_scores")
CONSUMER_GROUP   = _STREAMING_CFG.get("consumer_group",   "fraud_inference_group")
ANALYTICS_GROUP  = _STREAMING_CFG.get("analytics_group",  "fraud_analytics_group")

# Sliding-window durations (seconds)
WINDOW_1H  = _STREAMING_CFG.get("window_1h_secs",  3_600)
WINDOW_24H = _STREAMING_CFG.get("window_24h_secs", 86_400)

# Producer
RAW_DATA_PATH  = _STREAMING_CFG.get("raw_data_path", os.path.join(ROOT, "data", "raw", "Fraud_Data.csv"))
PRODUCER_DELAY = float(os.getenv("STREAMING_DELAY_SECS", str(_STREAMING_CFG.get("delay_secs", 0.05))))
MAX_EVENTS     = _STREAMING_CFG.get("max_events", None)   # None = replay entire file

# Analytics sink
ANALYTICS_DIR  = _STREAMING_CFG.get("analytics_dir", os.path.join(ROOT, "analytics"))
SQLITE_DB_PATH = _STREAMING_CFG.get("sqlite_db",    os.path.join(ROOT, "analytics", "fraud_analytics.db"))
ALERT_RATE_THR = _STREAMING_CFG.get("alert_rate_threshold", 0.15)   # flag hour if fraud rate > 15 %

# MLflow
_MLFLOW_CFG     = get_mlflow_config()
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", _MLFLOW_CFG.get("tracking_uri", "file:./mlruns"))
MODEL_REGISTRY  = _MLFLOW_CFG.get("model_registry_name", "fraud_detection")

# Graceful shutdown 
_SHUTDOWN = threading.Event()

def _handle_signal(sig, frame):               # noqa: ANN001
    logger.info(f"Signal {sig} received — initiating graceful shutdown …")
    _SHUTDOWN.set()

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# =============================================================================
# Model Loader
# =============================================================================

class FraudModelLoader:
    """
    Loads the latest registered model from the MLflow Model Registry
    and exposes predict_proba(). Falls back to a stub (all 0.0) if no
    model is found, so the pipeline can be tested without training first.
    """

    def __init__(self) -> None:
        self._model      = None
        self._threshold  = 0.5
        self._model_type = "stub"
        self._meta       = {}

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Try to load model from MLflow registry, then local meta JSON."""
        meta_path = os.path.join(ROOT, "artifacts", "models", "best_model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                self._meta = json.load(fh)
            self._threshold  = float(self._meta.get("threshold", 0.5))
            self._model_type = self._meta.get("model_type", "unknown")

        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            uri = f"models:/{MODEL_REGISTRY}/latest"
            self._model = mlflow.pyfunc.load_model(uri)
            logger.info(f"  Loaded '{MODEL_REGISTRY}/latest' from MLflow  "
                        f"(type={self._model_type}, threshold={self._threshold:.2f})")
        except Exception as exc:
            logger.warning(f"  MLflow model not available ({exc}). Using stub predictor.")
            self._model = None

    # ------------------------------------------------------------------
    def predict_proba(self, feature_row: np.ndarray) -> float:
        """
        Score a single feature row and return the fraud probability.

        Args:
            feature_row: 1-D numpy array of feature values

        Returns:
            Fraud probability in [0, 1]
        """
        if self._model is None:
            # Stub: return a random probability for integration testing
            return float(np.random.beta(2, 18))       # skews toward low fraud ~10%

        try:
            X = feature_row.reshape(1, -1)
            raw = self._model.predict(pd.DataFrame(X))
            # pyfunc may return probabilities or class labels
            val = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
            return min(max(val, 0.0), 1.0)
        except Exception as exc:
            logger.error(f"Scoring error: {exc}")
            return 0.0

    @property
    def threshold(self) -> float:
        return self._threshold


# =============================================================================
# Rolling Feature Store (in-memory)
# =============================================================================

class RollingFeatureStore:
    """
    Maintains per-device and per-IP rolling event windows for computing
    velocity features in-stream without a full state backend.

    Structure:
        _device_events[device_id] = deque of (unix_ts, purchase_value)
        _ip_events[ip_address]    = deque of (unix_ts, purchase_value)

    Thread-safe via a simple lock.
    """

    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._device_events: Dict[str, deque] = defaultdict(deque)
        self._ip_events:     Dict[str, deque] = defaultdict(deque)

    # ------------------------------------------------------------------
    def record(self, event: Dict[str, Any]) -> None:
        """Append a new event to the rolling store."""
        ts    = float(event.get("purchase_time_ts", time.time()))
        value = float(event.get("purchase_value", 0.0))
        dev   = str(event.get("device_id", ""))
        ip    = str(event.get("ip_address", ""))

        with self._lock:
            if dev:
                self._device_events[dev].append((ts, value))
            if ip:
                self._ip_events[ip].append((ts, value))

    # ------------------------------------------------------------------
    def get_features(self, event: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute leakage-safe rolling features for the given event.
        Uses only events that occurred BEFORE this event's timestamp
        (right-closed windows).

        Returns dict with keys:
          device_txn_count_1h, device_txn_count_24h,
          device_amount_sum_1h, device_amount_sum_24h,
          ip_txn_count_1h, ip_txn_count_24h,
          ip_amount_sum_1h, ip_amount_sum_24h,
          device_rarity, ip_rarity
        """
        ts  = float(event.get("purchase_time_ts", time.time()))
        dev = str(event.get("device_id", ""))
        ip  = str(event.get("ip_address", ""))

        with self._lock:
            d_events = list(self._device_events.get(dev, []))
            i_events = list(self._ip_events.get(ip, []))

        def _window_stats(events: List[Tuple[float, float]], window_secs: float):
            cut = ts - window_secs
            # exclude current event (ts < cut is wrong — we want past events only)
            past = [(t, v) for t, v in events if cut <= t < ts]
            return len(past), sum(v for _, v in past)

        d_cnt_1h,  d_sum_1h  = _window_stats(d_events, WINDOW_1H)
        d_cnt_24h, d_sum_24h = _window_stats(d_events, WINDOW_24H)
        i_cnt_1h,  i_sum_1h  = _window_stats(i_events, WINDOW_1H)
        i_cnt_24h, i_sum_24h = _window_stats(i_events, WINDOW_24H)

        # Rarity: 1 / (1 + count) — infrequent device/IP → higher score
        d_rarity = 1.0 / (1.0 + d_cnt_24h)
        i_rarity = 1.0 / (1.0 + i_cnt_24h)

        return {
            "device_txn_count_1h":   d_cnt_1h,
            "device_txn_count_24h":  d_cnt_24h,
            "device_amount_sum_1h":  d_sum_1h,
            "device_amount_sum_24h": d_sum_24h,
            "ip_txn_count_1h":       i_cnt_1h,
            "ip_txn_count_24h":      i_cnt_24h,
            "ip_amount_sum_1h":      i_sum_1h,
            "ip_amount_sum_24h":     i_sum_24h,
            "device_rarity":         d_rarity,
            "ip_rarity":             i_rarity,
        }

    # ------------------------------------------------------------------
    def evict_old(self, cutoff_secs: float = WINDOW_24H * 2) -> None:
        """Drop events older than 2×24h to prevent unbounded memory growth."""
        cutoff = time.time() - cutoff_secs
        with self._lock:
            for dq in self._device_events.values():
                while dq and dq[0][0] < cutoff:
                    dq.popleft()
            for dq in self._ip_events.values():
                while dq and dq[0][0] < cutoff:
                    dq.popleft()


# =============================================================================
# Component 1 — Producer
# =============================================================================

class TransactionProducer:
    """
    Reads the raw Fraud_Data.csv, sorts events by purchase_time (ascending),
    and publishes them to the Kafka 'purchases' topic at a configurable rate.

    Each Kafka message value is a UTF-8 JSON blob of the raw row plus a
    computed `purchase_time_ts` Unix timestamp field used by the consumer
    for windowing.
    """

    def __init__(self) -> None:
        self._producer: Optional[Any] = None

    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        if not KAFKA_AVAILABLE:
            logger.error("kafka-python is not installed. Run: pip install kafka-python")
            return False
        try:
            self._producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8") if k else None,
                acks="all",
                retries=5,
                linger_ms=10,
                batch_size=16_384,
            )
            logger.info(f"Producer connected → {KAFKA_BOOTSTRAP}  topic={TOPIC_PURCHASES}")
            return True
        except NoBrokersAvailable:
            logger.error(f"No Kafka brokers available at '{KAFKA_BOOTSTRAP}'. "
                         "Is docker-compose running?")
            return False
        except Exception as exc:
            logger.error(f"Producer connection failed: {exc}")
            return False

    # ------------------------------------------------------------------
    def _parse_ts(self, ts_str: str) -> float:
        """Parse a purchase_time string into a Unix timestamp (float)."""
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(ts_str).strip(), fmt).replace(
                    tzinfo=timezone.utc
                ).timestamp()
            except ValueError:
                continue
        return float(time.time())

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main producer loop — publishes events until file is exhausted or shutdown."""
        logger.info("=" * 70)
        logger.info("PRODUCER — starting")
        logger.info(f"  Source  : {RAW_DATA_PATH}")
        logger.info(f"  Topic   : {TOPIC_PURCHASES}")
        logger.info(f"  Delay   : {PRODUCER_DELAY}s per event")
        logger.info("=" * 70)

        if not os.path.exists(RAW_DATA_PATH):
            logger.error(f"Raw data not found: {RAW_DATA_PATH}. Cannot start producer.")
            return

        if not self._connect():
            return

        # Load & sort by purchase_time
        df = pd.read_csv(RAW_DATA_PATH)
        if "purchase_time" in df.columns:
            df = df.sort_values("purchase_time", ascending=True).reset_index(drop=True)
        logger.info(f"  Loaded {len(df):,} transactions sorted by purchase_time")

        if MAX_EVENTS:
            df = df.head(MAX_EVENTS)
            logger.info(f"  Capped at {MAX_EVENTS:,} events (max_events setting)")

        sent = 0
        errors = 0

        for _, row in df.iterrows():
            if _SHUTDOWN.is_set():
                break

            record = row.where(pd.notnull(row), None).to_dict()

            # Add Unix timestamp for windowing
            if "purchase_time" in record:
                record["purchase_time_ts"] = self._parse_ts(str(record["purchase_time"]))
            else:
                record["purchase_time_ts"] = time.time()

            key = record.get("device_id") or record.get("user_id")

            try:
                self._producer.send(TOPIC_PURCHASES, value=record, key=key)
                sent += 1

                if sent % 500 == 0:
                    self._producer.flush()
                    logger.info(f"  Sent {sent:,} events  |  errors={errors}")

            except Exception as exc:
                errors += 1
                logger.warning(f"  Publish error (event #{sent}): {exc}")
                if errors > 50:
                    logger.error("Too many publish errors — aborting producer.")
                    break

            time.sleep(PRODUCER_DELAY)

        # Flush remaining
        try:
            self._producer.flush(timeout=10)
        except Exception:
            pass

        logger.info(f"Producer done. Total sent={sent:,}  errors={errors}")
        self._producer.close()


# =============================================================================
# Component 2 — Inference Consumer
# =============================================================================

class InferenceConsumer:
    """
    Consumes raw transaction events from the 'purchases' topic, computes
    rolling velocity features (device / IP, 1h & 24h windows), scores each
    event with the MLflow model, and publishes the enriched, scored event
    to the 'fraud_scores' topic.
    """

    def __init__(self) -> None:
        self._consumer: Optional[Any] = None
        self._producer: Optional[Any] = None
        self._store  = RollingFeatureStore()
        self._model  = FraudModelLoader()
        self._stats  = {"processed": 0, "fraud_flagged": 0, "errors": 0}
        self._evict_timer = 0.0

    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        if not KAFKA_AVAILABLE:
            logger.error("kafka-python not installed.")
            return False
        try:
            self._consumer = KafkaConsumer(
                TOPIC_PURCHASES,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=CONSUMER_GROUP,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                consumer_timeout_ms=5_000,
            )
            self._producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks=1,
                linger_ms=5,
            )
            logger.info(f"Inference consumer connected  "
                        f"in={TOPIC_PURCHASES}  out={TOPIC_SCORES}  "
                        f"group={CONSUMER_GROUP}")
            return True
        except NoBrokersAvailable:
            logger.error(f"No Kafka brokers at '{KAFKA_BOOTSTRAP}'.")
            return False
        except Exception as exc:
            logger.error(f"Inference consumer connection failed: {exc}")
            return False

    # ------------------------------------------------------------------
    def _build_feature_vector(self, event: Dict, rolling: Dict) -> np.ndarray:
        """
        Assemble a numeric feature vector from the raw event + rolling features.
        Order must match the training feature order.  Fields missing from the
        event are substituted with 0.
        """
        # Core numeric fields that survived preprocessing
        purchase_value     = float(event.get("purchase_value",     0) or 0)
        age                = float(event.get("age",                 0) or 0)
        ip_address         = float(event.get("ip_address",         0) or 0)
        time_to_purchase   = float(event.get("time_to_purchase_secs", 0) or 0)
        purchase_hour      = float(event.get("purchase_hour",      0) or 0)
        purchase_day_of_week = float(event.get("purchase_day_of_week", 0) or 0)

        # Compute time features if raw timestamps exist but derived ones don't
        if time_to_purchase == 0 and event.get("purchase_time_ts") and event.get("signup_time"):
            try:
                signup_ts = datetime.strptime(
                    str(event["signup_time"]), "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc).timestamp()
                time_to_purchase = event["purchase_time_ts"] - signup_ts
            except Exception:
                pass

        if purchase_hour == 0 and event.get("purchase_time_ts"):
            dt = datetime.fromtimestamp(event["purchase_time_ts"], tz=timezone.utc)
            purchase_hour      = float(dt.hour)
            purchase_day_of_week = float(dt.weekday())

        vec = np.array([
            purchase_value,
            age,
            ip_address,
            time_to_purchase,
            purchase_hour,
            purchase_day_of_week,
            # Rolling features
            rolling["device_txn_count_1h"],
            rolling["device_txn_count_24h"],
            rolling["device_amount_sum_1h"],
            rolling["device_amount_sum_24h"],
            rolling["ip_txn_count_1h"],
            rolling["ip_txn_count_24h"],
            rolling["ip_amount_sum_1h"],
            rolling["ip_amount_sum_24h"],
            rolling["device_rarity"],
            rolling["ip_rarity"],
        ], dtype=float)

        return np.nan_to_num(vec, nan=0.0)

    # ------------------------------------------------------------------
    def _score_event(self, event: Dict) -> Dict:
        """Score a single event and return the enriched scored event dict."""
        # 1. Record in rolling store BEFORE computing features to maintain
        #    right-exclusive windows (current event not counted in its own window)
        rolling = self._store.get_features(event)
        self._store.record(event)

        # 2. Build feature vector
        features = self._build_feature_vector(event, rolling)

        # 3. Model inference
        fraud_prob  = self._model.predict_proba(features)
        is_fraud    = int(fraud_prob >= self._model.threshold)

        # 4. Build scored event
        scored = {
            "user_id":           event.get("user_id"),
            "device_id":         event.get("device_id"),
            "ip_address":        event.get("ip_address"),
            "purchase_time":     event.get("purchase_time"),
            "purchase_time_ts":  event.get("purchase_time_ts"),
            "purchase_value":    event.get("purchase_value"),
            "source":            event.get("source"),
            "browser":           event.get("browser"),
            "sex":               event.get("sex"),
            "age":               event.get("age"),
            "true_label":        event.get("class"),  # available in historical replay
            # Inference outputs
            "fraud_probability": round(fraud_prob, 6),
            "fraud_decision":    is_fraud,
            "threshold_used":    round(self._model.threshold, 4),
            # Rolling features (for analytics)
            **{f"rf_{k}": round(v, 4) for k, v in rolling.items()},
            # Metadata
            "scored_at":         datetime.now(timezone.utc).isoformat(),
            "model_registry":    MODEL_REGISTRY,
        }
        return scored

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main consumer loop."""
        logger.info("=" * 70)
        logger.info("INFERENCE CONSUMER — starting")
        logger.info(f"  Consuming: {TOPIC_PURCHASES}")
        logger.info(f"  Emitting : {TOPIC_SCORES}")
        logger.info("=" * 70)

        self._model.load()

        if not self._connect():
            return

        while not _SHUTDOWN.is_set():
            try:
                for msg in self._consumer:
                    if _SHUTDOWN.is_set():
                        break

                    event = msg.value
                    try:
                        scored = self._score_event(event)
                        self._producer.send(TOPIC_SCORES, value=scored)

                        self._stats["processed"] += 1
                        if scored["fraud_decision"]:
                            self._stats["fraud_flagged"] += 1

                        if self._stats["processed"] % 200 == 0:
                            rate = (
                                self._stats["fraud_flagged"]
                                / max(self._stats["processed"], 1)
                            )
                            logger.info(
                                f"[inference] processed={self._stats['processed']:,}  "
                                f"flagged={self._stats['fraud_flagged']:,}  "
                                f"fraud_rate={rate:.3f}"
                            )

                        # Periodic eviction of stale feature state
                        now = time.time()
                        if now - self._evict_timer > 3600:
                            self._store.evict_old()
                            self._evict_timer = now

                    except Exception as exc:
                        self._stats["errors"] += 1
                        logger.error(f"[inference] Error scoring event: {exc}", exc_info=False)

            except Exception as exc:
                if not _SHUTDOWN.is_set():
                    logger.warning(f"[inference] Consumer poll error: {exc} — retrying in 5s")
                    time.sleep(5)

        # Cleanup
        try:
            self._producer.flush(timeout=5)
            self._consumer.close()
            self._producer.close()
        except Exception:
            pass

        logger.info(
            f"Inference consumer stopped.  "
            f"processed={self._stats['processed']:,}  "
            f"flagged={self._stats['fraud_flagged']:,}  "
            f"errors={self._stats['errors']}"
        )


# =============================================================================
# Component 3 — Analytics / BI Consumer
# =============================================================================

class HourlyWindow:
    """Accumulates scored events within a UTC hour bucket for aggregation."""

    def __init__(self, hour_key: str) -> None:
        self.hour_key     = hour_key          # "YYYY-MM-DD HH"
        self.total        = 0
        self.fraud_count  = 0
        self.total_amount = 0.0
        self.top_devices: Dict[str, int] = defaultdict(int)
        self.top_ips:     Dict[str, int] = defaultdict(int)

    def ingest(self, event: Dict) -> None:
        self.total        += 1
        self.fraud_count  += int(event.get("fraud_decision", 0))
        self.total_amount += float(event.get("purchase_value") or 0)
        dev = event.get("device_id")
        ip  = event.get("ip_address")
        if dev:
            self.top_devices[str(dev)] += int(event.get("fraud_decision", 0))
        if ip:
            self.top_ips[str(ip)] += int(event.get("fraud_decision", 0))

    def to_dict(self) -> Dict:
        fraud_rate    = self.fraud_count / max(self.total, 1)
        top_dev_5     = sorted(self.top_devices.items(), key=lambda x: -x[1])[:5]
        top_ip_5      = sorted(self.top_ips.items(),     key=lambda x: -x[1])[:5]
        return {
            "hour_key":      self.hour_key,
            "total_txns":    self.total,
            "fraud_count":   self.fraud_count,
            "fraud_rate":    round(fraud_rate, 4),
            "total_amount":  round(self.total_amount, 2),
            "high_alert":    int(fraud_rate > ALERT_RATE_THR),
            "top_suspicious_devices": dict(top_dev_5),
            "top_suspicious_ips":     dict(top_ip_5),
        }


class AnalyticsConsumer:
    """
    Consumes scored events from 'fraud_scores', aggregates them into
    hourly windows, and writes to:
      - CSV  : analytics/recent_predictions_<timestamp>.csv
      - SQLite: analytics/fraud_analytics.db  (table: fraud_hourly_stats)
    """

    def __init__(self) -> None:
        self._consumer:   Optional[Any]           = None
        self._windows:    Dict[str, HourlyWindow] = {}
        self._all_scored: List[Dict]              = []
        self._stats       = {"consumed": 0, "errors": 0}
        self._flush_every = 100            # flush CSV/DB every N events

    # ------------------------------------------------------------------
    def _connect(self) -> bool:
        if not KAFKA_AVAILABLE:
            logger.error("kafka-python not installed.")
            return False
        try:
            self._consumer = KafkaConsumer(
                TOPIC_SCORES,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=ANALYTICS_GROUP,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                consumer_timeout_ms=5_000,
            )
            logger.info(f"Analytics consumer connected  topic={TOPIC_SCORES}  "
                        f"group={ANALYTICS_GROUP}")
            return True
        except NoBrokersAvailable:
            logger.error(f"No Kafka brokers at '{KAFKA_BOOTSTRAP}'.")
            return False
        except Exception as exc:
            logger.error(f"Analytics consumer connection failed: {exc}")
            return False

    # ------------------------------------------------------------------
    def _hour_key(self, event: Dict) -> str:
        ts = event.get("purchase_time_ts")
        if ts:
            try:
                dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                return dt.strftime("%Y-%m-%d %H")
            except Exception:
                pass
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H")

    # ------------------------------------------------------------------
    def _flush_csv(self) -> None:
        """Write all scored events accumulated so far to a timestamped CSV."""
        if not self._all_scored:
            return

        os.makedirs(ANALYTICS_DIR, exist_ok=True)
        ts_tag  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        csv_path = os.path.join(ANALYTICS_DIR, f"recent_predictions_{ts_tag}.csv")

        try:
            cols = list(self._all_scored[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self._all_scored)
            logger.info(f"[analytics] CSV flushed → {csv_path}  ({len(self._all_scored):,} rows)")
        except Exception as exc:
            logger.error(f"[analytics] CSV flush error: {exc}")

    # ------------------------------------------------------------------
    def _flush_sqlite(self) -> None:
        """Upsert hourly aggregates into the SQLite analytics DB."""
        if not self._windows:
            return

        os.makedirs(ANALYTICS_DIR, exist_ok=True)

        try:
            con = sqlite3.connect(SQLITE_DB_PATH)
            cur = con.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fraud_hourly_stats (
                    hour_key               TEXT PRIMARY KEY,
                    total_txns             INTEGER,
                    fraud_count            INTEGER,
                    fraud_rate             REAL,
                    total_amount           REAL,
                    high_alert             INTEGER,
                    top_suspicious_devices TEXT,
                    top_suspicious_ips     TEXT,
                    updated_at             TEXT
                )
            """)

            for hour_key, window in self._windows.items():
                row = window.to_dict()
                cur.execute("""
                    INSERT INTO fraud_hourly_stats
                        (hour_key, total_txns, fraud_count, fraud_rate, total_amount,
                         high_alert, top_suspicious_devices, top_suspicious_ips, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(hour_key) DO UPDATE SET
                        total_txns=excluded.total_txns,
                        fraud_count=excluded.fraud_count,
                        fraud_rate=excluded.fraud_rate,
                        total_amount=excluded.total_amount,
                        high_alert=excluded.high_alert,
                        top_suspicious_devices=excluded.top_suspicious_devices,
                        top_suspicious_ips=excluded.top_suspicious_ips,
                        updated_at=excluded.updated_at
                """, (
                    row["hour_key"],
                    row["total_txns"],
                    row["fraud_count"],
                    row["fraud_rate"],
                    row["total_amount"],
                    row["high_alert"],
                    json.dumps(row["top_suspicious_devices"]),
                    json.dumps(row["top_suspicious_ips"]),
                    datetime.now(timezone.utc).isoformat(),
                ))

            con.commit()
            con.close()
            logger.info(f"[analytics] SQLite flushed → {SQLITE_DB_PATH}  "
                        f"({len(self._windows)} hour buckets)")
        except Exception as exc:
            logger.error(f"[analytics] SQLite flush error: {exc}")

    # ------------------------------------------------------------------
    def _log_summary(self) -> None:
        """Print a human-readable summary of the current hour windows."""
        if not self._windows:
            return
        rows = sorted(
            [w.to_dict() for w in self._windows.values()],
            key=lambda r: r["hour_key"],
        )
        logger.info(f"\n{'─'*70}")
        logger.info(f"{'Hour (UTC)':<17} {'Txns':>6} {'Fraud':>6} {'Rate%':>7} {'Amount':>12} {'Alert':>6}")
        logger.info(f"{'─'*70}")
        for r in rows[-12:]:          # show last 12 hours
            alert_flag = "⚠ HIGH" if r["high_alert"] else ""
            logger.info(
                f"{r['hour_key']:<17} "
                f"{r['total_txns']:>6} "
                f"{r['fraud_count']:>6} "
                f"{r['fraud_rate']*100:>6.1f}% "
                f"${r['total_amount']:>11,.2f} "
                f"{alert_flag}"
            )
        logger.info(f"{'─'*70}\n")

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main analytics consumer loop."""
        logger.info("=" * 70)
        logger.info("ANALYTICS CONSUMER — starting")
        logger.info(f"  Consuming : {TOPIC_SCORES}")
        logger.info(f"  CSV sink  : {ANALYTICS_DIR}/recent_predictions_*.csv")
        logger.info(f"  SQLite    : {SQLITE_DB_PATH}")
        logger.info("=" * 70)

        if not self._connect():
            return

        last_summary = time.time()

        while not _SHUTDOWN.is_set():
            try:
                for msg in self._consumer:
                    if _SHUTDOWN.is_set():
                        break

                    event = msg.value
                    try:
                        hour_key = self._hour_key(event)
                        if hour_key not in self._windows:
                            self._windows[hour_key] = HourlyWindow(hour_key)
                        self._windows[hour_key].ingest(event)
                        self._all_scored.append(event)
                        self._stats["consumed"] += 1

                        # Periodic flushing
                        if self._stats["consumed"] % self._flush_every == 0:
                            self._flush_sqlite()
                            self._flush_csv()
                            self._all_scored.clear()   # clear to avoid unbounded RAM

                        # Periodic table summary log
                        now = time.time()
                        if now - last_summary > 60:
                            self._log_summary()
                            last_summary = now

                    except Exception as exc:
                        self._stats["errors"] += 1
                        logger.error(f"[analytics] Event processing error: {exc}")

            except Exception as exc:
                if not _SHUTDOWN.is_set():
                    logger.warning(f"[analytics] Poll error: {exc} — retrying in 5s")
                    time.sleep(5)

        # Final flush on shutdown
        self._flush_sqlite()
        self._flush_csv()
        self._log_summary()

        try:
            self._consumer.close()
        except Exception:
            pass

        logger.info(
            f"Analytics consumer stopped.  "
            f"consumed={self._stats['consumed']:,}  "
            f"errors={self._stats['errors']}"
        )


# =============================================================================
# Demo mode — in-process queue (no Kafka required)
# =============================================================================

class _InMemoryQueue:
    """Thread-safe in-memory queue that mimics a Kafka topic for local demos."""

    def __init__(self) -> None:
        self._q: deque = deque()
        self._lock = threading.Lock()

    def put(self, item: Any) -> None:
        with self._lock:
            self._q.append(item)

    def get(self, timeout: float = 1.0) -> Optional[Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._q:
                    return self._q.popleft()
            time.sleep(0.01)
        return None

    def __len__(self) -> int:
        return len(self._q)


def run_local_demo(max_events: int = 200) -> None:
    """
    Run all three pipeline stages in-process using in-memory queues.
    No Kafka required.  Useful for unit testing and CI validation.

    Args:
        max_events: Number of raw transactions to replay (default 200).
    """
    logger.info("=" * 70)
    logger.info("LOCAL DEMO MODE — no Kafka required")
    logger.info(f"  max_events={max_events}")
    logger.info("=" * 70)

    if not os.path.exists(RAW_DATA_PATH):
        logger.error(f"Raw data not found: {RAW_DATA_PATH}")
        return

    purchases_q  = _InMemoryQueue()
    scores_q     = _InMemoryQueue()

    # Load model 
    model_loader = FraudModelLoader()
    model_loader.load()

    store = RollingFeatureStore()
    consumer = InferenceConsumer()
    consumer._store = store
    consumer._model = model_loader

    analytics = AnalyticsConsumer()

    # Load & replay data    
    df = pd.read_csv(RAW_DATA_PATH)
    if "purchase_time" in df.columns:
        df = df.sort_values("purchase_time").reset_index(drop=True)
    df = df.head(max_events)
    logger.info(f"Loaded {len(df):,} rows for demo replay")

    def _parse_ts(ts_str: str) -> float:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(ts_str).strip(), fmt).replace(
                    tzinfo=timezone.utc
                ).timestamp()
            except ValueError:
                continue
        return time.time()

    total_fraud = 0
    total_txns  = 0

    for _, row in df.iterrows():
        event = row.where(pd.notnull(row), None).to_dict()
        if "purchase_time" in event:
            event["purchase_time_ts"] = _parse_ts(str(event["purchase_time"]))
        else:
            event["purchase_time_ts"] = time.time()

        # Score inline
        rolling = store.get_features(event)
        store.record(event)
        features = consumer._build_feature_vector(event, rolling)
        fraud_prob = model_loader.predict_proba(features)
        is_fraud   = int(fraud_prob >= model_loader.threshold)

        total_txns  += 1
        total_fraud += is_fraud

        scored = {
            **event,
            "fraud_probability": round(fraud_prob, 6),
            "fraud_decision":    is_fraud,
            "threshold_used":    round(model_loader.threshold, 4),
            **{f"rf_{k}": round(v, 4) for k, v in rolling.items()},
            "scored_at":         datetime.now(timezone.utc).isoformat(),
        }

        # Feed into analytics
        hour_key = analytics._hour_key(scored)
        if hour_key not in analytics._windows:
            analytics._windows[hour_key] = HourlyWindow(hour_key)
        analytics._windows[hour_key].ingest(scored)
        analytics._all_scored.append(scored)

    # Flush outputs
    analytics._flush_csv()
    analytics._flush_sqlite()
    analytics._log_summary()

    fraud_rate = total_fraud / max(total_txns, 1)
    logger.info(f"\nDemo complete: {total_txns} transactions  "
                f"fraud_flagged={total_fraud}  rate={fraud_rate:.3f}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def _usage() -> None:
    print(
        "\nUsage: python pipelines/streaming_inference_pipeline.py <mode>\n"
        "\nModes:\n"
        "  producer   — Publish raw events to Kafka 'purchases' topic\n"
        "  inference  — Score events and emit to 'fraud_scores' topic\n"
        "  analytics  — Aggregate scored events and write to CSV / SQLite\n"
        "  all        — Run all three in separate threads (dev shortcut)\n"
        "  demo       — In-process demo with no Kafka (local testing)\n"
    )


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "demo"

    if mode == "producer":
        TransactionProducer().run()

    elif mode == "inference":
        InferenceConsumer().run()

    elif mode == "analytics":
        AnalyticsConsumer().run()

    elif mode == "all":
        threads = [
            threading.Thread(target=TransactionProducer().run,  name="producer",  daemon=True),
            threading.Thread(target=InferenceConsumer().run,    name="inference", daemon=True),
            threading.Thread(target=AnalyticsConsumer().run,    name="analytics", daemon=True),
        ]
        for t in threads:
            t.start()
            time.sleep(2)    # stagger startup so consumer is ready before producer floods

        logger.info("All three streaming services running. Ctrl+C to stop.")
        try:
            while not _SHUTDOWN.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            _SHUTDOWN.set()
        for t in threads:
            t.join(timeout=15)

    elif mode == "demo":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        run_local_demo(max_events=n)

    else:
        _usage()
        sys.exit(1)
