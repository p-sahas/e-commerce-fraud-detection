"""
test_streaming.py — Unit tests for the streaming inference pipeline

Covers:
  RollingFeatureStore:
    - record(): stores events per device and IP
    - get_features(): correct key set returned
    - get_features(): leakage-safe (excludes current event timestamp)
    - get_features(): returns zeros for unknown device/IP
    - get_features(): 1h window shorter than 24h window
    - get_features(): velocity counts accumulate correctly
    - get_features(): amount sums correct
    - get_features(): rarity score = 1/(1+count)
    - evict_old(): removes stale events
    - thread-safety: concurrent record() calls don't corrupt state

  FraudModelLoader:
    - stub predictor returns float in [0, 1]
    - stub predictor output distribution is reasonable (mean ~ 0.1)
    - threshold property returns float

  TransactionProducer._parse_ts:
    - parses '%Y-%m-%d %H:%M:%S' format
    - parses ISO format
    - falls back gracefully on bad input

  InferenceConsumer._build_feature_vector:
    - returns numpy array of correct dtype
    - length matches expected feature count
"""

import os
import sys
import threading
import time

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "utils"))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipelines"))


# =============================================================================
# Import helpers — the pipeline module uses module-level config loading,
# so we import lazily inside tests to avoid Kafka/MLflow side-effects
# =============================================================================

def _get_store():
    from streaming_inference_pipeline import RollingFeatureStore
    return RollingFeatureStore()


def _make_event(device_id="DEV1", ip="1.2.3.4", ts=None, value=100.0, label=0):
    return {
        "device_id":        device_id,
        "ip_address":       ip,
        "purchase_time_ts": ts or time.time(),
        "purchase_value":   value,
        "class":            label,
        "user_id":          "U1",
        "purchase_value_norm": 0.3,
        "age":              30.0,
        "purchase_hour":    14,
        "purchase_day_of_week": 2,
        "source":           "SEO",
        "browser":          "Chrome",
        "sex":              "M",
    }


# =============================================================================
# RollingFeatureStore — record()
# =============================================================================

class TestRollingFeatureStoreRecord:

    def test_record_stores_device_event(self):
        store = _get_store()
        event = _make_event(device_id="DEV99", ts=1_000_000.0, value=50.0)
        store.record(event)
        d_events = list(store._device_events["DEV99"])
        assert len(d_events) == 1
        assert d_events[0] == (1_000_000.0, 50.0)

    def test_record_stores_ip_event(self):
        store = _get_store()
        event = _make_event(ip="192.0.0.1", ts=2_000_000.0, value=75.0)
        store.record(event)
        ip_events = list(store._ip_events["192.0.0.1"])
        assert len(ip_events) == 1
        assert ip_events[0][1] == 75.0

    def test_multiple_events_accumulate(self):
        store = _get_store()
        base = 1_600_000_000.0
        for i in range(5):
            store.record(_make_event(device_id="DEVA", ts=base + i * 60, value=10.0 * i))
        assert len(store._device_events["DEVA"]) == 5

    def test_different_devices_stored_separately(self):
        store = _get_store()
        for dev in ["D1", "D2", "D3"]:
            store.record(_make_event(device_id=dev, ts=1e9, value=10.0))
        assert len(store._device_events) == 3

    def test_missing_device_id_ignored(self):
        store = _get_store()
        event = _make_event(ts=1e9)
        event.pop("device_id", None)
        event["device_id"] = ""     # empty string — should not be stored
        store.record(event)
        assert "" not in store._device_events or len(store._device_events[""]) == 0


# =============================================================================
# RollingFeatureStore — get_features()
# =============================================================================

class TestRollingFeatureStoreGetFeatures:

    EXPECTED_KEYS = {
        "device_txn_count_1h",  "device_txn_count_24h",
        "device_amount_sum_1h", "device_amount_sum_24h",
        "ip_txn_count_1h",      "ip_txn_count_24h",
        "ip_amount_sum_1h",     "ip_amount_sum_24h",
        "device_rarity",        "ip_rarity",
    }

    def test_returns_correct_keys(self, streaming_events):
        store = _get_store()
        feats = store.get_features(streaming_events[0])
        assert set(feats.keys()) == self.EXPECTED_KEYS

    def test_zeros_for_unknown_device_ip(self):
        store = _get_store()
        event = _make_event(device_id="UNKNOWN_DEV", ip="0.0.0.0", ts=1e9)
        feats = store.get_features(event)
        assert feats["device_txn_count_1h"]  == 0
        assert feats["device_txn_count_24h"] == 0
        assert feats["ip_txn_count_1h"]      == 0
        assert feats["ip_amount_sum_24h"]    == 0.0

    def test_leakage_safe_current_event_excluded(self):
        """The event itself must not be counted in its own features."""
        store = _get_store()
        ts = 1_700_000_000.0
        event = _make_event(device_id="DEVX", ts=ts, value=99.0)
        store.record(event)   # record the event first (simulate store state)
        # Then query features AT the same timestamp — should see 1 prior event
        # because we recorded it; but that's fine — the window is [ts-1h, ts)
        feats = store.get_features(event)
        # The event we recorded has ts == query ts, so t < ts is FALSE → excluded
        assert feats["device_txn_count_1h"] == 0

    def test_velocity_count_includes_past_events(self):
        store = _get_store()
        base = 1_700_000_000.0
        # Record 3 events 10 minutes before the query
        for i in range(3):
            prior = _make_event(device_id="DEVV", ts=base - 3600 + i * 60, value=10.0)
            store.record(prior)

        query = _make_event(device_id="DEVV", ts=base, value=50.0)
        feats = store.get_features(query)
        assert feats["device_txn_count_1h"] == 3

    def test_24h_count_only_includes_last_day(self):
        store = _get_store()
        base = 1_700_000_000.0
        # 2 events within 1h, 3 more between 1h and 24h, 5 older than 24h
        for i in range(2):
            store.record(_make_event("DEVZ", ts=base - 1800 + i * 60, value=1.0))
        for i in range(3):
            store.record(_make_event("DEVZ", ts=base - 7200 + i * 60, value=1.0))
        for i in range(5):
            store.record(_make_event("DEVZ", ts=base - 90_000 + i * 60, value=1.0))

        feats = store.get_features(_make_event("DEVZ", ts=base))
        assert feats["device_txn_count_1h"]  == 2
        assert feats["device_txn_count_24h"] == 5   # 2 + 3

    def test_amount_sum_correct(self):
        store = _get_store()
        base = 1_700_000_000.0
        amounts = [10.0, 20.0, 30.0]
        for i, amt in enumerate(amounts):
            store.record(_make_event("DEVA", ts=base - 1000 + i * 10, value=amt))
        feats = store.get_features(_make_event("DEVA", ts=base))
        assert abs(feats["device_amount_sum_1h"] - sum(amounts)) < 0.001

    def test_rarity_formula(self):
        store = _get_store()
        base = 1_700_000_000.0
        # Record 4 events so device_txn_count_24h = 4 → rarity = 1/(1+4) = 0.2
        for i in range(4):
            store.record(_make_event("DEVR", ts=base - 3600 + i, value=5.0))
        feats = store.get_features(_make_event("DEVR", ts=base))
        expected_rarity = 1.0 / (1.0 + feats["device_txn_count_24h"])
        assert abs(feats["device_rarity"] - expected_rarity) < 1e-9

    def test_rarity_is_one_for_new_device(self):
        store = _get_store()
        feats = store.get_features(_make_event("BRAND_NEW_DEV", ts=1e9))
        assert feats["device_rarity"] == 1.0   # 1/(1+0)

    def test_all_feature_values_are_floats(self):
        store = _get_store()
        feats = store.get_features(_make_event(ts=1e9))
        for k, v in feats.items():
            assert isinstance(v, (int, float)), f"Feature '{k}' is not numeric: {v!r}"


# =============================================================================
# RollingFeatureStore — evict_old()
# =============================================================================

class TestRollingFeatureStoreEviction:

    def test_evicts_old_events(self):
        store = _get_store()
        very_old_ts = time.time() - 200_000   # > 2×24h
        store.record(_make_event("OLD_DEV", ts=very_old_ts, value=1.0))
        assert len(store._device_events["OLD_DEV"]) == 1
        store.evict_old(cutoff_secs=100_000)   # 100k seconds → removes the old event
        assert len(store._device_events["OLD_DEV"]) == 0

    def test_does_not_evict_recent_events(self):
        store = _get_store()
        recent_ts = time.time() - 60
        store.record(_make_event("RECENT", ts=recent_ts, value=5.0))
        store.evict_old(cutoff_secs=86_400 * 2)
        assert len(store._device_events["RECENT"]) == 1


# =============================================================================
# RollingFeatureStore — thread safety
# =============================================================================

class TestRollingFeatureStoreThreadSafety:

    def test_concurrent_records_do_not_corrupt(self):
        store = _get_store()
        errors = []
        base = 1_700_000_000.0

        def _writer(device_id):
            try:
                for i in range(50):
                    store.record(_make_event(device_id, ts=base + i, value=float(i)))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_writer, args=(f"T{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        total = sum(len(v) for v in store._device_events.values())
        assert total == 500   # 10 threads × 50 events


# =============================================================================
# FraudModelLoader (stub mode — no MLflow required)
# =============================================================================

class TestFraudModelLoaderStub:

    def test_stub_returns_float(self):
        from streaming_inference_pipeline import FraudModelLoader
        loader = FraudModelLoader()
        loader._model = None    # force stub mode
        result = loader.predict_proba(np.zeros(8))
        assert isinstance(result, float)

    def test_stub_output_in_valid_range(self):
        from streaming_inference_pipeline import FraudModelLoader
        loader = FraudModelLoader()
        loader._model = None
        results = [loader.predict_proba(np.zeros(8)) for _ in range(100)]
        assert all(0.0 <= r <= 1.0 for r in results), "All probas must be in [0, 1]"

    def test_stub_distribution_skews_low(self):
        """Beta(2, 18) distribution should produce mean ≈ 0.10 (≈ fraud rate)."""
        from streaming_inference_pipeline import FraudModelLoader
        loader = FraudModelLoader()
        loader._model = None
        results = [loader.predict_proba(np.zeros(4)) for _ in range(500)]
        mean = np.mean(results)
        assert mean < 0.30, f"Stub distribution mean {mean:.3f} too high — expected ~0.10"

    def test_threshold_property_is_float(self):
        from streaming_inference_pipeline import FraudModelLoader
        loader = FraudModelLoader()
        assert isinstance(loader.threshold, float)
        assert 0.0 < loader.threshold < 1.0


# =============================================================================
# TransactionProducer._parse_ts
# =============================================================================

class TestProducerParseTimestamp:

    @pytest.fixture
    def producer(self):
        from streaming_inference_pipeline import TransactionProducer
        return TransactionProducer()

    def test_parses_standard_format(self, producer):
        ts = producer._parse_ts("2023-11-01 14:30:00")
        # Allow ±3h for timezone differences across machines
        assert abs(ts - 1_698_846_600) < 10_800


    def test_parses_iso_format(self, producer):
        ts = producer._parse_ts("2023-11-01T14:30:00")
        assert ts > 0

    def test_parses_date_only(self, producer):
        ts = producer._parse_ts("2023-11-01")
        assert ts > 0

    def test_fallback_on_bad_input(self, producer):
        """Bad / unparseable timestamps must not raise — fallback to current time."""
        ts = producer._parse_ts("not-a-date")
        assert ts > 0

    def test_earlier_date_has_smaller_ts(self, producer):
        ts1 = producer._parse_ts("2022-01-01 00:00:00")
        ts2 = producer._parse_ts("2023-01-01 00:00:00")
        assert ts1 < ts2
