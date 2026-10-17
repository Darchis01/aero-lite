"""
Synthetic network simulator.

Replaces what Mininet (virtual topology) and the Ryu controller (live
OpenFlow port stats) provided in the original AERO. This module generates
the same shape of data -- a fixed topology plus per-link traffic metrics --
deterministically from a client-supplied "tick" and any manually injected
failures, so the whole system is stateless and safe to run as a serverless
function (no persistent process required).
"""
import math
import random
import zlib


def _stable_hash(*parts):
    """Process-independent hash. Python's built-in hash() on strings is
    randomized per process, which would make results differ between
    serverless instances."""
    return zlib.crc32('|'.join(str(p) for p in parts).encode('utf-8'))

# Fixed topology: mirrors the original AERO demo topology (2 edge hosts,
# a core of switches, 2 more hosts, and two simulated cloud endpoints).
NODES = [
    {"id": "h1", "label": "h1", "type": "host", "x": 80, "y": 120},
    {"id": "h2", "label": "h2", "type": "host", "x": 80, "y": 320},
    {"id": "s1", "label": "s1", "type": "switch", "x": 220, "y": 120},
    {"id": "s2", "label": "s2", "type": "switch", "x": 220, "y": 320},
    {"id": "s3", "label": "s3", "type": "switch", "x": 420, "y": 220},
    {"id": "s4", "label": "s4", "type": "switch", "x": 620, "y": 220},
    {"id": "s5", "label": "s5-gw", "type": "switch", "x": 820, "y": 220},
    {"id": "h3", "label": "h3", "type": "host", "x": 740, "y": 80},
    {"id": "h4", "label": "h4", "type": "host", "x": 740, "y": 200},
    {"id": "h5", "label": "AWS-SIM", "type": "cloud", "x": 950, "y": 100},
    {"id": "h6", "label": "Azure-SIM", "type": "cloud", "x": 950, "y": 280},
]

LINKS = [
    ("h1", "s1"), ("h2", "s2"), ("s1", "s3"), ("s2", "s3"),
    ("s3", "s4"), ("s4", "s5"), ("s4", "h3"), ("s4", "h4"),
    ("s5", "h5"), ("s5", "h6"),
]

STATUS_COLOR = {"up": "#10b981", "degraded": "#f59e0b", "failed": "#ef4444"}


def _link_seed(src, dst, tick):
    """Deterministic per-link, per-tick seed so repeated polls at the same
    tick are stable, but different links/ticks diverge."""
    return _stable_hash(src, dst, tick // 3)


def generate_link_metrics(src, dst, tick, injected_failures):
    """Produce one link's [utilization, loss, latency, queue, burst]
    feature vector for this tick, using the same correlated structure the
    model was trained on (see training/train_model.py)."""
    rng = random.Random(_link_seed(src, dst, tick))

    link_key = f"{src}-{dst}"
    is_injected = link_key in injected_failures or f"{dst}-{src}" in injected_failures

    # Smooth background load variation over time (diurnal-like wave) plus
    # per-link baseline so some links run hotter than others.
    baseline = (_stable_hash(src, dst) % 20) + 15
    wave = 15 * math.sin(tick / 6 + _stable_hash(link_key) % 10)
    util = max(2, min(99, baseline + wave + rng.gauss(0, 6)))

    if is_injected:
        util = rng.uniform(88, 99)

    latency = max(1, 5 + util * 0.55 + rng.gauss(0, 8))
    if util > 80:
        loss = max(0, rng.gauss(0.06, 0.03) + (util - 80) * 0.004)
    else:
        loss = max(0, rng.gauss(0.005, 0.006))
    loss = min(1, loss)

    queue = max(0, min(150, util * 0.9 + loss * 200 + rng.gauss(0, 10)))
    burst = max(0, min(1, util / 100 * 0.7 + rng.gauss(0, 0.15)))

    return {
        "utilization_pct": round(util, 1),
        "packet_loss_frac": round(loss, 4),
        "latency_ms": round(latency, 1),
        "queue_depth": round(queue, 1),
        "burst_rate": round(burst, 2),
    }


def build_state(tick, injected_failures, predictive_failover_enabled, predict_fn):
    """Build the full topology + per-link predictions for one simulated
    moment. predict_fn: callable(features_list) -> (pred, confidence, proba)."""
    edges = []
    predictions = []

    for idx, (src, dst) in enumerate(LINKS):
        metrics = generate_link_metrics(src, dst, tick, injected_failures)
        features = [
            metrics["utilization_pct"],
            metrics["packet_loss_frac"],
            metrics["latency_ms"],
            metrics["queue_depth"],
            metrics["burst_rate"],
        ]
        pred, confidence, proba = predict_fn(features)

        link_key = f"{src}-{dst}"
        is_injected = link_key in injected_failures or f"{dst}-{src}" in injected_failures

        if is_injected:
            status = "failed"
        elif pred == 1 and predictive_failover_enabled:
            status = "degraded"
        elif pred == 1:
            status = "degraded"
        else:
            status = "up"

        edges.append({
            "id": f"edge-{idx}-{src}-{dst}",
            "source": src,
            "target": dst,
            "label": link_key,
            "status": status,
            "color": STATUS_COLOR[status],
            "width": 3,
            "metrics": metrics,
            "prediction": {"risk": pred, "confidence": confidence, "proba": proba},
        })

        if pred == 1 or is_injected:
            predictions.append({
                "link": link_key,
                "risk": "failure-risk" if pred == 1 else "manual-failure",
                "confidence": round(confidence, 3),
                "metrics": metrics,
            })

    degraded_count = len([e for e in edges if e["status"] != "up"])
    active_up = [e for e in edges if e["status"] == "up"]
    avg_latency = round(sum(e["metrics"]["latency_ms"] for e in edges) / len(edges), 1)
    avg_util = round(sum(e["metrics"]["utilization_pct"] for e in edges) / len(edges), 1)
    avg_loss = round(sum(e["metrics"]["packet_loss_frac"] for e in edges) * 100 / len(edges), 2)
    throughput = round(sum(100 - e["metrics"]["utilization_pct"] for e in active_up) * 2.1, 1) if active_up else 0

    stats = {
        "active_predictions": len(predictions),
        "degraded_links": degraded_count,
        "total_links": len(edges),
        "avg_latency_ms": avg_latency,
        "avg_utilization_pct": avg_util,
        "avg_packet_loss_pct": avg_loss,
        "current_throughput_mbps": throughput,
        "predictive_failover_enabled": predictive_failover_enabled,
    }

    # Illustrative comparison: with prediction, degraded links get rerouted
    # proactively (bounded small downtime); without, they run to failure
    # before recovery (larger downtime). Scales with how many links are
    # actually at risk this tick, rather than being a fixed canned number.
    with_pred = {
        "downtime_ms": 40 + degraded_count * 15,
        "packet_loss_pct": round(0.2 + degraded_count * 0.3, 2),
    }
    without_pred = {
        "downtime_ms": 300 + degraded_count * 120,
        "packet_loss_pct": round(4.0 + degraded_count * 1.8, 2),
    }
    improvement_pct = round(
        (without_pred["downtime_ms"] - with_pred["downtime_ms"]) / without_pred["downtime_ms"] * 100, 1
    )

    return {
        "nodes": NODES,
        "edges": edges,
        "predictions": predictions[-20:],
        "stats": stats,
        "comparison": {
            "with_prediction": with_pred,
            "without_prediction": without_pred,
            "improvement_pct": improvement_pct,
        },
    }
