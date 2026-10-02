"""Probe Evidently 0.7.x API to find the drift detection approach."""
import evidently
print("Evidently:", evidently.__version__)

# Try legacy path (it should still be importable in 0.7)
try:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset
    print("Legacy path OK")
    LEGACY = True
except ImportError as e:
    print("Legacy path failed:", e)
    LEGACY = False

# Try new SDK path
try:
    from evidently import Report, Dataset
    from evidently.presets import DataDriftPreset as NewDDP
    print("New SDK path OK: Report, Dataset, DataDriftPreset")
    NEW = True
except ImportError as e:
    print("New path failed:", e)
    NEW = False

# Try metrics directly
try:
    from evidently.metrics import ValueDrift
    print("ValueDrift OK")
except ImportError as e:
    print("ValueDrift failed:", e)

# What's in presets?
try:
    import evidently.presets as ep
    print("presets dir:", [x for x in dir(ep) if not x.startswith('_')])
except Exception as e:
    print("presets error:", e)

# What's in metrics?
try:
    import evidently.metrics as em
    print("metrics dir:", [x for x in dir(em) if not x.startswith('_')][:20])
except Exception as e:
    print("metrics error:", e)
