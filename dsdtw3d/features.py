"""A nine-channel XYZ/velocity/acceleration baseline without synthetic pressure."""
import numpy as np

FEATURE_NAMES = ("x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az")


def extract_xyz(points, samples=128):
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2:
        raise ValueError("Expected at least two XYZ points, with shape (T, 3)")
    if not np.isfinite(xyz).all() or samples < 48:
        raise ValueError("Coordinates must be finite and samples must be >= 48")
    # One spatial scale for all axes preserves the relative XYZ geometry.
    span = max(float(np.ptp(xyz, axis=0).max()), 1e-8)
    xyz = (xyz - xyz.mean(axis=0)) / span
    source_time = np.linspace(0, 1, len(xyz))
    target_time = np.linspace(0, 1, samples)
    xyz = np.stack([np.interp(target_time, source_time, xyz[:, i]) for i in range(3)], axis=1)
    xyz -= xyz.mean(axis=0)
    velocity = np.gradient(xyz, target_time, axis=0)
    acceleration = np.gradient(velocity, target_time, axis=0)
    # Use one RMS per vector group; never divide by a constant-axis std.
    velocity /= max(float(np.sqrt(np.mean(velocity**2))), 1e-8)
    acceleration /= max(float(np.sqrt(np.mean(acceleration**2))), 1e-8)
    result = np.concatenate([xyz, velocity, acceleration], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite 3D features")
    return result
