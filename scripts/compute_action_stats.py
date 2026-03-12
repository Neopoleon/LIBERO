#!/usr/bin/env python
"""Compute per-dimension action mean and std for LIBERO training datasets.

Usage:
    python scripts/compute_action_stats.py \
        --suites libero_spatial libero_object libero_goal \
        --n_train_tasks 8 \
        --out action_stats.npz \
        --dataset_root /path/to/datasets  # optional, defaults to ~/.libero/datasets

Outputs:
    action_stats.npz with keys:
        mean: (7,) float32 per-dimension mean
        std:  (7,) float32 per-dimension std (clipped to >= 1e-3)
"""
import argparse
import os
import glob
import numpy as np
import h5py
import yaml


def get_default_dataset_root():
    """Determine the default dataset root directory.

    Resolution order:
      1. ~/.libero/config.yaml  → "datasets" key
      2. Hard-coded fallback: /home/jeff/LIBERO/libero/datasets
    """
    config_path = os.path.expanduser("~/.libero/config.yaml")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            if cfg and "datasets" in cfg:
                return cfg["datasets"]
        except Exception as e:
            print(f"Warning: could not read {config_path}: {e}")
    return "/home/jeff/LIBERO/libero/datasets"


def collect_actions_from_file(hdf5_path):
    """Read all demo actions from a single HDF5 file.

    Parameters
    ----------
    hdf5_path : str
        Path to an HDF5 dataset file.

    Returns
    -------
    list of np.ndarray
        Each element has shape (T, 7).
    int
        Number of demos read from this file.
    """
    demo_actions = []
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        for demo_key in data_group.keys():
            actions = data_group[demo_key]["actions"][:]  # (T, 7)
            demo_actions.append(actions)
    return demo_actions, len(demo_actions)


def find_suite_files(dataset_root, suite, n_train_tasks=None):
    """Find HDF5 task files for a given suite.

    Parameters
    ----------
    dataset_root : str
        Root directory containing suite subdirectories.
    suite : str
        Suite name, e.g. "libero_spatial".
    n_train_tasks : int or None
        If specified, return only the first N files sorted alphabetically.

    Returns
    -------
    list of str
        Sorted (and optionally truncated) list of HDF5 file paths, or an
        empty list if the suite directory does not exist.
    """
    suite_dir = os.path.join(dataset_root, suite)
    if not os.path.isdir(suite_dir):
        print(f"Warning: suite directory not found, skipping: {suite_dir}")
        return []

    pattern = os.path.join(suite_dir, "*.hdf5")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Warning: no .hdf5 files found in {suite_dir}")
        return []

    if n_train_tasks is not None:
        files = files[:n_train_tasks]

    return files


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-dimension action mean and std for LIBERO datasets."
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal"],
        help="LIBERO suite names to include (default: libero_spatial libero_object libero_goal)",
    )
    parser.add_argument(
        "--n_train_tasks",
        type=int,
        default=None,
        help=(
            "If specified, use only the first N task files (sorted alphabetically) "
            "from each suite. Useful for OOD splits (e.g. --n_train_tasks 8 keeps "
            "tasks 0-7 and holds out 8-9)."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="action_stats.npz",
        help="Output .npz file path (default: action_stats.npz)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help=(
            "Root directory containing suite subdirectories. "
            "Defaults to the 'datasets' key in ~/.libero/config.yaml, "
            "or /home/jeff/LIBERO/libero/datasets if that file is absent."
        ),
    )
    args = parser.parse_args()

    # Resolve dataset root
    if args.dataset_root is None:
        args.dataset_root = get_default_dataset_root()
    args.dataset_root = os.path.expanduser(args.dataset_root)
    print(f"Dataset root: {args.dataset_root}")

    # Collect HDF5 file paths across all suites
    all_files = []
    for suite in args.suites:
        suite_files = find_suite_files(args.dataset_root, suite, args.n_train_tasks)
        if suite_files:
            print(
                f"Suite '{suite}': found {len(suite_files)} task file(s)"
                + (f" (capped at n_train_tasks={args.n_train_tasks})" if args.n_train_tasks else "")
            )
            for f in suite_files:
                print(f"  {os.path.basename(f)}")
        all_files.extend(suite_files)

    if not all_files:
        raise RuntimeError(
            "No HDF5 dataset files were found. "
            "Check that the datasets are downloaded and --dataset_root is correct.\n"
            f"  dataset_root = {args.dataset_root}\n"
            f"  suites       = {args.suites}"
        )

    # Read actions from all files
    print(f"\nCollecting actions from {len(all_files)} file(s)...")
    all_action_arrays = []
    total_demos = 0
    for path in all_files:
        try:
            demo_actions, n_demos = collect_actions_from_file(path)
            all_action_arrays.extend(demo_actions)
            total_demos += n_demos
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")

    if not all_action_arrays:
        raise RuntimeError(
            "No action data could be read from any of the dataset files."
        )

    # Concatenate all actions → (N_total, 7)
    actions = np.concatenate(all_action_arrays, axis=0)  # (N_total, 7)
    total_timesteps = actions.shape[0]

    # Compute statistics
    mean = actions.mean(axis=0)                  # (7,)
    std = actions.std(axis=0)                    # (7,)
    std = np.clip(std, 1e-3, None)               # avoid division by zero

    # Save
    np.savez(args.out, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"\nSaved action stats to: {os.path.abspath(args.out)}")

    # Print summary
    dim_labels = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    print("\n--- Summary ---")
    print(f"  Files processed : {len(all_files)}")
    print(f"  Total demos     : {total_demos}")
    print(f"  Total timesteps : {total_timesteps}")
    print(f"\n  Per-dimension statistics (7 dims):")
    print(f"  {'dim':<10} {'label':<10} {'mean':>12} {'std':>12}")
    print(f"  {'-'*46}")
    for i, label in enumerate(dim_labels):
        print(f"  {i:<10} {label:<10} {mean[i]:>12.6f} {std[i]:>12.6f}")


if __name__ == "__main__":
    main()
