import dataclasses
import time
import logging
from typing import Tuple, Dict, Any

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@dataclasses.dataclass
class Args: 
    """Compare inference between latent_action=True and latent_action=False models.

    Notes/Assumptions:
    - You trained both variants and saved checkpoints. Provide their directories.
    - We construct a query observation by sampling a single item from the same
      training-style dataset used in `train_pi0_fast_ricl.py`.
    - Retrieval during inference requires a `demos_dir` containing per-demo
      folders with `processed_demo.npz` files (as used by `RiclPolicy`).
    """

    # Checkpoint directories. Fill these in.
    ckpt_dir_latent: str = "checkpoints/pi0_fast_droid_ricl/{ricl_latent_action}/9999"
    ckpt_dir_nonlatent: str = "pi0_fast_droid_ricl_checkpoint" 

    # Directory containing processed demos for retrieval at inference time.
    demos_dir: str = "2025-03-14_move_apple_to_the_right"

    # Training config name for RICL (must match your training run).
    config_name: str = "pi0_fast_droid_ricl"

    # If you fine-tuned on your own collected demos, set this; otherwise leave None.
    finetuning_collected_demos_dir: str = "preprocessing/collected_demos_training/"

    # Which sample from the dataset to use to build the query observation.
    sample_index: int = 0

    # Timing parameters
    num_warmup: int = 1
    num_iters: int = 3


def _build_configs(args: Args) -> tuple[_config.TrainConfig, _config.TrainConfig]:
    base_cfg = _config.get_config(args.config_name)

    # Ensure latent_action=True variant (matches named config defaults)
    model_latent = dataclasses.replace(base_cfg.model, latent_action=True)
    cfg_latent = dataclasses.replace(base_cfg, model=model_latent)

    # Create latent_action=False variant from the same base
    model_nonlatent = dataclasses.replace(base_cfg.model, latent_action=False)
    cfg_nonlatent = dataclasses.replace(base_cfg, model=model_nonlatent)

    # Optional fine-tuning demos dir for dataset creation (only affects how we sample a query
    # from training-style data, not the retrieval at inference which uses demos_dir below).
    if args.finetuning_collected_demos_dir is not None:
        cfg_latent = dataclasses.replace(
            cfg_latent, finetuning_collected_demos_dir=args.finetuning_collected_demos_dir
        )
        cfg_nonlatent = dataclasses.replace(
            cfg_nonlatent, finetuning_collected_demos_dir=args.finetuning_collected_demos_dir
        )

    return cfg_latent, cfg_nonlatent


def _make_query_from_training_sample(cfg: _config.TrainConfig, sample_index: int) -> dict:
    """Sample one item from RiclDroidDataset and extract a single-query observation.

    We only pass the query_* keys (and prompt) to the policy, so that retrieval is
    performed inside `RiclPolicy.retrieve(...)` using `demos_dir`.
    """
    try:
        dataset = _data_loader.RiclDroidDataset(cfg.model, cfg.finetuning_collected_demos_dir)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Failed to initialize RiclDroidDataset. "
            "TODO: Ensure the preprocessing files and paths used during training are available. "
            "You may need to set `Args.finetuning_collected_demos_dir` or mirror the same preprocessing layout."
        ) from e

    example = dataset[sample_index]

    # Construct a query observation as expected by the policy server example.
    obs = {
        "query_top_image": example["query_top_image"],
        "query_right_image": example["query_right_image"],
        "query_wrist_image": example["query_wrist_image"],
        "query_state": example["query_state"],
        # The dataset provides a per-sample prompt already
        "query_prompt": example.get("query_prompt", "do something"),
        # Optional, only used for organizing debug logs saved by the policy
        "prefix": "test_ricl",
    }
    return obs


def _load_policy(cfg: _config.TrainConfig, ckpt_dir: str, demos_dir: str):
    if not ckpt_dir:
        raise ValueError(
            "ckpt_dir is empty. TODO: set Args.ckpt_dir_latent / Args.ckpt_dir_nonlatent to your checkpoint directories."
        )
    if not demos_dir:
        raise ValueError(
            "demos_dir is empty. TODO: set Args.demos_dir to a folder of processed demos (with processed_demo.npz in subfolders)."
        )
    # Loads model params + norm stats from checkpoint, and wires up transforms
    # appropriate for RICL inference. Retrieval will be built from demos_dir.
    return _policy_config.create_trained_ricl_policy(cfg, ckpt_dir, demos_dir)


def _time_inference(policy, obs: dict, num_warmup: int, num_iters: int) -> Tuple[dict, float, float]:
    """
    This function is used to time the inference of the policy.
    It returns the last output, the mean and standard deviation of the inference time, in seconds.
    """
    # Warmup to trigger JIT compilation
    for _ in range(max(0, num_warmup)):
        policy.infer(obs)

    times_ms: list[float] = []
    last_output: dict | None = None
    for _ in range(max(1, num_iters)):
        t0 = time.time()
        out = policy.infer(obs)
        times_ms.append(time.time() - t0)
        last_output = out

    assert last_output is not None
    return last_output, float(np.mean(times_ms)), float(np.std(times_ms))


def _compare_actions(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return {
        "a_values": a,
        "b_values": b,
        "diff_l2_norm": float(np.linalg.norm(diff)),
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Building configs...")
    cfg_latent, cfg_nonlatent = _build_configs(args)

    logging.info("Sampling a training-style query observation...")
    obs = _make_query_from_training_sample(cfg_latent, args.sample_index)

    logging.info("Loading policies...")
    policy_latent = _load_policy(cfg_latent, args.ckpt_dir_latent, args.finetuning_collected_demos_dir+args.demos_dir)
    policy_nonlatent = _load_policy(cfg_nonlatent, args.ckpt_dir_nonlatent, args.finetuning_collected_demos_dir+args.demos_dir)

    logging.info("Running inference (latent_action=True)...")
    out_latent, mean_ms_latent, std_ms_latent = _time_inference(
        policy_latent, obs, args.num_warmup, args.num_iters
    )
    actions_latent = out_latent.get("query_actions")
    if actions_latent is None:
        raise RuntimeError("latent_action=True policy did not return 'query_actions'")

    logging.info("Running inference (latent_action=False)...")
    out_nonlatent, mean_ms_nonlatent, std_ms_nonlatent = _time_inference(
        policy_nonlatent, obs, args.num_warmup, args.num_iters
    )
    actions_nonlatent = out_nonlatent.get("query_actions")
    if actions_nonlatent is None:
        raise RuntimeError("latent_action=False policy did not return 'query_actions'")

    logging.info("Comparing outputs...")
    metrics = _compare_actions(actions_latent, actions_nonlatent)

    print()
    print("==== Inference timing ====")
    print(f"latent_action=True : {mean_ms_latent:.4f} s ± {std_ms_latent:.4f}rs over {max(1, args.num_iters)} iters")
    print(
        f"latent_action=False: {mean_ms_nonlatent:.4f} s ± {std_ms_nonlatent:.4f} ls over {max(1, args.num_iters)} iters"
    )

    print()
    print("==== Output comparison (query_actions) ====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Optional: add your own downstream evaluation metric here (e.g., task success proxy)
    # TODO: implement task-specific evaluation if available.


if __name__ == "__main__":
    main(Args)


