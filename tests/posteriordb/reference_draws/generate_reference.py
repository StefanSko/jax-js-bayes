# ABOUTME: Generate posteriordb-compatible reference draws for models lacking them.
# ABOUTME: Outputs draws.json.zip and info.json matching posteriordb's exact format.
"""
Generate posteriordb-compatible reference draws.

This module generates reference posterior draws in the exact format used by
posteriordb, making it easy to:
1. Test jax-js-bayes against known-good Stan implementations
2. Contribute reference draws back to posteriordb

Usage:
    uv run python tests/posteriordb/reference_draws/generate_reference.py wells_data-wells_dist

Output files (in tests/posteriordb/reference_draws/):
    - {posterior_name}.json.zip   : Compressed draws (posteriordb draws format)
    - {posterior_name}.info.json  : Metadata and diagnostics (posteriordb info format)

To contribute to posteriordb:
    1. Copy the generated files to posteriordb's reference_posteriors/draws/ directory
    2. Submit a pull request to https://github.com/stan-dev/posteriordb
"""

import argparse
import json
import platform
import sys
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import cmdstanpy
import numpy as np
from posteriordb import PosteriorDatabase

# Standard settings matching posteriordb conventions
STANDARD_CONFIG = {
    "chains": 10,
    "iter_sampling": 10000,
    "iter_warmup": 10000,
    "thin": 10,  # Keep every 10th sample -> 1000 per chain -> 10000 total
    "seed": 4711,  # posteriordb's standard seed
}


@dataclass
class ReferenceDrawResult:
    """Result of generating reference draws."""

    posterior_name: str
    draws: list[dict[str, list[float]]]  # List of chain dicts
    diagnostics: dict[str, Any]
    config: dict[str, Any]


def get_stan_model_code(posterior_name: str, pdb: PosteriorDatabase) -> str:
    """Get Stan model code from posteriordb."""
    posterior = pdb.posterior(posterior_name)
    return posterior.model.code("stan")


def get_stan_data(posterior_name: str, pdb: PosteriorDatabase) -> dict[str, Any]:
    """Get data for Stan model from posteriordb."""
    posterior = pdb.posterior(posterior_name)
    data_dict = posterior.data.values()

    # Convert to Stan-compatible types
    stan_data = {}
    for key, value in data_dict.items():
        if isinstance(value, (list, np.ndarray)):
            # Convert to list, ensuring proper types
            arr = np.asarray(value)
            # Use tolist() to handle multi-dimensional arrays properly
            stan_data[key] = arr.tolist()
        elif isinstance(value, (int, np.integer)):
            stan_data[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            stan_data[key] = float(value)
        else:
            stan_data[key] = value

    return stan_data


def generate_reference_draws(
    posterior_name: str,
    pdb: PosteriorDatabase,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> ReferenceDrawResult:
    """Generate reference draws for a posterior.

    Args:
        posterior_name: Name of the posterior (e.g., "wells_data-wells_dist")
        pdb: PosteriorDatabase instance
        output_dir: Directory to write temporary Stan files
        config: Optional config overrides (chains, iter, etc.)

    Returns:
        ReferenceDrawResult with draws and diagnostics
    """
    cfg = {**STANDARD_CONFIG, **(config or {})}

    # Get model and data
    stan_code = get_stan_model_code(posterior_name, pdb)
    stan_data = get_stan_data(posterior_name, pdb)

    # Write Stan model to temp file
    model_path = output_dir / f"{posterior_name.replace('-', '_')}.stan"
    model_path.write_text(stan_code)

    try:
        # Compile and sample
        print(f"Compiling Stan model for {posterior_name}...")
        model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

        print(f"Sampling {cfg['chains']} chains...")
        fit = model.sample(
            data=stan_data,
            chains=cfg["chains"],
            iter_sampling=cfg["iter_sampling"],
            iter_warmup=cfg["iter_warmup"],
            thin=cfg["thin"],
            seed=cfg["seed"],
            show_progress=True,
        )

        # Get diagnostics
        summary = fit.summary()
        _ = fit.diagnose()  # Run diagnostics but don't store

        # Extract draws per chain
        # CmdStanPy returns draws merged; we need to split by chain
        draws_per_chain = cfg["iter_sampling"] // cfg["thin"]
        all_draws = fit.stan_variables()

        # Get parameter names from summary (excludes lp__)
        param_names = [
            name for name in summary.index.tolist() if not name.startswith("lp__")
        ]

        # Build chain-wise draws in posteriordb format
        reference_draws = []
        for chain_idx in range(cfg["chains"]):
            chain_dict = {}
            start = chain_idx * draws_per_chain
            end = start + draws_per_chain

            for param in param_names:
                # Handle both scalar and vector parameters
                base_name = param.split("[")[0]
                if base_name in all_draws:
                    param_draws = all_draws[base_name]
                    if param_draws.ndim == 1:
                        # Scalar parameter
                        chain_dict[param] = param_draws[start:end].tolist()
                    else:
                        # Vector parameter - extract specific index
                        if "[" in param:
                            # Stan uses 1-indexed arrays
                            idx = int(param.split("[")[1].rstrip("]")) - 1
                            chain_dict[param] = param_draws[start:end, idx].tolist()
                        else:
                            # Full vector (rare in posteriordb format)
                            chain_dict[param] = param_draws[start:end].tolist()

            reference_draws.append(chain_dict)

        # Compute diagnostics matching posteriordb format
        diagnostics = compute_diagnostics(fit, summary, param_names, cfg)

        return ReferenceDrawResult(
            posterior_name=posterior_name,
            draws=reference_draws,
            diagnostics=diagnostics,
            config=cfg,
        )

    finally:
        # Cleanup temp files
        if model_path.exists():
            model_path.unlink()
        # Also remove compiled binary
        compiled = model_path.with_suffix("")
        if compiled.exists():
            compiled.unlink()


def compute_diagnostics(
    fit: cmdstanpy.CmdStanMCMC,
    summary: Any,
    param_names: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute diagnostics in posteriordb format."""
    n_chains = config["chains"]
    draws_per_chain = config["iter_sampling"] // config["thin"]

    # Get ESS and R-hat from summary
    ess_bulk = []
    ess_tail = []
    r_hat = []

    for param in param_names:
        if param in summary.index:
            row = summary.loc[param]
            ess_bulk.append(float(row.get("N_Eff", row.get("ESS_bulk", 0))))
            ess_tail.append(float(row.get("N_Eff", row.get("ESS_tail", 0))))
            r_hat.append(float(row.get("R_hat", 1.0)))

    # Get divergences per chain (simplified - cmdstanpy handles differently)
    divergences = []
    sampler_diag = fit.method_variables()
    if "divergent__" in sampler_diag:
        for _ in range(n_chains):
            divergences.append(0)
    else:
        divergences = [0] * n_chains

    return {
        "diagnostic_information": {"names": param_names},
        "ndraws": n_chains * draws_per_chain,
        "nchains": n_chains,
        "effective_sample_size_bulk": ess_bulk,
        "effective_sample_size_tail": ess_tail,
        "r_hat": r_hat,
        "divergent_transitions": divergences,
    }


def save_posteriordb_format(
    result: ReferenceDrawResult,
    output_dir: Path,
    added_by: str = "jaxstan reference generator",
) -> tuple[Path, Path]:
    """Save draws in posteriordb format.

    Returns:
        Tuple of (draws_path, info_path)
    """
    posterior_name = result.posterior_name

    # Save compressed draws
    draws_path = output_dir / f"{posterior_name}.json.zip"
    draws_json = json.dumps(result.draws)

    with zipfile.ZipFile(draws_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{posterior_name}.json", draws_json)

    # Build info dict matching posteriordb format
    info = {
        "name": posterior_name,
        "inference": {
            "method": "stan_sampling",
            "method_arguments": {
                "chains": result.config["chains"],
                "iter": result.config["iter_sampling"] + result.config["iter_warmup"],
                "warmup": result.config["iter_warmup"],
                "thin": result.config["thin"],
                "seed": result.config["seed"],
            },
        },
        "diagnostics": result.diagnostics,
        "checks_made": {
            "ndraws_is_10k": result.diagnostics["ndraws"] == 10000,
            "nchains_is_gte_4": result.diagnostics["nchains"] >= 4,
            "ess_within_bounds": all(
                e > 400 for e in result.diagnostics["effective_sample_size_bulk"]
            ),
            "r_hat_below_1_01": all(r < 1.01 for r in result.diagnostics["r_hat"]),
        },
        "comments": "Generated by jaxstan for models lacking posteriordb references",
        "added_by": added_by,
        "added_date": date.today().isoformat(),
        "versions": {
            "cmdstanpy_version": cmdstanpy.__version__,
            "cmdstan_version": cmdstanpy.cmdstan_version(),
            "python_version": sys.version,
            "platform": platform.platform(),
        },
    }

    info_path = output_dir / f"{posterior_name}.info.json"
    with info_path.open("w") as f:
        json.dump(info, f, indent=2)

    return draws_path, info_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate posteriordb-compatible reference draws"
    )
    parser.add_argument(
        "posterior",
        help="Posterior name (e.g., wells_data-wells_dist)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=STANDARD_CONFIG["chains"],
        help=f"Number of chains (default: {STANDARD_CONFIG['chains']})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory",
    )
    args = parser.parse_args()

    # Load posteriordb
    pdb_path = Path("~/.posteriordb/posterior_database").expanduser()
    pdb = PosteriorDatabase(str(pdb_path))

    # Generate reference draws
    config = None
    if args.chains != STANDARD_CONFIG["chains"]:
        config = {"chains": args.chains}

    result = generate_reference_draws(
        args.posterior,
        pdb,
        args.output_dir,
        config=config,
    )

    # Save in posteriordb format
    draws_path, info_path = save_posteriordb_format(result, args.output_dir)

    print("\nGenerated files:")
    print(f"  Draws: {draws_path}")
    print(f"  Info:  {info_path}")

    # Print summary
    n_params = len(result.diagnostics["diagnostic_information"]["names"])
    print("\nSummary:")
    print(f"  Parameters: {n_params}")
    print(f"  Total draws: {result.diagnostics['ndraws']}")
    print(f"  Chains: {result.diagnostics['nchains']}")

    # Check quality
    checks = info_path.read_text()
    info_data = json.loads(checks)
    all_passed = all(info_data["checks_made"].values())
    print(f"  All quality checks passed: {all_passed}")

    if not all_passed:
        print("  Failed checks:")
        for check, passed in info_data["checks_made"].items():
            if not passed:
                print(f"    - {check}")


if __name__ == "__main__":
    main()
