"""Future CLI for normalization, auditing, and split preparation.

Target command:
`python scripts/prepare_data.py --config configs/base.yaml`

Implementation checklist:
1. Load config and private data paths.
2. Parse records and resolve every ``code_file``.
3. Run missing-file, duplicate, schema, label, and leakage audits.
4. Create a versioned normalized manifest and fixed split IDs.
5. Write only approved derived artifacts to ignored data directories.
"""


def main() -> None:
    # TODO: Load config, inspect private raw data, normalize records, and persist manifests.
    raise NotImplementedError("Data preparation is not implemented in the scaffold.")


if __name__ == "__main__":
    main()
