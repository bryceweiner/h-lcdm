# Copilot Instructions for H-ΛCDM Analysis Framework

This document provides essential guidelines for AI coding agents working on the H-ΛCDM Analysis Framework. Follow these instructions to ensure productivity and alignment with project conventions.

## Project Overview

The H-ΛCDM Analysis Framework is a pipeline for analyzing Baryon Acoustic Oscillations (BAO) to test theoretical predictions against observational data. The pipeline supports multiple datasets and includes statistical validation, model comparison, and report generation.

### Key Components

- **`main.py`**: Entry point for running the pipeline. Supports various modes (e.g., `--bao`, `--recommendation`).
- **`data/`**: Contains data loaders, manifest files, and feature extractors for processing BAO datasets.
- **`results/`**: Stores generated reports, figures, and execution metadata.
- **`docs/`**: Includes documentation and research papers related to the project.
- **`docker/`**: Dockerfiles for containerized execution.

### Data Flow
1. **Input**: Observational data from `downloaded_data/`.
2. **Processing**: Data is loaded and processed using modules in `data/`.
3. **Analysis**: Theoretical predictions and statistical validations are performed.
4. **Output**: Results are saved in `results/`.

## Developer Workflows

### Running the Pipeline

- **Basic Run**:
  ```bash
  python main.py --bao
  ```
- **Validation**:
  ```bash
  python main.py --bao validate
  ```
- **Extended Validation**:
  ```bash
  python main.py --bao validate extended
  ```
- **Custom Output Directory**:
  ```bash
  python main.py --bao --output-dir ./my_results
  ```

### Testing
- Quick test run:
  ```bash
  python main.py --bao --quiet --output-dir ./test_results
  ```

### Debugging
- Use `--quiet` to suppress progress messages.
- Logs and metadata are stored in `results/execution_summary.json`.

## Project-Specific Conventions

- **Dataset Naming**: Use standardized names (e.g., `boss_dr12`, `desi_y1`). Refer to the `README.md` for the full list.
- **Validation Tiers**: Support for `validate` and `validate extended` modes.
- **Output Structure**:
  - Reports: `results/reports/`
  - Figures: `results/figures/`
  - Metadata: `results/execution_summary.json`

## Integration Points

- **External Dependencies**:
  - Python 3.8+
  - Libraries listed in `requirements.txt`.
- **Docker**:
  - Use `docker/Dockerfile` for containerized execution.

## Examples

- Full analysis with extended validation:
  ```bash
  python main.py --bao validate extended --generate-figures
  ```
- ML-derived recommendations:
  ```bash
  python main.py --recommendation 1
  ```

## References

- See `README.md` for detailed usage instructions.
- Refer to `docs/` for research papers and methodology.

---

For additional guidance, consult the project maintainers or the documentation in `docs/`.