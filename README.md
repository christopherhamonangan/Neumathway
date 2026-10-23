# Neumathway (NMN)

### A Neuro-Symbolic Engine for Mathematical Reasoning

Neumathway (NMN) is an experimental neuro-symbolic mathematical reasoning engine designed to process structured mathematical expressions using graph-based neural representations, recurrent decision-making, neural operator prediction, and symbolic verification.

Unlike general-purpose language models that process mathematics as part of natural-language sequences, NMN represents mathematical expressions as structured computational graphs. Its design combines learned operator selection with numerical computation and symbolic procedures for supported mathematical operations.

The project explores whether a compact, specialized reasoning architecture can improve mathematical execution, structural generalization, and computational efficiency without relying on a general-purpose large language model.

> **Project status:** Experimental research implementation. Reported benchmark results should be interpreted within their documented evaluation scope and independently reproduced before being treated as verified performance claims.

---

## Table of Contents

- [Overview](#overview)
- [Research Objectives](#research-objectives)
- [Architecture](#architecture)
- [Supported Mathematical Operations](#supported-mathematical-operations)
- [Dataset](#dataset)
- [Reported Experimental Results](#reported-experimental-results)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Model Configuration](#model-configuration)
- [Limitations](#limitations)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Overview

NMN is built around the idea that mathematical reasoning can benefit from a representation and execution pipeline specifically designed for mathematical structure.

The architecture combines several components:

- **Typed Expression Graph:** represents mathematical expressions through structured nodes and relationships.
- **Relation-Aware Graph Attention Network:** learns contextual relationships between mathematical operators, operands, and expression structure.
- **Recurrent Reasoning Controller:** coordinates iterative reasoning decisions.
- **Neural Operator Prediction:** predicts mathematical operations and transformations.
- **Neural Matrix Computation:** provides a learned numerical computation pathway.
- **Adaptive Strategy Router:** selects between supported execution strategies.
- **Symbolic Computation Engine:** performs exact or deterministic calculations for supported operations.
- **Strategy Memory:** reuses previously established execution strategies when a compatible expression structure is encountered.
- **Evolutionary Training Framework:** explores candidate strategies and model configurations.

The intended design is a hybrid system: neural components support learned decisions, while the symbolic engine provides deterministic execution for operations covered by its implementation.

## Research Objectives

NMN investigates the following research questions:

1. Can structured expression graphs improve the representation of mathematical expressions compared with treating them as flat token sequences?
2. Can a compact neural controller learn to select appropriate mathematical operators and execution strategies?
3. Can neural and symbolic computation be combined to improve the reliability of mathematical execution?
4. Can strategy reuse reduce repeated computation for recurring expression structures?
5. What trade-offs arise between numerical accuracy, structural generalization, inference latency, and memory consumption?

The central research objective is to evaluate a specialized mathematical reasoning system against appropriate symbolic, neural, and hybrid baselines.

These questions are research objectives, not claims that NMN has already demonstrated superiority over existing methods.

---

## Architecture

### 1. Typed Expression Graph

Mathematical inputs are represented as structured expression graphs rather than being processed exclusively as ordinary text.

The encoder uses an eight-dimensional feature representation:

| Feature | Description |
|---|---|
| `is_op` | Indicates whether a node represents an operator |
| `token_id` | Encoded identifier for the node's token |
| `signed_log_value` | Signed logarithmic representation of numerical values |
| `depth` | Node depth within the expression structure |
| `is_left_child` | Indicates a left-child relationship |
| `is_right_child` | Indicates a right-child relationship |
| `degree` | Encoded local structural degree |
| `in_lhs` | Indicates membership in the left-hand side of an equation |

The exact numerical encoding, normalization, and handling of special tokens are determined by the implementation.

### 2. Relation-Aware Graph Attention

The graph encoder uses four relation-aware graph attention layers with residual connections.

Its purpose is to propagate contextual information between related expression nodes, allowing the model to learn structural dependencies between operands, operators, and subexpressions.

### 3. Recurrent Reasoning Controller

The controller uses a two-layer gated recurrent unit (GRU) and cross-attention.

It coordinates sequential reasoning decisions and conditions its decisions on information obtained from the encoded expression graph.

### 4. Dual-Head Neural Computation

NMN contains two conceptual prediction pathways.

**Neural Operator Prediction (NOP)**

- Predicts among 16 configured action classes.
- Uses a learned temperature mechanism.
- Supports learned operator or transformation selection.

**Neural Matrix Computation (NMC)**

- Predicts continuous numerical representations.
- Uses signed-log numerical encoding.
- Uses Smooth L1 loss during training.

The exact semantics of the 16 action classes, numerical decoding procedure, and loss-weighting scheme should be documented alongside the corresponding source code.

### 5. Adaptive Strategy Router

A soft decision-tree router selects between supported execution strategies, including arithmetic expression reduction and algebraic transposition.

The router is intended to learn which strategy is appropriate for a given expression state. Its actual routing accuracy must be evaluated independently of final answer accuracy.

### 6. Symbolic Computation Engine

The symbolic engine provides deterministic procedures for explicitly supported operations, including:

- Basic arithmetic
- Quadratic equations using the ABC formula
- Polynomial differentiation
- Indefinite polynomial integration
- Determinants of 2 × 2 matrices
- Systems of two linear equations in two variables (SPLDV)

Exact arithmetic using rational numbers is implemented with Python's `Fraction` where applicable.

Symbolic correctness depends on the operation's implementation, input validation, domain handling, and output representation. The presence of a symbolic engine does not by itself establish correctness for every possible mathematical expression.

### 7. Strategy Memory

Strategy Memory provides a fast path for recurring expression structures whose execution strategies can be reused.

The reported implementation can bypass some neural inference steps when a compatible strategy is available. This can reduce latency for repeated structures, but performance depends on cache coverage, matching rules, and the cost of cache lookup.

Reported speedups must therefore be evaluated separately for cache hits, cache misses, and unseen expression structures.

### 8. Evolutionary Training

The training framework reportedly uses:

- A population of 32 candidate agents
- Six-to-one polygenic synthesis
- Strategy evolution
- An Occam's Razor-inspired preference for simpler candidates

The exact evolutionary operators, selection criteria, random seeds, and stopping conditions should be available in the training implementation for independent reproduction.

---

## Supported Mathematical Operations

The following operations are part of the intended scope of the current design.

| Category | Examples |
|---|---|
| Arithmetic | Addition, subtraction, multiplication, division |
| Algebra | Linear equations and algebraic transposition |
| Quadratic equations | Solutions using the ABC formula |
| Calculus | Polynomial differentiation and indefinite integration |
| Linear algebra | Determinants of 2 × 2 matrices |
| Systems of equations | SPLDV |
| Structural expressions | Nested expressions represented by expression graphs |

This list describes the stated implementation scope. It does not guarantee that every syntax variant, edge case, or arbitrary expression is supported.

For operations requiring symbolic integration, equation solving, or domain-sensitive transformations, correctness must be evaluated against explicitly defined mathematical rules.

---

## Dataset

The reported training dataset contains approximately 8.27 million generated samples distributed across nine categories.

| Category | Samples | Reported Share |
|---|---:|---:|
| ARI | 1,040,000 | 12.58% |
| LIN | 2,230,000 | 26.96% |
| SYS | 200,000 | 2.45% |
| QAD | 200,000 | 2.44% |
| DIF | 200,000 | 2.45% |
| INT | 200,000 | 2.44% |
| LIM | 200,000 | 2.44% |
| MAT | 600,000 | 7.31% |
| SPC | 3,390,000 | 40.93% |
| **Total** | **8,260,000** | **100% approximately** |

The category abbreviations should be mapped to their full names in the dataset-generation documentation.

The figures above are the reported distribution. The exact sample count, duplicate policy, and split boundaries should be confirmed against the generated dataset manifests before publication.

### Training and Validation

The reported experiment uses approximately:

- Training samples: 8.27 million
- Validation samples: 1.625 million
- Combined samples: approximately 9.89 million

Training and validation data should be separated by expression template or structural family where possible, rather than relying exclusively on random sample-level splitting. This helps measure generalization to unseen mathematical structures and reduces the risk of near-duplicate leakage.

---

## Reported Experimental Results

The following figures were reported for the NMN v4.0 implementation. They are included as experiment-reported results, not independently verified claims.

| Metric | Reported Result |
|---|---:|
| Top-1 operator accuracy | 100% |
| Macro F1 | 100% |
| NMC median absolute error | 0.0338 |
| Mean inference latency per step | 0.0183 ms |
| Throughput | 19,459 probabilities/s |
| Reported division/domain violations | 0% |
| Symbolic contract pass rate | 96.55% |
| Reasoning-turn reduction | 57.5% |
| Strategy Memory speedup | 38.5×–85× |
| Peak RAM usage | 893 MB |

### Interpretation

These metrics measure different aspects of the system and should not be treated as interchangeable.

- **Operator accuracy and macro F1:** require a clearly defined evaluation set, label mapping, and class distribution.
- **Numerical error:** should specify the target representation, unit of measurement, test distribution, and treatment of exact versus approximate answers.
- **Inference latency and throughput:** should include hardware, software versions, batch size, warm-up policy, and whether preprocessing or symbolic execution is included.
- **Symbolic contract pass rate:** requires a definition of the contracts, the number of test cases, and failure categories.
- **Strategy Memory speedup:** should report baseline latency and cache-hit rate, as well as results for unseen expressions.
- **Zero domain violations:** should specify the number and types of domain-sensitive tests used.

A 100% operator-classification score does not automatically establish 100% mathematical answer accuracy. Likewise, a 96.55% symbolic contract pass rate indicates that the tested contract suite was not passed perfectly; the remaining failures should be categorized and investigated.

A complete evaluation should compare NMN with a conventional symbolic solver and relevant neural or neuro-symbolic baselines under the same test conditions.

---

## Installation

### Requirements

- Python 3.10 or later, subject to compatibility with the current source code
- PyTorch version compatible with the checkpoint and model implementation
- Dependencies specified by the project

Clone the repository:

```bash
git clone https://github.com/christopherhamonangan/Neumathway.git
cd Neumathway
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present in the repository, install the dependencies documented by the source code instead. Pin tested dependency versions before publishing a reproducible release.

---

## Training

The following commands reflect the reported script names and must be checked against each script's actual command-line arguments.

### 1. Generate the dataset

```bash
python generate_dataset.py
```

The dataset generator should document category definitions, generation parameters, random seeds, output format, and train-validation splitting.

### 2. Train NMN

```bash
python train_nmn.py
```

The reported configuration includes:

| Parameter | Value |
|---|---:|
| Hidden dimension | 256 |
| Graph encoder layers | 4 |
| Batch size | 256 |
| Epochs | 8 |

These values are the reported configuration, not a guarantee of the defaults used by every script revision.

### 3. Evaluate the model

```bash
python benchmark_eval.py
```

### 4. Run rigorous benchmarks

```bash
python deep_rigorous_benchmark.py
```

For reproducible results, record the checkpoint hash, configuration, random seed, dependency versions, dataset version, hardware, and evaluation command alongside every benchmark report.

---

## Inference

The exact inference API depends on the implementation in the repository.

A minimal usage example should be added after confirming the actual model-loading interface. For example, if the project exposes a Python function named `solve`, document its real signature and return type.

```python
# Illustrative example only.
# Replace this with the actual public API from the repository.

from neumathway import solve

result = solve("1 + 2")
print(result)
```

Do not use the example above as a working command unless the corresponding module and function exist. A published release should include at least one tested example that loads the checkpoint and solves a supported expression from end to end.

---

## Model Configuration

The reported v4.0 architecture includes:

| Component | Configuration |
|---|---|
| Expression representation | Typed expression graph |
| Node feature dimensions | 8 |
| Graph encoder | Relation-aware graph attention |
| Graph encoder depth | 4 layers |
| Reasoning controller | 2-layer GRU with cross-attention |
| Operator prediction | 16 action classes |
| Numerical pathway | Continuous latent representation |
| Strategy selection | Soft decision-tree router |
| Symbolic execution | Dedicated symbolic engine |
| Strategy reuse | Strategy Memory fast path |

The total parameter count has not been specified in this document. It should be measured from the released model checkpoint rather than inferred from hidden dimensions alone.

For a PyTorch model, a basic parameter count can be obtained using:

```python
total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)
```

This snippet assumes that `model` has already been instantiated and loaded according to the project's actual implementation.

---

## Limitations

NMN is an experimental system, and several limitations should be considered.

1. **Evaluation scope:** Performance on generated expressions may not generalize to arbitrary mathematical notation or externally sourced problems.
2. **Structural generalization:** Generalization to unseen expression structures must be measured separately from accuracy on familiar templates.
3. **Symbolic correctness:** Deterministic symbolic execution is only reliable within the operations, assumptions, and domain checks correctly implemented by the engine.
4. **Numerical precision:** Floating-point output and learned numerical predictions may differ from exact rational or symbolic answers.
5. **Benchmark comparability:** Latency, throughput, and memory results are meaningful only when the measurement setup is documented.
6. **Strategy reuse:** Cached strategies can improve repeated workloads but may provide less benefit on workloads dominated by unseen structures.
7. **Reproducibility:** Dataset manifests, checkpoint versions, dependency versions, and benchmark scripts are needed to independently verify reported results.

The project should not be described as universally correct or superior to general-purpose models without comprehensive, reproducible comparative evaluation.

---

## Reproducibility

To make NMN useful to other researchers, the project should provide:

- Dataset generation scripts and category definitions
- Train-validation-test split manifests
- Model configuration files
- Checkpoint files and their hashes
- Dependency versions
- Training logs and random seeds
- Benchmark scripts and raw results
- Symbolic-engine unit tests
- Explicit definitions of every reported metric
- Baseline comparisons using the same evaluation data

A separate held-out test set should be used for final performance reporting. For mathematical reasoning, tests should include unseen compositions, larger numerical values, nested expressions, edge cases, and domain-invalid inputs.

---

## Citation

If you use NMN in research, please cite the project using the following metadata after the publication details are finalized.

```bibtex
@software{hamonangan_neumathway,
  author = {Hamonangan, Christopher},
  title = {Neumathway: A Neuro-Symbolic Mathematical Reasoning Engine},
  year = {2026},
  url = {https://github.com/christopherhamonangan/Neumathway}
}
```

This software citation identifies the repository. It should not be interpreted as a citation to a peer-reviewed research paper.

---

## License

This project is distributed under the Apache License 2.0, provided that the repository's `LICENSE` file contains the corresponding license text.

See the [LICENSE](LICENSE) file for the full terms.

---

## Author

**Christopher Hamonangan**

Independent AI Researcher

Research interests include neural architectures, mathematical reasoning, neuro-symbolic systems, and efficient machine learning.

- GitHub: [christopherhamonangan](https://github.com/christopherhamonangan)
- Project repository: [Neumathway](https://github.com/christopherhamonangan/Neumathway)

---

*Neumathway is an experimental research project. Architecture descriptions and reported metrics should be evaluated against the released source code, checkpoint, and reproducible benchmark results.*
