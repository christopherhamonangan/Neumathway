# Neumathway (NMN)

### Neuro-Symbolic Mathematical Reasoning Engine

Neumathway (NMN) is an experimental neuro-symbolic mathematical reasoning engine designed to process structured mathematical expressions through graph-based neural representations, recurrent reasoning, neural operator prediction, and symbolic computation.

Rather than treating mathematical problems exclusively as natural-language sequences, NMN represents expressions as structured computational graphs. Its architecture combines learned decision-making, numerical prediction, adaptive strategy selection, and deterministic symbolic execution for supported mathematical operations.

NMN explores specialized mathematical reasoning through a combination of neural computation and symbolic methods, with an emphasis on structural representation, computational efficiency, and reproducible evaluation.

**Repository:** [github.com/christopherhamonangan/Neumathway](https://github.com/christopherhamonangan/Neumathway)

---

## Table of Contents

- [Overview](#overview)
- [Research Objectives](#research-objectives)
- [Architecture](#architecture)
- [Mathematical Capabilities](#mathematical-capabilities)
- [Training Dataset](#training-dataset)
- [Model Configuration](#model-configuration)
- [Reported Experimental Results](#reported-experimental-results)
- [Installation](#installation)
- [Dataset Generation](#dataset-generation)
- [Training](#training)
- [Inference](#inference)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)
- [Author](#author)

---

## Overview

Neumathway is designed to investigate whether a specialized neuro-symbolic architecture can perform mathematical reasoning by combining structured expression representations with learned computational strategies and symbolic execution.

The system integrates the following components:

- **Typed Expression Graph:** Encodes mathematical expressions as structured computational graphs.
- **Relation-Aware Graph Attention Network:** Learns contextual relationships between mathematical operators, operands, and expression nodes.
- **Recurrent Reasoning Controller:** Coordinates sequential reasoning decisions using recurrent neural computation and cross-attention.
- **Neural Operator Prediction (NOP):** Predicts mathematical operators and transformations from a predefined action space.
- **Neural Matrix Computation (NMC):** Provides a learned numerical computation pathway.
- **Adaptive Strategy Router:** Selects between supported mathematical execution strategies.
- **Symbolic Computation Engine:** Executes supported operations using deterministic mathematical procedures.
- **Strategy Memory:** Reuses compatible execution strategies to reduce repeated computational overhead.
- **Evolutionary Training Framework:** Explores candidate strategies and model configurations through evolutionary optimization.

The architecture separates learned decision-making from symbolic execution. Neural components predict operators, numerical representations, or strategies, while the symbolic engine handles operations covered by its implementation.

## Research Objectives

NMN investigates several questions in specialized mathematical reasoning:

1. Can typed expression graphs provide useful structural representations for mathematical computation?
2. Can a compact neural controller learn operator selection and mathematical execution strategies?
3. Can neural prediction and symbolic computation be combined to improve the reliability of mathematical operations?
4. Can strategy reuse reduce inference overhead for recurring expression structures?
5. What trade-offs emerge between numerical accuracy, structural generalization, latency, and memory consumption?

The intended contribution is an experimental framework for evaluating structured neural reasoning and symbolic execution together. Claims of superiority over existing solvers require controlled comparative experiments.

---

## Architecture

### 1. Typed Expression Graph and 8D Encoder

Mathematical expressions are represented as structured graphs. Each node uses an eight-dimensional feature representation:

| Feature | Description |
|---|---|
| `is_op` | Indicates whether a node represents an operator |
| `token_id` | Encoded identifier for the node token |
| `signed_log_value` | Signed logarithmic representation of numerical values |
| `depth` | Node depth in the expression structure |
| `is_left_child` | Indicates a left-child relationship |
| `is_right_child` | Indicates a right-child relationship |
| `degree` | Encoded local graph degree |
| `in_lhs` | Indicates membership in the left-hand side of an equation |

This representation is intended to preserve structural information that would otherwise need to be inferred from a flat sequence of tokens.

### 2. Relation-Aware Graph Attention Network

The graph encoder uses four relation-aware graph attention layers with residual connections.

Its purpose is to propagate contextual information across related expression nodes and represent relationships between operators, operands, and subexpressions.

### 3. Recurrent Reasoning Controller

The reasoning controller consists of a two-layer gated recurrent unit (GRU) and cross-attention.

The recurrent component maintains a hidden reasoning state, while cross-attention connects that state to information encoded in the expression graph.

### 4. Dual-Head Neural Computation

NMN includes two neural computation pathways.

**Neural Operator Prediction (NOP)**

- Uses 16 action classes.
- Incorporates a learned temperature mechanism.
- Predicts operators or transformations from the configured action space.

**Neural Matrix Computation (NMC)**

- Predicts continuous numerical representations.
- Uses signed-log numerical encoding.
- Uses Smooth L1 loss during training.

The operator definitions, numerical decoding procedure, and loss-weighting configuration are determined by the implementation.

### 5. Adaptive Strategy Router

A soft decision-tree router selects between supported execution strategies, including arithmetic tree reduction and algebraic transposition.

The router is intended to adapt execution strategy to the structure and state of a mathematical expression.

### 6. Symbolic Computation Engine

The symbolic engine implements deterministic procedures for supported mathematical operations. Exact rational arithmetic uses Python's `Fraction` where applicable.

The reported scope includes:

- Basic arithmetic
- Quadratic equations using the ABC formula
- Polynomial differentiation
- Indefinite polynomial integration
- Determinants of \(2 \times 2\) matrices
- Systems of two linear equations in two variables (SPLDV)

The engine's correctness depends on its implementation, domain checks, supported syntax, and output validation.

### 7. Strategy Memory

Strategy Memory provides a fast execution path for compatible, recurring expression structures.

When a stored strategy can be reused, the system can bypass some neural inference steps. The resulting speedup depends on strategy matching, cache-hit rate, lookup overhead, and workload composition.

### 8. Evolutionary Training Framework

The reported training design includes:

- A population of 32 candidate agents
- Six-to-one polygenic synthesis
- Evolutionary strategy exploration
- An Occam's Razor-inspired preference for simpler candidates

Reproduction requires the corresponding selection rules, mutation or synthesis procedures, random seeds, and stopping criteria to be documented in the source code.

---

## Mathematical Capabilities

The intended mathematical scope spans elementary arithmetic through selected undergraduate-level operations.

| Domain | Supported scope |
|---|---|
| Elementary arithmetic | Nested expressions, arithmetic operators, and PEMDAS evaluation |
| Linear equations | Variable isolation, algebraic transposition, and fractions |
| Algebraic systems | Simultaneous linear equations in two variables |
| Quadratic equations | Discriminants and the ABC formula |
| Differential calculus | Polynomial differentiation and power rule |
| Integral calculus | Polynomial antiderivatives and indefinite integration |
| Limits and continuity | Supported limit expressions and rational substitution |
| Matrix algebra | \(2 \times 2\) determinants, transpose, and supported matrix operations |
| Special functions | Supported trigonometric functions, logarithms, exponentials, and discrete operations |

The listed scope does not imply support for every mathematical expression, notation, domain condition, or arbitrary problem in these fields. Supported syntax and operation-specific limitations should be defined by the implementation and test suite.

---

## Training Dataset

The active checkpoint, `nmn_checkpoint.pth`, was trained on **8.27 million pure training samples** stored in `train_shards`. Validation datasets stored in `val_shards` were kept strictly isolated from the training datasets.

The dataset was generated through a grammar-guided syntactic pipeline with domain boundary constraints across nine mathematical dataset categories.

| Dataset Domain | Code | Training Samples | Proportion | Scope and Mathematical Invariants |
|---|:---:|---:|---:|---|
| **Elementary Arithmetic** | `ARI` | 1.04 M | 12.58% | Deep arithmetic expressions, nested parentheses, PEMDAS hierarchy |
| **Linear Equations and Balancing** | `LIN` | 2.23 M | 26.96% | Single- and multi-step transposition, variable isolation, fractions, two-sided variable balance |
| **Algebraic Systems (2-Variable)** | `SYS` | 0.20 M | 2.45% | Simultaneous linear systems (SPLDV) using Cramer's rule and elimination for \(x\) and \(y\) |
| **Quadratic Equations** | `QAD` | 0.20 M | 2.44% | Polynomial roots using the discriminant \(D=b^2-4ac\) and the ABC formula |
| **Differential Calculus** | `DIF` | 0.20 M | 2.45% | Symbolic differentiation, \(d/dx\), polynomial power rule, linearity |
| **Integral Calculus** | `INT` | 0.20 M | 2.44% | Indefinite integration, \(\int f(x)\,dx\), polynomial antiderivative rule |
| **Continuity and Limits** | `LIM` | 0.20 M | 2.44% | Limit evaluation, \(\lim_{x\to c} f(x)\), rational continuity substitution |
| **Matrix and Linear Algebra** | `MAT` | 0.60 M | 7.31% | \(2 \times 2\) matrix determinants \((ad-bc)\), transpose, non-commutative matrix multiplication |
| **Special and Discrete Functions** | `SPC` | 3.39 M | 40.93% | Trigonometry (`sin`, `cos`, `tan`), logarithms (`ln`), exponentials, Pythagorean `hypot`, greatest common divisor (`gcd`), factorial (`fact`) |
| **TOTAL CORPUS** | `ALL` | **8.27 M** | **100.00%** | **K–12 to selected undergraduate pure mathematics** |

### Dataset Design

The dataset-generation pipeline uses grammar-guided expression construction and domain boundary constraints to produce mathematical samples within predefined categories.

The dataset is intended to cover:

- Nested mathematical expressions
- Algebraic transformations
- Structured equation solving
- Polynomial calculus
- Selected limit and continuity problems
- Matrix operations
- Special and discrete mathematical functions

The category proportions describe the reported training distribution. Validation data remain separate from `train_shards`.

For rigorous generalization testing, expression templates and structural families should also be separated across training, validation, and test partitions. This helps distinguish memorization of familiar expression patterns from generalization to unseen mathematical structures.

---

## Model Configuration

| Component | Configuration |
|---|---|
| Expression representation | Typed expression graph |
| Node feature dimension | 8 |
| Graph encoder | Relation-aware graph attention |
| Graph encoder depth | 4 layers |
| Reasoning controller | 2-layer GRU with cross-attention |
| Operator prediction | 16 action classes |
| Numerical pathway | Continuous latent representation |
| Strategy selection | Soft decision-tree router |
| Symbolic execution | Dedicated symbolic computation engine |
| Strategy reuse | Strategy Memory fast path |
| Training batch size | 256 |
| Training epochs | 8 |
| Hidden dimension | 256 |

These values describe the reported NMN v4.0 configuration. The total parameter count must be measured from the instantiated model or checkpoint and should not be inferred from hidden dimension alone.

---

## Reported Experimental Results

The following figures are reported for the NMN v4.0 implementation.

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

### Metric Interpretation

The metrics describe different aspects of the system and require separate evaluation protocols.

- **Top-1 operator accuracy and macro F1:** Measure operator prediction against labeled targets. The evaluation split, class definitions, and label distribution must be documented.
- **NMC median absolute error:** Measures numerical prediction error under the evaluation's defined target representation.
- **Inference latency and throughput:** Depend on hardware, software versions, batch size, warm-up policy, and which processing stages are included.
- **Division/domain violations:** Depend on the test cases and the definition of a violation.
- **Symbolic contract pass rate:** Measures compliance with the defined symbolic test contracts. A pass rate of 96.55% indicates that the tested contracts were not all passed.
- **Reasoning-turn reduction:** Requires a defined baseline and a consistent counting method.
- **Strategy Memory speedup:** Requires baseline timing, cache-hit rate, cache-miss measurements, and evaluation on unseen structures.
- **Peak RAM usage:** Depends on the measurement environment, dataset loading behavior, and the stages included in the measurement.

A 100% operator prediction score does not independently establish 100% final-answer accuracy. Likewise, deterministic symbolic execution does not guarantee correctness for unsupported operations or unhandled edge cases.

These figures should be accompanied by benchmark scripts, evaluation-set definitions, hardware specifications, and raw results to enable independent reproduction.

---

## Installation

### Requirements

- Python version compatible with the repository's source code
- PyTorch version compatible with the model implementation and checkpoint
- Dependencies listed in `requirements.txt`

Clone the repository:

```bash
git clone https://github.com/christopherhamonangan/Neumathway.git
cd Neumathway
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependency versions should be pinned to the versions used for the reported training and evaluation runs.

---

## Dataset Generation

Generate the dataset using the repository's dataset-generation script:

```bash
python generate_dataset.py
```

The generator should document its command-line arguments, category definitions, random seeds, output format, and train-validation partitioning.

The reported training dataset consists of 8.27 million samples in `train_shards`, with validation datasets isolated in `val_shards`.

---

## Training

Train the model using:

```bash
python train_nmn.py
```

The reported configuration uses a hidden dimension of 256, four graph encoder layers, a batch size of 256, and eight epochs.

For reproducibility, the training configuration should be saved with the checkpoint, together with the dataset version, random seed, dependency versions, and training logs.

---

## Evaluation

Run the standard benchmark:

```bash
python benchmark_eval.py
```

Run the rigorous benchmark suite:

```bash
python deep_rigorous_benchmark.py
```

Evaluation should report operator prediction metrics, numerical error, final-answer correctness, symbolic contract compliance, domain violations, inference latency, throughput, and memory consumption separately.

Where applicable, benchmarks should distinguish results for familiar structures, unseen expression templates, Strategy Memory cache hits, and cache misses.

---

## Inference

The inference interface should be documented according to the actual public API exposed by the repository.

A usage example should specify:

1. How to load `nmn_checkpoint.pth`.
2. How to initialize the architecture and configuration.
3. Which mathematical expression formats are accepted.
4. How results and errors are represented.
5. How unsupported operations and invalid domains are handled.

The exact API depends on the implementation. An executable example should be included once the checkpoint-loading function and inference interface are confirmed against the source code.

---

## Reproducibility

Independent reproduction of NMN's reported results requires:

- Dataset-generation scripts and category definitions
- Training and validation split manifests
- Model configuration files
- Checkpoint and dataset version identifiers
- Dependency versions and random seeds
- Training logs
- Benchmark scripts and raw outputs
- Symbolic-engine unit tests
- Definitions of each evaluation metric
- Hardware and runtime specifications
- Baseline implementations and comparison protocols

A held-out test set should be used for final performance reporting. Structural generalization should be evaluated using expression families or templates not represented in the training set.

---

## Limitations

NMN is an experimental mathematical reasoning system with a defined implementation scope.

1. **Generalization:** Performance on grammar-generated samples may not generalize to arbitrary mathematical notation or externally sourced problems.
2. **Symbolic coverage:** Deterministic execution applies only to operations correctly implemented by the symbolic engine.
3. **Numerical precision:** Learned numerical predictions may differ from exact rational or symbolic results.
4. **Domain handling:** Division by zero, undefined expressions, invalid logarithm inputs, and other domain conditions require explicit handling.
5. **Benchmark comparability:** Performance claims require consistent baselines and documented measurement protocols.
6. **Strategy Memory:** Speedup depends on structural matching, cache coverage, and workload composition.
7. **Evaluation completeness:** Operator accuracy, final-answer correctness, and symbolic contract compliance are distinct metrics and should be reported independently.

The architecture and reported measurements should be evaluated against the released implementation and reproducible experimental evidence.

---

## Citation

If Neumathway is used in research, cite the software repository using the following BibTeX entry:

```bibtex
@software{hamonangan_neumathway,
  author = {Hamonangan, Christopher},
  title = {Neumathway: A Neuro-Symbolic Mathematical Reasoning Engine},
  year = {2026},
  url = {https://github.com/christopherhamonangan/Neumathway}
}
```

This entry cites the software repository and does not imply that the project has been peer-reviewed or published as an academic paper.

---

## License

This project is distributed under the Apache License 2.0.

See the [LICENSE](LICENSE) file for the complete license terms.

---

## Author

**Christopher Alberto Hamonangan**

Independent AI Researcher

Research interests include neural architectures, mathematical reasoning, neuro-symbolic systems, and efficient machine learning.

- GitHub: [christopherhamonangan](https://github.com/christopherhamonangan)
- Project: [Neumathway](https://github.com/christopherhamonangan/Neumathway)

---

*Neumathway is an experimental research project. Its architecture, capabilities, and reported performance should be evaluated through the released source code, model checkpoint, and reproducible benchmark results.*
