---
name: Bug Report
about: Create a report to help us improve sparse_embedding
title: "[Bug]: "
labels: ["bug", "triage"]
assignees: ""
---

<!--
Before reporting, please verify that the issue has not already been reported by searching existing issues.
-->

### 🐛 Bug Description

A clear and concise description of what the bug is.

### 🧩 Affected Component

- [ ] C++ Core / Storage Engine (Slot Directory, Page Directory, SeqLock)
- [ ] Python Bindings / DLPack Bridge (Nanobind, _core)
- [ ] Checkpointing & Storage (Lance, Gravitino, 2PC)
- [ ] Build & Packaging (Bazel, setup.py, pip, wheel)
- [ ] Optimizers & SIMD (AVX-512, OptimizerExtensionFacade)
- [ ] Partitioning & Routing (PartitionerRegistry, Reference Vectors)
- [ ] Other (please specify)

### 🔬 Minimal Reproducible Example

Please provide a self-contained code snippet or CLI commands to reproduce the issue:

```python
import sparse_embedding

# Minimal code snippet here
```

### 🎯 Expected Behavior

A clear and concise description of what you expected to happen.

### 📋 Logs and Tracebacks

<details>
<summary>Click to view logs and tracebacks</summary>

```shell
# Paste your terminal output, compiler/linker logs, or traceback here
```

</details>

### 💻 Environment Info

- **OS & Architecture**: (e.g. Ubuntu 24.04 x86_64)
- **CPU Model & AVX-512 Support**: (e.g. Intel Xeon / lscpu)
- **GCC / Clang Version**: 
- **Python Version**: 
- **Bazel Version**: 
- **`sparse_embedding` Version or Git Commit**: 
- **Lance Integration Enabled (`SE_BUILD_WITH_LANCE`)**: [Yes / No]

### ✅ Checklist

- [ ] I have searched existing issues and verified this is not a duplicate.
- [ ] I have provided a minimal reproducible example or actionable stack trace.
