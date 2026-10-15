# 构建与测试指南

本文档阐述 `sparse_embedding` 的混合构建架构、打包契约以及测试验证流程。

---

## 1. 架构与构建契约

`sparse_embedding` 采用分工明确的双层混合构建架构：

```
[pip / build / setuptools]
       │
       ▼ (PEP 517 构建前端)
    setup.py ──────────────────► 组装安装包、生成类型存根、命名空间包发现
       │
       ▼ (子进程调用)
  Bazel 系统 ──────────────────► C++23 编译、Nanobind 绑定、SIMD 与底层优化
```

- **Bazel 作为底层的原生构建引擎**：
  - 负责编译高性能 C++23 核心库（`//:core_extension`）、SIMD 优化器、Hogwild 存储引擎以及 DLPack 零拷贝桥接代码。
  - 承载底层原生单元与集成测试（`bazel test //sparse_embedding/...`）。
  - 通过统一的 `//:native_artifacts` filegroup 导出编译好的动态链接库。
- **Setuptools 作为分发与打包单一主体**：
  - 作为标准的 PEP 517 构建后端，处理 `pip install`、editable 开发安装以及二进制 wheel 打包。
  - 动态查询 Bazel 的 `//:native_artifacts` 产物集合，避免硬编码输出路径，确保与当前构建配置严格对齐。
  - 自动生成 PEP 561 类型标记（`py.typed`）与 Nanobind 类型存根（`_core.pyi`）。
  - 严格隔离测试代码，防止测试文件泄露至分发包中。

---

## 2. 特性开关与构建配置

Lance 检查点集成通过统一的特性开关在两层构建体系间同步传递：

| 配置选项 | Bazel 选项 | 环境变量 (pip) | 打包的原生动态库 |
| :--- | :--- | :--- | :--- |
| **启用 Lance** *(默认)* | `--//:with_lance=true` | `SE_BUILD_WITH_LANCE=1` | `_core.so`, `liblance_c.so` |
| **独立核心 (禁用 Lance)** | `--//:with_lance=false` | `SE_BUILD_WITH_LANCE=0` | `_core.so` |

---

## 3. 标准操作流程与常用命令

### Python 安装与打包

```bash
# 从源码编译并安装
pip install .

# 本地开发模式 (Editable install)
pip install -e .

# 独立构建 (不依赖 Lance)
SE_BUILD_WITH_LANCE=0 pip install -e .

# 构建标准的二进制 wheel 分发包
pip wheel . --no-build-isolation
```

### 运行测试

```bash
# 安装测试依赖
pip install -e ".[test]"

# 运行完整的 Python 测试套件
pytest

# 仅运行非慢速单元测试
pytest -m "not slow"
```

### 通过 Bazel 构建与测试 C++ 目标

```bash
# 编译所有经过优化的 C++ 目标
bazel build -c opt //sparse_embedding/...

# 运行所有 C++ 单元测试与集成测试
bazel test -c opt //sparse_embedding/...
```

---

## 4. 打包一致性与质量约束

所有构建流程均强制执行以下打包规范：

1. **分发树纯净性**：`setup.py` 中的命名空间包发现严格排除 `sparse_embedding.tests`，避免测试用例与测试数据打入正式 wheel 包。
2. **失效产物清理**：在复制最新 Bazel 产物前，`setup.py` 会清空目标目录并移除历史残留的 `.so` 库，防止多配置切换时产生 ABI 混杂。
3. **类型契约完整性**：分发包内同时包含 `py.typed` 标记与自动生成的 `_core.pyi` 存根，为下游使用者提供完整的静态类型推导与 IDE 智能补全。
