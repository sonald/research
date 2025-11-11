# OpenAI Codex 沙箱实现分析 - macOS 篇

## 高层概述

OpenAI Codex CLI 在 macOS 上的沙箱实现，依赖于苹果操作系统内置的 Seatbelt（`sandbox-exec`）技术。Seatbelt 是一个强大的内核级安全工具，它通过一套规则（称为“配置文件”）来限制进程的行为，例如文件系统访问、网络通信和系统调用。

Codex 的沙箱设计核心思想是“默认拒绝”，即除非明确允许，否则一切操作都被禁止。沙箱的严格程度由用户的配置（通过命令行参数或 `config.toml` 文件）动态决定。

### 沙箱允许和拒绝的行为

*   **文件系统访问**:
    *   **只读模式 (`read-only`)**: 默认情况下，沙箱允许读取整个文件系统，但禁止任何写操作。
    *   **工作区写入模式 (`workspace-write`)**: 当一个目录被信任后，沙箱允许在当前工作目录（以及临时目录）内进行写操作。为了增加安全性，对 `.git` 目录的写操作通常是禁止的，以防止对版本控制的破坏。
    *   **完全访问 (`full-access`)**: 可以配置为允许对整个文件系统的完全读写权限。

*   **网络访问**:
    *   默认情况下，沙箱会阻止所有出站和入站的网络连接。
    *   当用户明确启用网络访问后，沙箱会放开网络限制，但仍然会通过 `mach-lookup` 对一些系统服务的访问进行控制。

*   **进程执行**:
    *   沙箱允许创建子进程（`process-fork` 和 `process-exec`），但子进程会继承父进程的沙箱配置文件，从而确保安全策略的延续。

## 实现细节

Codex 的沙箱实现主要位于 `codex-rs/core/src/seatbelt.rs` 文件中。其核心是 `create_seatbelt_command_args` 函数，该函数负责动态生成 Seatbelt 的配置文件，并构建传递给 `/usr/bin/sandbox-exec` 的命令行参数。

### 动态配置文件生成

配置文件的生成过程如下：

1.  **基础策略**: 加载一个基础的、高度限制性的策略文件 `seatbelt_base_policy.sbpl`。这个文件以 `(deny default)` 开头，确立了“默认拒绝”的原则，并只允许一些最基本的操作，例如进程创建、读取用户偏好设置和一些安全的系统信息查询。

    ```scheme
    ; codex-rs/core/src/seatbelt_base_policy.sbpl

    (version 1)

    ; start with closed-by-default
    (deny default)

    ; child processes inherit the policy of their parent
    (allow process-exec)
    (allow process-fork)
    (allow signal (target same-sandbox))
    ```

2.  **文件访问策略**: 根据用户的配置（`read-only`, `workspace-write`, 等），动态生成文件访问规则。

    *   在 `workspace-write` 模式下，代码会获取所有可写的目录，并为每个目录生成一个 `(subpath (param "WRITABLE_ROOT_X"))` 规则。这些路径参数随后会通过 `-D` 标志传递给 `sandbox-exec`。

        ```rust
        // codex-rs/core/src/seatbelt.rs

        let (file_write_policy, file_write_dir_params) = {
            if sandbox_policy.has_full_disk_write_access() {
                // ...
            } else {
                let writable_roots = sandbox_policy.get_writable_roots_with_cwd(sandbox_policy_cwd);
                // ...
                for (index, wr) in writable_roots.iter().enumerate() {
                    // ...
                    let root_param = format!("WRITABLE_ROOT_{index}");
                    file_write_params.push((root_param.clone(), canonical_root));
                    // ...
                }
            }
        };
        ```

3.  **网络策略**: 如果用户启用了网络访问，`seatbelt_network_policy.sbpl` 文件的内容会被附加到策略中。这个文件允许了 `network-outbound` 和 `system-socket` 等操作。

    ```scheme
    ; codex-rs/core/src/seatbelt_network_policy.sbpl

    (allow network-outbound)
    (allow system-socket)
    ```

4.  **最终命令**: `create_seatbelt_command_args` 函数将所有这些部分组合成一个完整的 `sandbox-exec` 命令，包括策略字符串（通过 `-p` 标志）和所有路径参数（通过 `-D` 标志）。

    ```rust
    // codex-rs/core/src/seatbelt.rs

    let mut seatbelt_args: Vec<String> = vec!["-p".to_string(), full_policy];
    let definition_args = dir_params
        .into_iter()
        .map(|(key, value)| format!("-D{key}={value}", value = value.to_string_lossy()));
    seatbelt_args.extend(definition_args);
    seatbelt_args.push("--".to_string());
    seatbelt_args.extend(command);
    ```

### 沙箱的调用

`spawn_command_under_seatbelt` 函数负责调用 `sandbox-exec` 来执行被沙箱化的命令。它会硬编码 `sandbox-exec` 的路径为 `/usr/bin/sandbox-exec`，以防止通过 PATH 注入恶意程序。

```rust
// codex-rs/core/src/seatbelt.rs

pub(crate) const MACOS_PATH_TO_SEATBELT_EXECUTABLE: &str = "/usr/bin/sandbox-exec";

pub async fn spawn_command_under_seatbelt(
    // ...
) -> std::io::Result<Child> {
    let args = create_seatbelt_command_args(command, sandbox_policy, sandbox_policy_cwd);
    // ...
    spawn_child_async(
        PathBuf::from(MACOS_PATH_TO_SEATBELT_EXECUTABLE),
        args,
        // ...
    )
    .await
}
```

总的来说，Codex 在 macOS 上的沙箱实现是一个经过深思熟虑的设计，它有效地利用了操作系统原生的安全特性，并通过动态生成配置文件的方式，在安全性和灵活性之间取得了很好的平衡。
