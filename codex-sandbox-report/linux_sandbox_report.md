# OpenAI Codex 沙箱实现分析 - Linux 篇

## 高层概述

与 macOS 不同，Linux 没有一个统一的、像 Seatbelt 那样的沙箱框架。因此，Codex 在 Linux 上的沙箱实现采用了“纵深防御”的策略，结合了两种不同的内核安全技术：**Landlock** 和 **seccomp-bpf**。

*   **Landlock**: 用于限制文件系统访问。它允许一个进程限制其自身以及其子进程对文件系统的访问权限，例如，将写权限限制在特定的目录树中。
*   **seccomp-bpf**: 用于过滤系统调用（syscall）。Codex 使用 seccomp 来阻止所有与网络相关的系统调用，从而实现网络隔离。

这种组合提供了一个与 macOS Seatbelt 功能相似的沙箱环境，即同时控制文件系统和网络访问。

### 沙箱允许和拒绝的行为

*   **文件系统访问 (Landlock)**:
    *   **只读模式**: 沙箱默认允许对整个文件系统的只读访问。
    *   **工作区写入模式**: 当配置为 `workspace-write` 时，沙箱会明确地为工作区目录和临时目录授予写权限。所有在这些目录之外的写操作都会被内核拒绝。

*   **网络访问 (seccomp)**:
    *   默认情况下，沙箱会安装一个 seccomp 过滤器，该过滤器会拦截并拒绝大多数与网络相关的系统调用，例如 `connect`, `bind`, `accept` 等。
    *   唯一的例外是 `AF_UNIX` 域套接字，它被允许用于本地进程间通信。这使得一些需要使用套接字对（socketpair）进行子进程管理的工具（如 `cargo clippy`）能够正常工作。
    *   当网络访问被明确启用时，这个 seccomp 过滤器不会被安装。

## 实现细节

Codex 的 Linux 沙箱实现主要分布在 `codex-rs/linux-sandbox` 和 `codex-rs/core` 这两个 crate 中。与 macOS 的实现方式不同，Linux 的沙箱是通过一个独立的帮助程序（`codex-linux-sandbox`）来应用的。

### 沙箱的调用流程

1.  **主进程 (`codex-exec`)**: 当需要执行一个被沙箱化的命令时，主进程不会直接应用 Landlock 和 seccomp 规则。相反，它会构造一个命令来调用 `codex-linux-sandbox` 这个帮助程序。
2.  **策略传递**: `SandboxPolicy` 对象会被序列化成一个 JSON 字符串，并通过命令行参数 `--sandbox-policy` 传递给帮助程序。

    ```rust
    // codex-rs/core/src/landlock.rs

    pub(crate) fn create_linux_sandbox_command_args(
        command: Vec<String>,
        sandbox_policy: &SandboxPolicy,
        sandbox_policy_cwd: &Path,
    ) -> Vec<String> {
        // ...
        let sandbox_policy_json =
            serde_json::to_string(sandbox_policy).expect("Failed to serialize SandboxPolicy to JSON");

        let mut linux_cmd: Vec<String> = vec![
            // ...
            "--sandbox-policy".to_string(),
            sandbox_policy_json,
            "--".to_string(),
        ];
        // ...
    }
    ```

3.  **帮助程序 (`codex-linux-sandbox`)**: 这个帮助程序在启动后，会首先解析传入的 `SandboxPolicy` JSON。然后，它会调用 `apply_sandbox_policy_to_current_thread` 函数，在该线程中应用 Landlock 和 seccomp 规则。

4.  **执行命令**: 在应用了沙箱规则之后，帮助程序会 `exec` 到用户请求的实际命令。由于 Landlock 和 seccomp 的规则在 `exec` 之后仍然有效，所以最终的命令将在沙箱的保护下运行。

### 文件系统沙箱 (Landlock)

`install_filesystem_landlock_rules_on_current_thread` 函数负责设置 Landlock 规则。

```rust
// codex-rs/linux-sandbox/src/landlock.rs

fn install_filesystem_landlock_rules_on_current_thread(writable_roots: Vec<PathBuf>) -> Result<()> {
    let abi = ABI::V5;
    let access_rw = AccessFs::from_all(abi);
    let access_ro = AccessFs::from_read(abi);

    let mut ruleset = Ruleset::default()
        // ...
        .add_rules(landlock::path_beneath_rules(&["/"], access_ro))? // 1. 全局只读
        .add_rules(landlock::path_beneath_rules(&["/dev/null"], access_rw))? // 2. /dev/null 可写
        .set_no_new_privs(true);

    if !writable_roots.is_empty() {
        ruleset = ruleset.add_rules(landlock::path_beneath_rules(&writable_roots, access_rw))?; // 3. 工作区可写
    }

    ruleset.restrict_self()?; // 4. 应用规则

    Ok(())
}
```

### 网络沙箱 (seccomp)

`install_network_seccomp_filter_on_current_thread` 函数使用 `seccompiler` 库来构建一个 BPF 程序，该程序会过滤掉所有危险的网络系统调用。

```rust
// codex-rs/linux-sandbox/src/landlock.rs

fn install_network_seccomp_filter_on_current_thread() -> std::result::Result<(), SandboxErr> {
    let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();

    let mut deny_syscall = |nr: i64| {
        rules.insert(nr, vec![]);
    };

    deny_syscall(libc::SYS_connect);
    deny_syscall(libc::SYS_accept);
    // ... more denied syscalls

    // 只允许 AF_UNIX 域的 socket
    let unix_only_rule = SeccompRule::new(vec![SeccompCondition::new(
        0, // 第一个参数 (domain)
        SeccompCmpArgLen::Dword,
        SeccompCmpOp::Ne,
        libc::AF_UNIX as u64,
    )?])?;

    rules.insert(libc::SYS_socket, vec![unix_only_rule.clone()]);

    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,                     // 默认允许
        SeccompAction::Errno(libc::EPERM as u32), // 规则匹配时返回 EPERM
        // ...
    )?;

    let prog: BpfProgram = filter.try_into()?;
    apply_filter(&prog)?;

    Ok(())
}
```

通过这种方式，Codex 在 Linux 上实现了一个健壮的沙箱，尽管它需要一个额外的帮助程序，但其安全性和灵活性与 macOS 的实现相当。
