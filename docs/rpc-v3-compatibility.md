# Aeloon RPC 3.0.0 compatibility report

This file is generated from `docs/rpc-v3.json`; do not edit by hand.

Minor-compatible changes are additive optional methods, events, fields, and error codes.
Removing an existing item, narrowing a shape, or changing existing semantics requires a major version.

## Method differences

| Method | Change | Source | Note |
| --- | --- | --- | --- |
| `system.handshake` | `changed` | `core` | 删掉 attachment_roots（客户端不再传本机路径）；protocol 从字面量升为 semver 区间协商；新增 auth 与 workspace_roots。limits.file_bytes 保持 25 MiB，单帧上限为 40 MiB |
| `system.health` | `same` | `core` | 原样保留 |
| `system.capabilities` | `new` | `new` | UI 显示什么面板的唯一来源。插件启停后发 capabilities.updated |
| `system.snapshot` | `changed` | `workbench` | 原 app.snapshot。移除 UI_VERSION 与 host（改由客户端本地探测自己的宿主） |
| `system.shutdown` | `changed` | `core` | 单租户下无权限位——通过鉴权即所有者。不做共享会话，因此不需要区分「关掉自己的」和「关掉别人的」 |
| `system.uninstall_inspect` | `changed` | `workbench` | 只报告 Runtime 侧数据；客户端本地 app 数据由客户端自己算 |
| `system.uninstall_prepare` | `same` | `workbench` | 活动 operation 或脏 worktree 时拒绝 |
| `events.subscribe` | `changed` | `core` | 只改 session_ids → thread_ids。断线续传机制**今天已经有**：adapter 用 maxlen 有界 deque 保留事件，after_seq 增量重放，replay_complete 告诉客户端缓冲区是否已经滚过去（滚过去就必须走 thread.get 全量重取），server_instance_id 用于识别 Runtime 重启。v3 原样保留，不要重新发明 |
| `project.list` | `new` | `new` | 从 app.snapshot 里拆出来，让项目列表可以单独刷新 |
| `project.add` | `changed` | `workbench` | path 现在是 Runtime 主机上的路径，必须落在 workspace_roots 内 |
| `project.remove` | `same` | `workbench` | 级联关闭该项目下所有线程的 PTY |
| `project.refresh` | `same` | `workbench` | 重新探测名称与是否 git 仓 |
| `thread.create` | `changed` | `both` | 今天要 4 次跨进程调用（session.create + settings.get + session.configure + 写库），合并后是一个事务 |
| `thread.list` | `changed` | `workbench` | 吸收 thread.archived.list |
| `thread.get` | `changed` | `workbench` | 原 thread.snapshot。合并后不再需要 flushEvents 与 refreshSessionSnapshot 的双路径 |
| `thread.delete` | `changed` | `both` | 运行中拒绝；连带删附件、关 PTY、删 session |
| `thread.rename` | `changed` | `both` | 今天要同时写 core session 和本地库，合并后一次 |
| `thread.configure` | `changed` | `both` | 同上 |
| `thread.reorder` | `same` | `workbench` | 拖拽排序持久化 |
| `thread.set_pinned` | `changed` | `workbench` | 原 thread.pin，改为显式布尔 |
| `thread.set_archived` | `changed` | `workbench` | 合并 thread.archive 与 thread.unarchive |
| `thread.set_read` | `changed` | `workbench` | 合并 thread.read 与 thread.unread |
| `thread.context` | `changed` | `workbench` | 今天要 catalog.get + session.get 两次调用，合并后一次 |
| `thread.tree` | `changed` | `core` | 原 session.tree。thread 与 session 定为 1:1，分支是 session 内部概念，方法签名不带 session_id |
| `thread.navigate` | `changed` | `core` | 原 session.navigate |
| `thread.compact` | `changed` | `core` | 原 session.compact。注意与 Memory 插件区分：这是会话内压缩，不是跨会话记忆 |
| `thread.next_turn` | `changed` | `core` | 原 session.next_turn |
| `turn.start` | `changed` | `both` | attachments 从内联对象改为 attachment_ids，配合 attachment.upload |
| `turn.cancel` | `changed` | `both` | 今天 workbench 叫 turnId、core 叫 operation_id，统一为 operation_id |
| `turn.steer` | `same` | `core` | workbench 今天没暴露，v3 直接给 UI |
| `turn.follow_up` | `same` | `core` | 同上 |
| `fs.roots` | `new` | `new` | 替代 workspace.choose。根由 Runtime 启动参数 --workspace-root 声明，默认为启动目录 ./，不暴露 $HOME。桌面启动器必须显式传参 |
| `fs.list` | `changed` | `workbench` | 原 file.list。必须做 workspace 越界检查 |
| `fs.read` | `same` | `workbench` | 原 file.read，仍是预览用途 |
| `git.status` | `same` | `workbench` |  |
| `git.github_status` | `changed` | `workbench` | 原 git.github.status（三段命名）。authenticated 语义从「笔记本已登录」变成「Runtime 主机已登录」 |
| `git.diff` | `same` | `workbench` | 带 path 时返回单文件 diff |
| `git.changes` | `same` | `workbench` |  |
| `git.stage` | `same` | `workbench` | paths 省略则全部 |
| `git.unstage` | `same` | `workbench` |  |
| `git.branches` | `same` | `workbench` |  |
| `git.branch_create` | `changed` | `workbench` | 原 git.branch.create |
| `git.commit` | `same` | `workbench` |  |
| `git.push` | `same` | `workbench` | 凭据在 Runtime 主机上 |
| `git.pr_create` | `changed` | `workbench` | 原 git.pr.create。需要 Runtime 主机上 gh 已登录 |
| `terminal.open` | `same` | `workbench` | PTY 跑在 Runtime 主机上 |
| `terminal.input` | `same` | `workbench` | 高频小包，远程时是背压主要来源 |
| `terminal.resize` | `same` | `workbench` |  |
| `terminal.close` | `same` | `workbench` |  |
| `attachment.upload` | `changed` | `workbench` | 原 attachment.save。今天 UI 已经在传 base64，所以远程化不需要新造上传通道——只需要让 Runtime 返回 id 而不是路径。单次传完，上限见握手 limits.file_bytes |
| `attachment.delete` | `changed` | `workbench` | 参数从 {id, path} 变成只要 id |
| `attachment.preview` | `changed` | `workbench` | 同上 |
| `attachment.download` | `new` | `new` | 替代 attachment.open / attachment.reveal。Runtime 给字节，客户端在本机打开或另存 |
| `artifact.resolve` | `same` | `workbench` |  |
| `artifact.download` | `new` | `new` | 替代 artifact.open / artifact.reveal |
| `catalog.get` | `same` | `both` | workbench 今天是纯代理，合并后直连 |
| `provider.list` | `same` | `both` | 纯代理，合并后直连 |
| `provider.refresh` | `same` | `both` | 同上 |
| `provider.add` | `same` | `both` | 同上 |
| `provider.remove` | `same` | `both` | 同上 |
| `settings.get` | `same` | `both` | 纯代理，合并后直连。必须一并返回 plugins 段——与 plugins.configure 共用同一份 Config 与同一个 revision |
| `settings.update` | `same` | `both` | 乐观并发，冲突返回 revision_conflict。与 plugins.configure 是同一个写路径的两个入口 |
| `tools.search_test` | `changed` | `both` | 原 tools.search.test（三段命名） |
| `plugins.list` | `new` | `new` | 比 system.capabilities 更细：含未启用、加载失败的插件与失败原因 |
| `plugins.set_enabled` | `new` | `new` | L1 热插拔。disable 时把 Port 换成 Null 实现，不重启 Runtime |
| `plugins.configure` | `new` | `new` | settings 由插件自己的 pydantic model 校验；失败只禁用该插件并上报。与 settings.update 写同一份 Config、共用 revision；单独成方法是因为它还要触发插件生命周期（重新装配 Provider） |

## Plugin-contributed methods

- `plugin.cloud.account_status` — contributed namespace; available as the hard-wired Cloud capability
- `plugin.cloud.account_login` — contributed namespace; available as the hard-wired Cloud capability
- `plugin.cloud.account_logout` — contributed namespace; available as the hard-wired Cloud capability
- `plugin.memory.remember` — contributed namespace; returns capability_unavailable in the base release
- `plugin.memory.recall` — contributed namespace; returns capability_unavailable in the base release
- `plugin.memory.forget` — contributed namespace; returns capability_unavailable in the base release
- `plugin.knowledge.index` — contributed namespace; returns capability_unavailable in the base release
- `plugin.knowledge.search` — contributed namespace; returns capability_unavailable in the base release
- `plugin.knowledge.stat` — contributed namespace; returns capability_unavailable in the base release

## Removed or collapsed legacy surface

- `workspace.choose` — 客户端本地: 调 Electron 原生目录对话框。远程 Runtime 上无意义。改用 fs.roots + fs.list 做远程浏览器
- `attachment.open` — 客户端本地: 在 Runtime 主机的桌面上打开文件是错的。改用 attachment.download 后由客户端本地打开
- `attachment.reveal` — 客户端本地: 同上，「在访达中显示」在远程语境下没有意义
- `artifact.open` — 客户端本地: 同上
- `artifact.reveal` — 客户端本地: 同上
- `core.mark_interrupted` — 合并后消失: 跨进程重连后把活动 turn 标记为中断的对账动作，进程内不需要
- `thread.archive / thread.unarchive` — 合并: → thread.set_archived
- `thread.read / thread.unread` — 合并: → thread.set_read
- `thread.pin` — 合并: → thread.set_pinned（显式布尔）
- `thread.archived.list` — 合并: → thread.list(filter="archived")
- `session.* 全部 10 个` — 改名: session.create/list/get/delete/rename/configure/tree/navigate/compact/next_turn → thread.*。thread 是持久身份，session 降为内部投影
- `13 个代理方法` — 塌缩: turn.start/cancel、catalog.get、provider.*4、settings.*2、tools.search.test、cloud.account.*3 今天在 workbench 和 core 各有一份，合并后只剩一份

## Event differences

- `session.compacted` → `thread.compacted`
- `session.navigated` → `thread.navigated`
- `session.renamed` → `thread.renamed`
- `cloud.account.updated` → `plugin.cloud.account_updated`
- `capabilities.updated` — new
- `plugin.<id>.*` — new
- `core.ready / core.disconnected / core.unresponsive` — removed: 没有跨进程 core 连接了
- `core.events` — removed: 批量封套消失，事件直接投递
- `server.ready` — removed: workbench 进程不存在了
- `thread.reconciled` — removed: 进程内无需对账

Envelope: 所有事件的 session_id 字段改为 thread_id；需要时附带 session_id 作为内部投影引用

## Error differences

- `unauthorized` (-32011) — 握手未通过鉴权，或方法需要更高权限位
- `forbidden` (-32012) — 路径越出 workspace_roots；远程连接调用受限方法如 system.shutdown
- `capability_unavailable` (-32013) — 调了未启用插件贡献的方法
- `payload_too_large` (-32014) — 超出握手声明的 limits
- `session_not_found` → `thread_not_found` — 语义跟随身份改名
