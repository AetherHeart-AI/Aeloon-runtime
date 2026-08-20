# Aeloon RPC 3.0.0

This file is generated from `docs/rpc-v3.json`; do not edit by hand.

- Protocol: `aeloon-rpc`
- Frame: `40 MiB`
- File/image: `25 MiB` / `10 MiB`

## system

- `system.handshake` — `{protocol:{min,max}, client:{name,version,platform}, auth?:{scheme:"bearer"|"mtls", token?}}` → `{protocol, runtime:{version,commit}, limits:{prompt_chars,attachments,image_bytes,file_bytes,request_bytes,retained_events}, workspace_roots[]}`
- `system.health` — `{workspace?}` → `{ok, uptime_s, active_operations}`
- `system.capabilities` — `{}` → `{capabilities:[{id,version,kind,enabled,settings_schema,methods[],events[],ui:{group,order}}]}`
- `system.snapshot` — `{}` → `{default_workspace, projects[], threads[], runtime:{version,commit,protocol}}`
- `system.shutdown` — `{}` → `{accepted}`
- `system.uninstall_inspect` — `{}` → `{data_paths[], estimated_bytes, active_operations, worktrees[]}`
- `system.uninstall_prepare` — `{}` → `{prepared, removed_worktrees[]}`

## events

- `events.subscribe` — `{thread_ids[], after_seq?, server_instance_id?}` → `{server_instance_id, current_seq, replay_complete, events[], cursor:{server_instance_id,seq}}`

## project

- `project.list` — `{}` → `{projects[]}`
- `project.add` — `{path}` → `{project, projects[]}`
- `project.remove` — `{project_id}` → `{removed}`
- `project.refresh` — `{project_id}` → `{project, projects[]}`

## thread

- `thread.create` — `{project_id|workspace, title?, kind:"standard"|"worktree", branch?, model_id?}` → `{thread, threads[]}`
- `thread.list` — `{project_id?, filter:"active"|"archived"|"all"}` → `{threads[]}`
- `thread.get` — `{thread_id, refresh?}` → `{thread, stats, history[], turns[], events[]}`
- `thread.delete` — `{thread_id}` → `{deleted}`
- `thread.rename` — `{thread_id, title}` → `{thread}`
- `thread.configure` — `{thread_id, model_id?, thinking_level?}` → `{thread}`
- `thread.reorder` — `{project_id, thread_ids[]}` → `{threads[]}`
- `thread.set_pinned` — `{thread_id, pinned}` → `{thread}`
- `thread.set_archived` — `{thread_id, archived}` → `{thread}`
- `thread.set_read` — `{thread_id, read}` → `{updated}`
- `thread.context` — `{thread_id}` → `{used, maximum, ratio}`
- `thread.tree` — `{thread_id}` → `{nodes[], current}`
- `thread.navigate` — `{thread_id, node_id}` → `{current, stats}`
- `thread.compact` — `{thread_id, mode?}` → `{compacted, stats}`
- `thread.next_turn` — `{thread_id}` → `{turn_id}`

## turn

- `turn.start` — `{thread_id, text, attachment_ids[]}` → `{operation_id, turn_id}`
- `turn.cancel` — `{operation_id}` → `{cancelling}`
- `turn.steer` — `{operation_id, text}` → `{accepted}`
- `turn.follow_up` — `{thread_id, text}` → `{accepted}`

## fs

- `fs.roots` — `{}` → `{roots:[{path,label,writable}]}`
- `fs.list` — `{thread_id|root, path?}` → `{entries:[{name,kind,size,mtime}]}`
- `fs.read` — `{thread_id, path, max_bytes?}` → `{content, truncated, encoding}`

## git

- `git.status` — `{thread_id}` → `{ok,branch,entries[],stderr}`
- `git.github_status` — `{thread_id}` → `{github_origin,authenticated,remote,stderr}`
- `git.diff` — `{thread_id, scope:"changes"|"staged", path?}` → `{ok,scope,path,patch,binary,truncated,stderr}`
- `git.changes` — `{thread_id}` → `{branch,staged,changes}`
- `git.stage` — `{thread_id, paths[]?}` → `{ok,stdout,stderr}`
- `git.unstage` — `{thread_id, paths[]?}` → `{ok,stdout,stderr}`
- `git.branches` — `{thread_id}` → `{ok,branches[],current,stderr}`
- `git.branch_create` — `{thread_id, branch}` → `{ok,branch,stdout,stderr}`
- `git.commit` — `{thread_id, message}` → `{ok,commit,stdout,stderr}`
- `git.push` — `{thread_id}` → `{ok,pushed,remote,stdout,stderr}`
- `git.pr_create` — `{thread_id, title, body?}` → `{ok,url,stdout,stderr}`

## terminal

- `terminal.open` — `{thread_id}` → `{opened, terminal_id, columns, rows}`
- `terminal.input` — `{thread_id, data}` → `{accepted}`
- `terminal.resize` — `{thread_id, columns, rows}` → `{accepted}`
- `terminal.close` — `{thread_id}` → `{closed}`

## attachment

- `attachment.upload` — `{name, mime_type?, data_base64}` → `{attachment_id, name, mime_type, size}`
- `attachment.delete` — `{attachment_id}` → `{deleted}`
- `attachment.preview` — `{attachment_id}` → `{kind, preview}`
- `attachment.download` — `{attachment_id}` → `{data_base64, mime_type, name}`

## artifact

- `artifact.resolve` — `{thread_id, paths[]}` → `{artifacts:[{path,kind,exists}]}`
- `artifact.download` — `{thread_id, path}` → `{data_base64, mime_type}`

## catalog

- `catalog.get` — `{thread_id?, workspace?}` → `{models[], skills[], tools[], providers[], prompt_templates[], default_model_id}`
- `provider.list` — `{workspace?}` → `{providers[]}`
- `provider.refresh` — `{provider_id, revision?}` → `{provider}`
- `provider.add` — `{provider_id, config, revision?}` → `{provider}`
- `provider.remove` — `{provider_id, revision?}` → `{removed}`
- `settings.get` — `{workspace?}` → `{settings, revision}`
- `settings.update` — `{patch, revision, workspace?, secret_actions?}` → `{settings, revision}`
- `tools.search_test` — `{query, provider?}` → `{results[], engine, error?}`

## plugins

- `plugins.list` — `{}` → `{plugins:[{id,version,enabled,state,ports[],error?}]}`
- `plugins.set_enabled` — `{plugin_id, enabled}` → `{plugin}`
- `plugins.configure` — `{plugin_id, settings, revision}` → `{plugin, revision}`

## Plugin-contributed methods

- `plugin.cloud.account_status` — contributed namespace (cloud.account.status)
- `plugin.cloud.account_login` — contributed namespace (cloud.account.login)
- `plugin.cloud.account_logout` — contributed namespace (cloud.account.logout)
- `plugin.memory.remember` — contributed namespace (新增)
- `plugin.memory.recall` — contributed namespace (新增)
- `plugin.memory.forget` — contributed namespace (新增)
- `plugin.knowledge.index` — contributed namespace (新增)
- `plugin.knowledge.search` — contributed namespace (新增)
- `plugin.knowledge.stat` — contributed namespace (新增)

## Events

- `capabilities.updated`
- `content.completed`
- `content.delta`
- `content.started`
- `content.updated`
- `log.entry`
- `operation.cancelled`
- `operation.cancelling`
- `operation.completed`
- `operation.failed`
- `operation.queued`
- `operation.started`
- `plugin.<id>.*`
- `plugin.cloud.account_updated`
- `provider.updated`
- `queue.updated`
- `retry.completed`
- `retry.started`
- `settings.updated`
- `system.shutdown`
- `terminal.exit`
- `terminal.opened`
- `terminal.output`
- `thread.compacted`
- `thread.navigated`
- `thread.renamed`
- `tool.completed`
- `tool.started`
- `tool.updated`
- `turn.created`
- `usage.updated`

## Errors

- `protocol_incompatible` — retained
- `invalid_argument` — retained
- `thread_not_found` — retained
- `operation_not_found` — retained
- `busy` — retained
- `invalid_state` — retained
- `invalid_attachment` — retained
- `attachment_processing_failed` — retained
- `revision_conflict` — retained
- `authentication_failed` — retained
- `internal_error` — retained
- `method_not_found` — retained
- `unauthorized` (-32011) — added
- `forbidden` (-32012) — added
- `capability_unavailable` (-32013) — added
- `payload_too_large` (-32014) — added
- `session_not_found` → `thread_not_found` — renamed
