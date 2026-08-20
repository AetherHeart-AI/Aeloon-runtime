/* eslint-disable */
/** Generated from aeloon_runtime/rpc/aeloon-rpc-v3.manifest.json. DO NOT EDIT. */

export interface Event_capabilities_updated_v3 {
  capabilities: ({ [k: string]: unknown })[];
}

export interface Event_content_completed_v3 {
  block_id: string;
  patch: { [k: string]: unknown };
}

export interface Event_content_delta_v3 {
  block_id: string;
  delta: string;
}

export interface Event_content_started_v3 {
  block: { [k: string]: unknown };
}

export interface Event_content_updated_v3 {
  block_id: string;
  patch: { [k: string]: unknown };
}

export interface Event_log_entry_v3 {
  action?: string;
  attachment_id?: string;
  canonical_path?: string;
  category?: string;
  core_commit?: string;
  core_version?: string;
  data?: { [k: string]: unknown };
  dpi?: number;
  level?: string;
  message?: string;
  metadata?: { [k: string]: unknown };
  pages?: (number)[];
}

export interface Event_operation_cancelled_v3 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_operation_cancelling_v3 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_operation_completed_v3 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_operation_failed_v3 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_operation_queued_v3 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_operation_started_v3 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_plugin_cloud_account_updated_v3 {
  authenticated: boolean;
  base_url: string;
  enabled: boolean;
  ok?: boolean;
  user: Record<string, never> | null;
  vault_kind: string;
}

export interface Event_plugin_id_v3 {
  [k: string]: unknown;
}

export interface Event_provider_updated_v3 {
  action: string;
  provider_id: string;
}

export interface Event_queue_updated_v3 {
  active_operation_id?: string | null;
  queued_operation_ids?: (string)[];
}

export interface Event_retry_completed_v3 {
  [k: string]: unknown;
}

export interface Event_retry_started_v3 {
  [k: string]: unknown;
}

export interface Event_settings_updated_v3 {
  revision: number;
}

export interface Event_system_shutdown_v3 {
  intentional: boolean;
  reason: string;
}

export interface Event_terminal_exit_v3 {
  exit_code?: number;
  signal?: number;
  status: string;
}

export interface Event_terminal_opened_v3 {
  columns: number;
  rows: number;
}

export interface Event_terminal_output_v3 {
  data: string;
}

export interface Event_thread_compacted_v3 {
  [k: string]: unknown;
}

export interface Event_thread_navigated_v3 {
  [k: string]: unknown;
}

export interface Event_thread_renamed_v3 {
  source?: string;
  title: string | null;
}

export interface Event_tool_completed_v3 {
  block_id: string;
  patch: { [k: string]: unknown };
}

export interface Event_tool_started_v3 {
  block: { [k: string]: unknown };
}

export interface Event_tool_updated_v3 {
  block_id: string;
  patch: { [k: string]: unknown };
}

export interface Event_turn_created_v3 {
  [k: string]: unknown;
}

export interface Event_usage_updated_v3 {
  stats?: { [k: string]: unknown };
  usage?: { [k: string]: unknown };
}

export interface GitChangeFile_v3 {
  additions: number;
  binary: boolean;
  deletions: number;
  path: string;
  renamed_from?: string;
  status: string;
}

export interface GitChangeGroup_v3 {
  additions: number;
  deletions: number;
  files: (GitChangeFile_v3)[];
}

export interface Params_artifact_download_v3 {
  path: string;
  thread_id: string;
}

export interface Params_artifact_resolve_v3 {
  paths: (string)[];
  thread_id: string;
}

export interface Params_attachment_delete_v3 {
  attachment_id: string;
}

export interface Params_attachment_download_v3 {
  attachment_id: string;
}

export interface Params_attachment_preview_v3 {
  attachment_id: string;
}

export interface Params_attachment_upload_v3 {
  data_base64: string;
  mime_type?: string;
  name: string;
}

export interface Params_catalog_get_v3 {
  thread_id?: string;
  workspace?: string;
}

export interface Params_events_subscribe_v3 {
  after_seq?: number;
  server_instance_id?: string;
  thread_ids: (string)[];
}

export type Params_fs_list_v3 = { [k: string]: unknown } | { [k: string]: unknown };
export interface Params_fs_read_v3 {
  max_bytes?: number;
  path: string;
  thread_id: string;
}

export interface Params_fs_roots_v3 {
}

export interface Params_git_branch_create_v3 {
  branch: string | null;
  thread_id: string;
}

export interface Params_git_branches_v3 {
  thread_id: string;
}

export interface Params_git_changes_v3 {
  thread_id: string;
}

export interface Params_git_commit_v3 {
  message: string;
  thread_id: string;
}

export interface Params_git_diff_v3 {
  path?: string;
  scope: "changes" | "staged";
  thread_id: string;
}

export interface Params_git_github_status_v3 {
  thread_id: string;
}

export interface Params_git_pr_create_v3 {
  body?: string;
  thread_id: string;
  title: string;
}

export interface Params_git_push_v3 {
  thread_id: string;
}

export interface Params_git_stage_v3 {
  paths?: (string)[];
  thread_id: string;
}

export interface Params_git_status_v3 {
  thread_id: string;
}

export interface Params_git_unstage_v3 {
  paths?: (string)[];
  thread_id: string;
}

export interface Params_plugin_cloud_account_login_v3 {
  password: string;
  username: string;
  workspace?: string;
}

export interface Params_plugin_cloud_account_logout_v3 {
  workspace?: string;
}

export interface Params_plugin_cloud_account_status_v3 {
  workspace?: string;
}

export interface Params_plugin_knowledge_index_v3 {
  [k: string]: unknown;
}

export interface Params_plugin_knowledge_search_v3 {
  [k: string]: unknown;
}

export interface Params_plugin_knowledge_stat_v3 {
  [k: string]: unknown;
}

export interface Params_plugin_memory_forget_v3 {
  [k: string]: unknown;
}

export interface Params_plugin_memory_recall_v3 {
  [k: string]: unknown;
}

export interface Params_plugin_memory_remember_v3 {
  [k: string]: unknown;
}

export interface Params_plugins_configure_v3 {
  plugin_id: string;
  revision: number;
  settings: { [k: string]: unknown };
}

export interface Params_plugins_list_v3 {
}

export interface Params_plugins_set_enabled_v3 {
  enabled: boolean;
  plugin_id: string;
}

export interface Params_project_add_v3 {
  path: string;
}

export interface Params_project_list_v3 {
}

export interface Params_project_refresh_v3 {
  project_id: string;
}

export interface Params_project_remove_v3 {
  project_id: string;
}

export interface Params_provider_add_v3 {
  config: { [k: string]: unknown };
  provider_id: string;
  revision?: number;
}

export interface Params_provider_list_v3 {
  workspace?: string;
}

export interface Params_provider_refresh_v3 {
  provider_id: string;
  revision?: number;
}

export interface Params_provider_remove_v3 {
  provider_id: string;
  revision?: number;
}

export interface Params_settings_get_v3 {
  workspace?: string;
}

export interface Params_settings_update_v3 {
  patch: { [k: string]: unknown };
  revision: number;
  secret_actions?: ({ [k: string]: unknown })[];
  workspace?: string;
}

export interface Params_system_capabilities_v3 {
}

export interface Params_system_handshake_v3 {
  auth?: { [k: string]: unknown };
  client: { [k: string]: unknown };
  protocol: { [k: string]: unknown };
}

export interface Params_system_health_v3 {
  workspace?: string;
}

export interface Params_system_shutdown_v3 {
}

export interface Params_system_snapshot_v3 {
}

export interface Params_system_uninstall_inspect_v3 {
}

export interface Params_system_uninstall_prepare_v3 {
}

export interface Params_terminal_close_v3 {
  thread_id: string;
}

export interface Params_terminal_input_v3 {
  data: string;
  thread_id: string;
}

export interface Params_terminal_open_v3 {
  thread_id: string;
}

export interface Params_terminal_resize_v3 {
  columns: number;
  rows: number;
  thread_id: string;
}

export interface Params_thread_compact_v3 {
  mode?: string;
  thread_id: string;
}

export interface Params_thread_configure_v3 {
  model_id?: string;
  thinking_level?: string;
  thread_id: string;
}

export interface Params_thread_context_v3 {
  thread_id: string;
}

export type Params_thread_create_v3 = { [k: string]: unknown } | { [k: string]: unknown };
export interface Params_thread_delete_v3 {
  thread_id: string;
}

export interface Params_thread_get_v3 {
  refresh?: boolean;
  thread_id: string;
}

export interface Params_thread_list_v3 {
  filter: "active" | "archived" | "all";
  project_id?: string;
}

export interface Params_thread_navigate_v3 {
  node_id: string;
  thread_id: string;
}

export interface Params_thread_next_turn_v3 {
  thread_id: string;
}

export interface Params_thread_rename_v3 {
  thread_id: string;
  title: string;
}

export interface Params_thread_reorder_v3 {
  project_id: string;
  thread_ids: (string)[];
}

export interface Params_thread_set_archived_v3 {
  archived: boolean;
  thread_id: string;
}

export interface Params_thread_set_pinned_v3 {
  pinned: boolean;
  thread_id: string;
}

export interface Params_thread_set_read_v3 {
  read: boolean;
  thread_id: string;
}

export interface Params_thread_tree_v3 {
  thread_id: string;
}

export interface Params_tools_search_test_v3 {
  provider?: { [k: string]: unknown };
  query: string;
}

export interface Params_turn_cancel_v3 {
  operation_id: string;
}

export interface Params_turn_follow_up_v3 {
  text: string;
  thread_id: string;
}

export interface Params_turn_start_v3 {
  attachment_ids: (string)[];
  text: string;
  thread_id: string;
}

export interface Params_turn_steer_v3 {
  operation_id: string;
  text: string;
}

export interface Result_artifact_download_v3 {
  data_base64: string;
  mime_type: string;
}

export interface Result_artifact_resolve_v3 {
  artifacts: ({ [k: string]: unknown })[];
}

export interface Result_attachment_delete_v3 {
  deleted: boolean;
}

export interface Result_attachment_download_v3 {
  data_base64: string;
  mime_type: string;
  name: string;
}

export interface Result_attachment_preview_v3 {
  kind: string;
  preview: string;
}

export interface Result_attachment_upload_v3 {
  attachment_id: string;
  mime_type: string;
  name: string;
  size: number;
}

export interface Result_catalog_get_v3 {
  default_model_id: string | null;
  models: ({ [k: string]: unknown })[];
  prompt_templates: ({ [k: string]: unknown })[];
  providers: ({ [k: string]: unknown })[];
  skills: ({ [k: string]: unknown })[];
  tools: ({ [k: string]: unknown })[];
}

export interface Result_events_subscribe_v3 {
  current_seq: number;
  cursor: { [k: string]: unknown };
  events: ({ [k: string]: unknown })[];
  replay_complete: boolean;
  server_instance_id: string;
}

export interface Result_fs_list_v3 {
  entries: ({ [k: string]: unknown })[];
}

export interface Result_fs_read_v3 {
  content: string;
  encoding: string;
  truncated: boolean;
}

export interface Result_fs_roots_v3 {
  roots: ({ [k: string]: unknown })[];
}

export interface Result_git_branch_create_v3 {
  branch: string | null;
  ok: boolean;
  stderr: string;
  stdout: string;
}

export interface Result_git_branches_v3 {
  branches: (string)[];
  current: string | null;
  ok: boolean;
  stderr: string;
}

export interface Result_git_changes_v3 {
  branch: string | null;
  changes: GitChangeGroup_v3;
  staged: GitChangeGroup_v3;
}

export interface Result_git_commit_v3 {
  commit: string;
  ok: boolean;
  stderr: string;
  stdout: string;
}

export interface Result_git_diff_v3 {
  binary: boolean;
  ok: boolean;
  patch: { [k: string]: unknown };
  path: string;
  scope: string;
  stderr: string;
  truncated: boolean;
}

export interface Result_git_github_status_v3 {
  authenticated: boolean;
  github_origin: string;
  remote: string;
  stderr: string;
}

export interface Result_git_pr_create_v3 {
  ok: boolean;
  stderr: string;
  stdout: string;
  url: string;
}

export interface Result_git_push_v3 {
  ok: boolean;
  pushed: boolean;
  remote: string;
  stderr: string;
  stdout: string;
}

export interface Result_git_stage_v3 {
  ok: boolean;
  stderr: string;
  stdout: string;
}

export interface Result_git_status_v3 {
  branch: string | null;
  entries: ({ [k: string]: unknown })[];
  ok: boolean;
  stderr: string;
}

export interface Result_git_unstage_v3 {
  ok: boolean;
  stderr: string;
  stdout: string;
}

export interface Result_plugin_cloud_account_status_v3 {
  authenticated: boolean;
  base_url: string;
  enabled: boolean;
  ok?: boolean;
  user: Record<string, never> | null;
  vault_kind: string;
}

export interface Result_plugin_knowledge_index_v3 {
  [k: string]: unknown;
}

export interface Result_plugin_knowledge_search_v3 {
  [k: string]: unknown;
}

export interface Result_plugin_knowledge_stat_v3 {
  [k: string]: unknown;
}

export interface Result_plugin_memory_forget_v3 {
  [k: string]: unknown;
}

export interface Result_plugin_memory_recall_v3 {
  [k: string]: unknown;
}

export interface Result_plugin_memory_remember_v3 {
  [k: string]: unknown;
}

export interface Result_plugins_configure_v3 {
  plugin: string;
  revision: number;
}

export interface Result_plugins_list_v3 {
  plugins: ({ [k: string]: unknown })[];
}

export interface Result_plugins_set_enabled_v3 {
  plugin: string;
}

export interface Result_project_add_v3 {
  project: { [k: string]: unknown };
  projects: ({ [k: string]: unknown })[];
}

export interface Result_project_list_v3 {
  projects: ({ [k: string]: unknown })[];
}

export interface Result_project_refresh_v3 {
  project: { [k: string]: unknown };
  projects: ({ [k: string]: unknown })[];
}

export interface Result_project_remove_v3 {
  removed: boolean;
}

export interface Result_provider_add_v3 {
  provider: { [k: string]: unknown };
}

export interface Result_provider_list_v3 {
  providers: ({ [k: string]: unknown })[];
}

export interface Result_provider_refresh_v3 {
  provider: { [k: string]: unknown };
}

export interface Result_provider_remove_v3 {
  removed: boolean;
}

export interface Result_settings_get_v3 {
  revision: number;
  settings: { [k: string]: unknown };
}

export interface Result_settings_update_v3 {
  revision: number;
  settings: { [k: string]: unknown };
}

export interface Result_system_capabilities_v3 {
  capabilities: ({ [k: string]: unknown })[];
}

export interface Result_system_handshake_v3 {
  limits: { [k: string]: unknown };
  protocol: string;
  runtime: { [k: string]: unknown };
  workspace_roots: (string)[];
}

export interface Result_system_health_v3 {
  active_operations: number;
  ok: boolean;
  uptime_s: number;
}

export interface Result_system_shutdown_v3 {
  accepted: boolean;
}

export interface Result_system_snapshot_v3 {
  default_workspace: string;
  projects: ({ [k: string]: unknown })[];
  runtime: { [k: string]: unknown };
  threads: ({ [k: string]: unknown })[];
}

export interface Result_system_uninstall_inspect_v3 {
  active_operations: number;
  data_paths: (string)[];
  estimated_bytes: number;
  worktrees: ({ [k: string]: unknown })[];
}

export interface Result_system_uninstall_prepare_v3 {
  prepared: boolean;
  removed_worktrees: ({ [k: string]: unknown })[];
}

export interface Result_terminal_close_v3 {
  closed: boolean;
}

export interface Result_terminal_input_v3 {
  accepted: boolean;
}

export interface Result_terminal_open_v3 {
  columns: number;
  opened: boolean;
  rows: number;
  terminal_id: string;
}

export interface Result_terminal_resize_v3 {
  accepted: boolean;
}

export interface Result_thread_compact_v3 {
  compacted: boolean;
  stats: { [k: string]: unknown };
}

export interface Result_thread_configure_v3 {
  thread: { [k: string]: unknown };
}

export interface Result_thread_context_v3 {
  maximum: number | null;
  ratio: number | null;
  used: number;
}

export interface Result_thread_create_v3 {
  thread: { [k: string]: unknown };
  threads: ({ [k: string]: unknown })[];
}

export interface Result_thread_delete_v3 {
  deleted: boolean;
}

export interface Result_thread_get_v3 {
  events: ({ [k: string]: unknown })[];
  history: ({ [k: string]: unknown })[];
  stats: { [k: string]: unknown };
  thread: { [k: string]: unknown };
  turns: ({ [k: string]: unknown })[];
}

export interface Result_thread_list_v3 {
  threads: ({ [k: string]: unknown })[];
}

export interface Result_thread_navigate_v3 {
  current: string | null;
  stats: { [k: string]: unknown };
}

export interface Result_thread_next_turn_v3 {
  turn_id: string;
}

export interface Result_thread_rename_v3 {
  thread: { [k: string]: unknown };
}

export interface Result_thread_reorder_v3 {
  threads: ({ [k: string]: unknown })[];
}

export interface Result_thread_set_archived_v3 {
  thread: { [k: string]: unknown };
}

export interface Result_thread_set_pinned_v3 {
  thread: { [k: string]: unknown };
}

export interface Result_thread_set_read_v3 {
  updated: boolean;
}

export interface Result_thread_tree_v3 {
  current: string | null;
  nodes: ({ [k: string]: unknown })[];
}

export interface Result_tools_search_test_v3 {
  engine: string;
  error?: string;
  results: ({ [k: string]: unknown })[];
}

export interface Result_turn_cancel_v3 {
  cancelling: boolean;
}

export interface Result_turn_follow_up_v3 {
  accepted: boolean;
}

export interface Result_turn_start_v3 {
  operation_id: string;
  turn_id: string;
}

export interface Result_turn_steer_v3 {
  accepted: boolean;
}

export interface AeloonRuntimeRpcDefinitions {}

export interface RuntimeRpcMethodMap {
  "artifact.download": { params: Params_artifact_download_v3; result: Result_artifact_download_v3 };
  "artifact.resolve": { params: Params_artifact_resolve_v3; result: Result_artifact_resolve_v3 };
  "attachment.delete": { params: Params_attachment_delete_v3; result: Result_attachment_delete_v3 };
  "attachment.download": { params: Params_attachment_download_v3; result: Result_attachment_download_v3 };
  "attachment.preview": { params: Params_attachment_preview_v3; result: Result_attachment_preview_v3 };
  "attachment.upload": { params: Params_attachment_upload_v3; result: Result_attachment_upload_v3 };
  "catalog.get": { params: Params_catalog_get_v3; result: Result_catalog_get_v3 };
  "events.subscribe": { params: Params_events_subscribe_v3; result: Result_events_subscribe_v3 };
  "fs.list": { params: Params_fs_list_v3; result: Result_fs_list_v3 };
  "fs.read": { params: Params_fs_read_v3; result: Result_fs_read_v3 };
  "fs.roots": { params: Params_fs_roots_v3; result: Result_fs_roots_v3 };
  "git.branch_create": { params: Params_git_branch_create_v3; result: Result_git_branch_create_v3 };
  "git.branches": { params: Params_git_branches_v3; result: Result_git_branches_v3 };
  "git.changes": { params: Params_git_changes_v3; result: Result_git_changes_v3 };
  "git.commit": { params: Params_git_commit_v3; result: Result_git_commit_v3 };
  "git.diff": { params: Params_git_diff_v3; result: Result_git_diff_v3 };
  "git.github_status": { params: Params_git_github_status_v3; result: Result_git_github_status_v3 };
  "git.pr_create": { params: Params_git_pr_create_v3; result: Result_git_pr_create_v3 };
  "git.push": { params: Params_git_push_v3; result: Result_git_push_v3 };
  "git.stage": { params: Params_git_stage_v3; result: Result_git_stage_v3 };
  "git.status": { params: Params_git_status_v3; result: Result_git_status_v3 };
  "git.unstage": { params: Params_git_unstage_v3; result: Result_git_unstage_v3 };
  "plugins.configure": { params: Params_plugins_configure_v3; result: Result_plugins_configure_v3 };
  "plugins.list": { params: Params_plugins_list_v3; result: Result_plugins_list_v3 };
  "plugins.set_enabled": { params: Params_plugins_set_enabled_v3; result: Result_plugins_set_enabled_v3 };
  "project.add": { params: Params_project_add_v3; result: Result_project_add_v3 };
  "project.list": { params: Params_project_list_v3; result: Result_project_list_v3 };
  "project.refresh": { params: Params_project_refresh_v3; result: Result_project_refresh_v3 };
  "project.remove": { params: Params_project_remove_v3; result: Result_project_remove_v3 };
  "provider.add": { params: Params_provider_add_v3; result: Result_provider_add_v3 };
  "provider.list": { params: Params_provider_list_v3; result: Result_provider_list_v3 };
  "provider.refresh": { params: Params_provider_refresh_v3; result: Result_provider_refresh_v3 };
  "provider.remove": { params: Params_provider_remove_v3; result: Result_provider_remove_v3 };
  "settings.get": { params: Params_settings_get_v3; result: Result_settings_get_v3 };
  "settings.update": { params: Params_settings_update_v3; result: Result_settings_update_v3 };
  "system.capabilities": { params: Params_system_capabilities_v3; result: Result_system_capabilities_v3 };
  "system.handshake": { params: Params_system_handshake_v3; result: Result_system_handshake_v3 };
  "system.health": { params: Params_system_health_v3; result: Result_system_health_v3 };
  "system.shutdown": { params: Params_system_shutdown_v3; result: Result_system_shutdown_v3 };
  "system.snapshot": { params: Params_system_snapshot_v3; result: Result_system_snapshot_v3 };
  "system.uninstall_inspect": { params: Params_system_uninstall_inspect_v3; result: Result_system_uninstall_inspect_v3 };
  "system.uninstall_prepare": { params: Params_system_uninstall_prepare_v3; result: Result_system_uninstall_prepare_v3 };
  "terminal.close": { params: Params_terminal_close_v3; result: Result_terminal_close_v3 };
  "terminal.input": { params: Params_terminal_input_v3; result: Result_terminal_input_v3 };
  "terminal.open": { params: Params_terminal_open_v3; result: Result_terminal_open_v3 };
  "terminal.resize": { params: Params_terminal_resize_v3; result: Result_terminal_resize_v3 };
  "thread.compact": { params: Params_thread_compact_v3; result: Result_thread_compact_v3 };
  "thread.configure": { params: Params_thread_configure_v3; result: Result_thread_configure_v3 };
  "thread.context": { params: Params_thread_context_v3; result: Result_thread_context_v3 };
  "thread.create": { params: Params_thread_create_v3; result: Result_thread_create_v3 };
  "thread.delete": { params: Params_thread_delete_v3; result: Result_thread_delete_v3 };
  "thread.get": { params: Params_thread_get_v3; result: Result_thread_get_v3 };
  "thread.list": { params: Params_thread_list_v3; result: Result_thread_list_v3 };
  "thread.navigate": { params: Params_thread_navigate_v3; result: Result_thread_navigate_v3 };
  "thread.next_turn": { params: Params_thread_next_turn_v3; result: Result_thread_next_turn_v3 };
  "thread.rename": { params: Params_thread_rename_v3; result: Result_thread_rename_v3 };
  "thread.reorder": { params: Params_thread_reorder_v3; result: Result_thread_reorder_v3 };
  "thread.set_archived": { params: Params_thread_set_archived_v3; result: Result_thread_set_archived_v3 };
  "thread.set_pinned": { params: Params_thread_set_pinned_v3; result: Result_thread_set_pinned_v3 };
  "thread.set_read": { params: Params_thread_set_read_v3; result: Result_thread_set_read_v3 };
  "thread.tree": { params: Params_thread_tree_v3; result: Result_thread_tree_v3 };
  "tools.search_test": { params: Params_tools_search_test_v3; result: Result_tools_search_test_v3 };
  "turn.cancel": { params: Params_turn_cancel_v3; result: Result_turn_cancel_v3 };
  "turn.follow_up": { params: Params_turn_follow_up_v3; result: Result_turn_follow_up_v3 };
  "turn.start": { params: Params_turn_start_v3; result: Result_turn_start_v3 };
  "turn.steer": { params: Params_turn_steer_v3; result: Result_turn_steer_v3 };
}
export type RuntimeMethod = keyof RuntimeRpcMethodMap;
export type RuntimeRpcParams<M extends RuntimeMethod> = RuntimeRpcMethodMap[M]["params"];
export type RuntimeRpcResult<M extends RuntimeMethod> = RuntimeRpcMethodMap[M]["result"];

export interface RuntimePluginMethodMap {
  "plugin.cloud.account_login": { params: Params_plugin_cloud_account_login_v3; result: Result_plugin_cloud_account_status_v3 };
  "plugin.cloud.account_logout": { params: Params_plugin_cloud_account_logout_v3; result: Result_plugin_cloud_account_status_v3 };
  "plugin.cloud.account_status": { params: Params_plugin_cloud_account_status_v3; result: Result_plugin_cloud_account_status_v3 };
  "plugin.knowledge.index": { params: Params_plugin_knowledge_index_v3; result: Result_plugin_knowledge_index_v3 };
  "plugin.knowledge.search": { params: Params_plugin_knowledge_search_v3; result: Result_plugin_knowledge_search_v3 };
  "plugin.knowledge.stat": { params: Params_plugin_knowledge_stat_v3; result: Result_plugin_knowledge_stat_v3 };
  "plugin.memory.forget": { params: Params_plugin_memory_forget_v3; result: Result_plugin_memory_forget_v3 };
  "plugin.memory.recall": { params: Params_plugin_memory_recall_v3; result: Result_plugin_memory_recall_v3 };
  "plugin.memory.remember": { params: Params_plugin_memory_remember_v3; result: Result_plugin_memory_remember_v3 };
}
export type RuntimePluginMethod = keyof RuntimePluginMethodMap;
export type RuntimePluginRpcParams<M extends RuntimePluginMethod> = RuntimePluginMethodMap[M]["params"];
export type RuntimePluginRpcResult<M extends RuntimePluginMethod> = RuntimePluginMethodMap[M]["result"];

export interface RuntimeEventBase {
  seq: number;
  time?: string;
  thread_id?: string | null;
  operation_id?: string | null;
  terminal_id?: string | null;
  workspace?: string | null;
}
export type RuntimeEvent =
  | (RuntimeEventBase & { name: "capabilities.updated"; payload: Event_capabilities_updated_v3 })
  | (RuntimeEventBase & { name: "content.completed"; payload: Event_content_completed_v3 })
  | (RuntimeEventBase & { name: "content.delta"; payload: Event_content_delta_v3 })
  | (RuntimeEventBase & { name: "content.started"; payload: Event_content_started_v3 })
  | (RuntimeEventBase & { name: "content.updated"; payload: Event_content_updated_v3 })
  | (RuntimeEventBase & { name: "log.entry"; payload: Event_log_entry_v3 })
  | (RuntimeEventBase & { name: "operation.cancelled"; payload: Event_operation_cancelled_v3 })
  | (RuntimeEventBase & { name: "operation.cancelling"; payload: Event_operation_cancelling_v3 })
  | (RuntimeEventBase & { name: "operation.completed"; payload: Event_operation_completed_v3 })
  | (RuntimeEventBase & { name: "operation.failed"; payload: Event_operation_failed_v3 })
  | (RuntimeEventBase & { name: "operation.queued"; payload: Event_operation_queued_v3 })
  | (RuntimeEventBase & { name: "operation.started"; payload: Event_operation_started_v3 })
  | (RuntimeEventBase & { name: "plugin.<id>.*"; payload: Event_plugin_id_v3 })
  | (RuntimeEventBase & { name: "plugin.cloud.account_updated"; payload: Event_plugin_cloud_account_updated_v3 })
  | (RuntimeEventBase & { name: "provider.updated"; payload: Event_provider_updated_v3 })
  | (RuntimeEventBase & { name: "queue.updated"; payload: Event_queue_updated_v3 })
  | (RuntimeEventBase & { name: "retry.completed"; payload: Event_retry_completed_v3 })
  | (RuntimeEventBase & { name: "retry.started"; payload: Event_retry_started_v3 })
  | (RuntimeEventBase & { name: "settings.updated"; payload: Event_settings_updated_v3 })
  | (RuntimeEventBase & { name: "system.shutdown"; payload: Event_system_shutdown_v3 })
  | (RuntimeEventBase & { name: "terminal.exit"; payload: Event_terminal_exit_v3 })
  | (RuntimeEventBase & { name: "terminal.opened"; payload: Event_terminal_opened_v3 })
  | (RuntimeEventBase & { name: "terminal.output"; payload: Event_terminal_output_v3 })
  | (RuntimeEventBase & { name: "thread.compacted"; payload: Event_thread_compacted_v3 })
  | (RuntimeEventBase & { name: "thread.navigated"; payload: Event_thread_navigated_v3 })
  | (RuntimeEventBase & { name: "thread.renamed"; payload: Event_thread_renamed_v3 })
  | (RuntimeEventBase & { name: "tool.completed"; payload: Event_tool_completed_v3 })
  | (RuntimeEventBase & { name: "tool.started"; payload: Event_tool_started_v3 })
  | (RuntimeEventBase & { name: "tool.updated"; payload: Event_tool_updated_v3 })
  | (RuntimeEventBase & { name: "turn.created"; payload: Event_turn_created_v3 })
  | (RuntimeEventBase & { name: "usage.updated"; payload: Event_usage_updated_v3 })
;
export type RuntimeEventName = RuntimeEvent["name"];

export const RUNTIME_RPC_PROTOCOL = "aeloon-rpc" as const;
export const RUNTIME_RPC_VERSION = "3.0.0" as const;
export const RUNTIME_RPC_MAX_FRAME_BYTES = 41943040 as const;
