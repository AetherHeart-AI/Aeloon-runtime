/* eslint-disable */
/** Generated from aeloon_runtime/rpc/aeloon-rpc-v4.manifest.json. DO NOT EDIT. */

export interface Device_v4 {
  connected?: boolean;
  id: string;
  last_seen_at?: string | null;
  name: string;
  paired_at: string;
  platform: string;
}

export interface Event_capabilities_updated_v4 {
  capabilities: ({ [k: string]: unknown })[];
}

export interface Event_content_completed_v4 {
  block_id: string;
  patch: { [k: string]: unknown };
}

export interface Event_content_delta_v4 {
  block_id: string;
  delta: string;
}

export interface Event_content_started_v4 {
  block: { [k: string]: unknown };
}

export interface Event_content_updated_v4 {
  block_id: string;
  patch: { [k: string]: unknown };
}

export interface Event_log_entry_v4 {
  action?: string;
  attachment_id?: string;
  canonical_path?: string;
  category?: string;
  data?: { [k: string]: unknown };
  dpi?: number;
  level?: string;
  message?: string;
  metadata?: { [k: string]: unknown };
  pages?: (number)[];
  runtime_commit?: string;
  runtime_version?: string;
}

export interface Event_operation_cancelled_v4 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_operation_cancelling_v4 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_operation_completed_v4 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_operation_failed_v4 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_operation_queued_v4 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_operation_started_v4 {
  attachment_ids?: (string)[];
  code?: string;
  duration_ms?: number;
  error?: string;
  kind: string;
  queue_position?: number;
  skill_id?: string;
}

export interface Event_plugin_cloud_account_updated_v4 {
  authenticated: boolean;
  base_url: string;
  enabled: boolean;
  ok?: boolean;
  user: Record<string, never> | null;
  vault_kind: string;
}

export interface Event_plugin_id_v4 {
  [k: string]: unknown;
}

export interface Event_provider_updated_v4 {
  action: string;
  provider_id: string;
}

export interface Event_queue_updated_v4 {
  active_operation_id?: string | null;
  queued_operation_ids?: (string)[];
}

export interface Event_retry_completed_v4 {
  [k: string]: unknown;
}

export interface Event_retry_started_v4 {
  [k: string]: unknown;
}

export interface Event_settings_updated_v4 {
  revision: number;
}

export interface Event_system_shutdown_v4 {
  intentional: boolean;
  reason: string;
}

export interface Event_terminal_exit_v4 {
  exit_code?: number;
  signal?: number;
  status: string;
}

export interface Event_terminal_opened_v4 {
  columns: number;
  rows: number;
}

export interface Event_terminal_output_v4 {
  data: string;
}

export interface Event_thread_compacted_v4 {
  [k: string]: unknown;
}

export interface Event_thread_navigated_v4 {
  [k: string]: unknown;
}

export interface Event_thread_renamed_v4 {
  source?: string;
  title: string | null;
}

export interface Event_tool_completed_v4 {
  block_id: string;
  patch: { [k: string]: unknown };
}

export interface Event_tool_started_v4 {
  block: { [k: string]: unknown };
}

export interface Event_tool_updated_v4 {
  block_id: string;
  patch: { [k: string]: unknown };
}

export interface Event_turn_created_v4 {
  [k: string]: unknown;
}

export interface Event_usage_updated_v4 {
  stats?: { [k: string]: unknown };
  usage?: { [k: string]: unknown };
}

export interface GitChangeFile_v4 {
  additions: number;
  binary: boolean;
  deletions: number;
  path: string;
  renamed_from?: string;
  status: string;
}

export interface GitChangeGroup_v4 {
  additions: number;
  deletions: number;
  files: (GitChangeFile_v4)[];
}

export interface Params_artifact_download_v4 {
  path: string;
  thread_id: string;
}

export interface Params_artifact_resolve_v4 {
  paths: (string)[];
  thread_id: string;
}

export interface Params_attachment_delete_v4 {
  attachment_id: string;
}

export interface Params_attachment_download_v4 {
  attachment_id: string;
}

export interface Params_attachment_preview_v4 {
  attachment_id: string;
}

export interface Params_attachment_upload_v4 {
  data_base64: string;
  mime_type?: string;
  name: string;
}

export interface Params_catalog_get_v4 {
  thread_id?: string;
  workspace?: string;
}

export interface Params_devices_claim_v4 {
  client: { [k: string]: unknown };
  code: string;
}

export interface Params_devices_enroll_v4 {
}

export interface Params_devices_list_v4 {
}

export interface Params_devices_revoke_v4 {
  device_id: string;
}

export interface Params_diagnostics_logs_v4 {
  limit?: number;
}

export interface Params_events_subscribe_v4 {
  after_seq?: number;
  server_instance_id?: string;
  thread_ids: (string)[];
}

export type Params_fs_list_v4 = { [k: string]: unknown } | { [k: string]: unknown };
export interface Params_fs_read_v4 {
  max_bytes?: number;
  path: string;
  thread_id: string;
}

export interface Params_fs_roots_v4 {
}

export interface Params_git_branch_create_v4 {
  branch: string | null;
  thread_id: string;
}

export interface Params_git_branches_v4 {
  thread_id: string;
}

export interface Params_git_changes_v4 {
  thread_id: string;
}

export interface Params_git_commit_v4 {
  message: string;
  thread_id: string;
}

export interface Params_git_diff_v4 {
  path?: string;
  scope: "changes" | "staged";
  thread_id: string;
}

export interface Params_git_github_status_v4 {
  thread_id: string;
}

export interface Params_git_pr_create_v4 {
  body?: string;
  thread_id: string;
  title: string;
}

export interface Params_git_push_v4 {
  thread_id: string;
}

export interface Params_git_stage_v4 {
  paths?: (string)[];
  thread_id: string;
}

export interface Params_git_status_v4 {
  thread_id: string;
}

export interface Params_git_unstage_v4 {
  paths?: (string)[];
  thread_id: string;
}

export interface Params_plugin_cloud_account_login_v4 {
  password: string;
  username: string;
  workspace?: string;
}

export interface Params_plugin_cloud_account_logout_v4 {
  workspace?: string;
}

export interface Params_plugin_cloud_account_status_v4 {
  workspace?: string;
}

export interface Params_plugin_knowledge_index_v4 {
  [k: string]: unknown;
}

export interface Params_plugin_knowledge_search_v4 {
  [k: string]: unknown;
}

export interface Params_plugin_knowledge_stat_v4 {
  [k: string]: unknown;
}

export interface Params_plugin_memory_forget_v4 {
  [k: string]: unknown;
}

export interface Params_plugin_memory_recall_v4 {
  [k: string]: unknown;
}

export interface Params_plugin_memory_remember_v4 {
  [k: string]: unknown;
}

export interface Params_plugins_configure_v4 {
  plugin_id: string;
  revision: number;
  settings: { [k: string]: unknown };
}

export interface Params_plugins_list_v4 {
}

export interface Params_plugins_set_enabled_v4 {
  enabled: boolean;
  plugin_id: string;
}

export interface Params_project_add_v4 {
  relative_path: string;
  root_id: string;
}

export interface Params_project_list_v4 {
}

export interface Params_project_refresh_v4 {
  project_id: string;
}

export interface Params_project_remove_v4 {
  project_id: string;
}

export interface Params_provider_add_v4 {
  config: { [k: string]: unknown };
  provider_id: string;
  revision?: number;
}

export interface Params_provider_list_v4 {
  workspace?: string;
}

export interface Params_provider_refresh_v4 {
  provider_id: string;
  revision?: number;
}

export interface Params_provider_remove_v4 {
  provider_id: string;
  revision?: number;
}

export interface Params_settings_get_v4 {
  workspace?: string;
}

export interface Params_settings_update_v4 {
  patch: { [k: string]: unknown };
  revision: number;
  secret_actions?: ({ [k: string]: unknown })[];
  workspace?: string;
}

export interface Params_system_capabilities_v4 {
}

export interface Params_system_handshake_v4 {
  auth?: { [k: string]: unknown };
  client: { [k: string]: unknown };
  protocol: { [k: string]: unknown };
}

export interface Params_system_health_v4 {
  workspace?: string;
}

export interface Params_system_shutdown_v4 {
}

export interface Params_system_snapshot_v4 {
}

export interface Params_system_uninstall_inspect_v4 {
}

export interface Params_system_uninstall_prepare_v4 {
}

export interface Params_terminal_close_v4 {
  thread_id: string;
}

export interface Params_terminal_input_v4 {
  data: string;
  thread_id: string;
}

export interface Params_terminal_open_v4 {
  thread_id: string;
}

export interface Params_terminal_resize_v4 {
  columns: number;
  rows: number;
  thread_id: string;
}

export interface Params_thread_compact_v4 {
  mode?: string;
  thread_id: string;
}

export interface Params_thread_configure_v4 {
  model_id?: string;
  thinking_level?: string;
  thread_id: string;
}

export interface Params_thread_context_v4 {
  thread_id: string;
}

export type Params_thread_create_v4 = { [k: string]: unknown } | { [k: string]: unknown };
export interface Params_thread_delete_v4 {
  thread_id: string;
}

export interface Params_thread_get_v4 {
  refresh?: boolean;
  thread_id: string;
}

export interface Params_thread_list_v4 {
  filter: "active" | "archived" | "all";
  project_id?: string;
}

export interface Params_thread_navigate_v4 {
  node_id: string;
  thread_id: string;
}

export interface Params_thread_next_turn_v4 {
  thread_id: string;
}

export interface Params_thread_rename_v4 {
  thread_id: string;
  title: string;
}

export interface Params_thread_reorder_v4 {
  project_id: string;
  thread_ids: (string)[];
}

export interface Params_thread_set_archived_v4 {
  archived: boolean;
  thread_id: string;
}

export interface Params_thread_set_pinned_v4 {
  pinned: boolean;
  thread_id: string;
}

export interface Params_thread_set_read_v4 {
  read: boolean;
  thread_id: string;
}

export interface Params_thread_tree_v4 {
  thread_id: string;
}

export interface Params_tools_search_test_v4 {
  provider?: { [k: string]: unknown };
  query: string;
}

export interface Params_turn_cancel_v4 {
  operation_id: string;
}

export interface Params_turn_follow_up_v4 {
  text: string;
  thread_id: string;
}

export interface Params_turn_start_v4 {
  attachment_ids: (string)[];
  text: string;
  thread_id: string;
}

export interface Params_turn_steer_v4 {
  operation_id: string;
  text: string;
}

export interface Params_workspace_list_v4 {
  relative_path: string;
  root_id: string;
}

export interface Params_workspace_roots_v4 {
}

export interface Result_artifact_download_v4 {
  data_base64: string;
  mime_type: string;
}

export interface Result_artifact_resolve_v4 {
  artifacts: ({ [k: string]: unknown })[];
}

export interface Result_attachment_delete_v4 {
  deleted: boolean;
}

export interface Result_attachment_download_v4 {
  data_base64: string;
  mime_type: string;
  name: string;
}

export interface Result_attachment_preview_v4 {
  kind: string;
  preview: string;
}

export interface Result_attachment_upload_v4 {
  attachment_id: string;
  mime_type: string;
  name: string;
  size: number;
}

export interface Result_catalog_get_v4 {
  default_model_id: string | null;
  models: ({ [k: string]: unknown })[];
  prompt_templates: ({ [k: string]: unknown })[];
  providers: ({ [k: string]: unknown })[];
  skills: ({ [k: string]: unknown })[];
  tools: ({ [k: string]: unknown })[];
}

export interface Result_devices_claim_v4 {
  device_id: string;
  token: string;
}

export interface Result_devices_enroll_v4 {
  code: string;
  expires_at: string;
  pairing_url: string;
}

export interface Result_devices_list_v4 {
  devices: (Device_v4)[];
}

export interface Result_devices_revoke_v4 {
  devices: (Device_v4)[];
  revoked: boolean;
}

export interface Result_diagnostics_logs_v4 {
  entries: ({ [k: string]: unknown })[];
  truncated: boolean;
}

export interface Result_events_subscribe_v4 {
  current_seq: number;
  cursor: { [k: string]: unknown };
  events: ({ [k: string]: unknown })[];
  replay_complete: boolean;
  server_instance_id: string;
}

export interface Result_fs_list_v4 {
  entries: ({ [k: string]: unknown })[];
}

export interface Result_fs_read_v4 {
  content: string;
  encoding: string;
  truncated: boolean;
}

export interface Result_fs_roots_v4 {
  roots: ({ [k: string]: unknown })[];
}

export interface Result_git_branch_create_v4 {
  branch: string | null;
  ok: boolean;
  stderr: string;
  stdout: string;
}

export interface Result_git_branches_v4 {
  branches: (string)[];
  current: string | null;
  ok: boolean;
  stderr: string;
}

export interface Result_git_changes_v4 {
  branch: string | null;
  changes: GitChangeGroup_v4;
  staged: GitChangeGroup_v4;
}

export interface Result_git_commit_v4 {
  commit: string;
  ok: boolean;
  stderr: string;
  stdout: string;
}

export interface Result_git_diff_v4 {
  binary: boolean;
  ok: boolean;
  patch: { [k: string]: unknown };
  path: string;
  scope: string;
  stderr: string;
  truncated: boolean;
}

export interface Result_git_github_status_v4 {
  authenticated: boolean;
  github_origin: string;
  remote: string;
  stderr: string;
}

export interface Result_git_pr_create_v4 {
  ok: boolean;
  stderr: string;
  stdout: string;
  url: string;
}

export interface Result_git_push_v4 {
  ok: boolean;
  pushed: boolean;
  remote: string;
  stderr: string;
  stdout: string;
}

export interface Result_git_stage_v4 {
  ok: boolean;
  stderr: string;
  stdout: string;
}

export interface Result_git_status_v4 {
  branch: string | null;
  entries: ({ [k: string]: unknown })[];
  ok: boolean;
  stderr: string;
}

export interface Result_git_unstage_v4 {
  ok: boolean;
  stderr: string;
  stdout: string;
}

export interface Result_plugin_cloud_account_status_v4 {
  authenticated: boolean;
  base_url: string;
  enabled: boolean;
  ok?: boolean;
  user: Record<string, never> | null;
  vault_kind: string;
}

export interface Result_plugin_knowledge_index_v4 {
  [k: string]: unknown;
}

export interface Result_plugin_knowledge_search_v4 {
  [k: string]: unknown;
}

export interface Result_plugin_knowledge_stat_v4 {
  [k: string]: unknown;
}

export interface Result_plugin_memory_forget_v4 {
  [k: string]: unknown;
}

export interface Result_plugin_memory_recall_v4 {
  [k: string]: unknown;
}

export interface Result_plugin_memory_remember_v4 {
  [k: string]: unknown;
}

export interface Result_plugins_configure_v4 {
  plugin: string;
  revision: number;
}

export interface Result_plugins_list_v4 {
  plugins: ({ [k: string]: unknown })[];
}

export interface Result_plugins_set_enabled_v4 {
  plugin: string;
}

export interface Result_project_add_v4 {
  project: { [k: string]: unknown };
  projects: ({ [k: string]: unknown })[];
}

export interface Result_project_list_v4 {
  projects: ({ [k: string]: unknown })[];
}

export interface Result_project_refresh_v4 {
  project: { [k: string]: unknown };
  projects: ({ [k: string]: unknown })[];
}

export interface Result_project_remove_v4 {
  removed: boolean;
}

export interface Result_provider_add_v4 {
  provider: { [k: string]: unknown };
}

export interface Result_provider_list_v4 {
  providers: ({ [k: string]: unknown })[];
}

export interface Result_provider_refresh_v4 {
  provider: { [k: string]: unknown };
}

export interface Result_provider_remove_v4 {
  removed: boolean;
}

export interface Result_settings_get_v4 {
  revision: number;
  settings: { [k: string]: unknown };
}

export interface Result_settings_update_v4 {
  revision: number;
  settings: { [k: string]: unknown };
}

export interface Result_system_capabilities_v4 {
  capabilities: ({ [k: string]: unknown })[];
}

export interface Result_system_handshake_v4 {
  host: { [k: string]: unknown };
  limits: { [k: string]: unknown };
  protocol: "4.0.0";
  runtime: { [k: string]: unknown };
}

export interface Result_system_health_v4 {
  active_operations: number;
  ok: boolean;
  uptime_s: number;
}

export interface Result_system_shutdown_v4 {
  accepted: boolean;
}

export interface Result_system_snapshot_v4 {
  default_workspace: string;
  projects: ({ [k: string]: unknown })[];
  threads: ({ [k: string]: unknown })[];
}

export interface Result_system_uninstall_inspect_v4 {
  active_operations: number;
  data_paths: (string)[];
  estimated_bytes: number;
  worktrees: ({ [k: string]: unknown })[];
}

export interface Result_system_uninstall_prepare_v4 {
  prepared: boolean;
  removed_worktrees: ({ [k: string]: unknown })[];
}

export interface Result_terminal_close_v4 {
  closed: boolean;
}

export interface Result_terminal_input_v4 {
  accepted: boolean;
}

export interface Result_terminal_open_v4 {
  columns: number;
  opened: boolean;
  rows: number;
  terminal_id: string;
}

export interface Result_terminal_resize_v4 {
  accepted: boolean;
}

export interface Result_thread_compact_v4 {
  compacted: boolean;
  stats: { [k: string]: unknown };
}

export interface Result_thread_configure_v4 {
  thread: { [k: string]: unknown };
}

export interface Result_thread_context_v4 {
  maximum: number | null;
  ratio: number | null;
  used: number;
}

export interface Result_thread_create_v4 {
  thread: { [k: string]: unknown };
  threads: ({ [k: string]: unknown })[];
}

export interface Result_thread_delete_v4 {
  deleted: boolean;
}

export interface Result_thread_get_v4 {
  events: ({ [k: string]: unknown })[];
  history: ({ [k: string]: unknown })[];
  stats: { [k: string]: unknown };
  thread: { [k: string]: unknown };
  turns: ({ [k: string]: unknown })[];
}

export interface Result_thread_list_v4 {
  threads: ({ [k: string]: unknown })[];
}

export interface Result_thread_navigate_v4 {
  current: string | null;
  stats: { [k: string]: unknown };
}

export interface Result_thread_next_turn_v4 {
  turn_id: string;
}

export interface Result_thread_rename_v4 {
  thread: { [k: string]: unknown };
}

export interface Result_thread_reorder_v4 {
  threads: ({ [k: string]: unknown })[];
}

export interface Result_thread_set_archived_v4 {
  thread: { [k: string]: unknown };
}

export interface Result_thread_set_pinned_v4 {
  thread: { [k: string]: unknown };
}

export interface Result_thread_set_read_v4 {
  updated: boolean;
}

export interface Result_thread_tree_v4 {
  current: string | null;
  nodes: ({ [k: string]: unknown })[];
}

export interface Result_tools_search_test_v4 {
  engine: string;
  error?: string;
  results: ({ [k: string]: unknown })[];
}

export interface Result_turn_cancel_v4 {
  cancelling: boolean;
}

export interface Result_turn_follow_up_v4 {
  accepted: boolean;
}

export interface Result_turn_start_v4 {
  operation_id: string;
  turn_id: string;
}

export interface Result_turn_steer_v4 {
  accepted: boolean;
}

export interface Result_workspace_list_v4 {
  directory: string;
  entries: ({ [k: string]: unknown })[];
}

export interface Result_workspace_roots_v4 {
  roots: ({ [k: string]: unknown })[];
}

export interface AeloonRuntimeRpcDefinitions {}

export interface RuntimeRpcMethodMap {
  "artifact.download": { params: Params_artifact_download_v4; result: Result_artifact_download_v4 };
  "artifact.resolve": { params: Params_artifact_resolve_v4; result: Result_artifact_resolve_v4 };
  "attachment.delete": { params: Params_attachment_delete_v4; result: Result_attachment_delete_v4 };
  "attachment.download": { params: Params_attachment_download_v4; result: Result_attachment_download_v4 };
  "attachment.preview": { params: Params_attachment_preview_v4; result: Result_attachment_preview_v4 };
  "attachment.upload": { params: Params_attachment_upload_v4; result: Result_attachment_upload_v4 };
  "catalog.get": { params: Params_catalog_get_v4; result: Result_catalog_get_v4 };
  "devices.claim": { params: Params_devices_claim_v4; result: Result_devices_claim_v4 };
  "devices.enroll": { params: Params_devices_enroll_v4; result: Result_devices_enroll_v4 };
  "devices.list": { params: Params_devices_list_v4; result: Result_devices_list_v4 };
  "devices.revoke": { params: Params_devices_revoke_v4; result: Result_devices_revoke_v4 };
  "diagnostics.logs": { params: Params_diagnostics_logs_v4; result: Result_diagnostics_logs_v4 };
  "events.subscribe": { params: Params_events_subscribe_v4; result: Result_events_subscribe_v4 };
  "fs.list": { params: Params_fs_list_v4; result: Result_fs_list_v4 };
  "fs.read": { params: Params_fs_read_v4; result: Result_fs_read_v4 };
  "fs.roots": { params: Params_fs_roots_v4; result: Result_fs_roots_v4 };
  "git.branch_create": { params: Params_git_branch_create_v4; result: Result_git_branch_create_v4 };
  "git.branches": { params: Params_git_branches_v4; result: Result_git_branches_v4 };
  "git.changes": { params: Params_git_changes_v4; result: Result_git_changes_v4 };
  "git.commit": { params: Params_git_commit_v4; result: Result_git_commit_v4 };
  "git.diff": { params: Params_git_diff_v4; result: Result_git_diff_v4 };
  "git.github_status": { params: Params_git_github_status_v4; result: Result_git_github_status_v4 };
  "git.pr_create": { params: Params_git_pr_create_v4; result: Result_git_pr_create_v4 };
  "git.push": { params: Params_git_push_v4; result: Result_git_push_v4 };
  "git.stage": { params: Params_git_stage_v4; result: Result_git_stage_v4 };
  "git.status": { params: Params_git_status_v4; result: Result_git_status_v4 };
  "git.unstage": { params: Params_git_unstage_v4; result: Result_git_unstage_v4 };
  "plugins.configure": { params: Params_plugins_configure_v4; result: Result_plugins_configure_v4 };
  "plugins.list": { params: Params_plugins_list_v4; result: Result_plugins_list_v4 };
  "plugins.set_enabled": { params: Params_plugins_set_enabled_v4; result: Result_plugins_set_enabled_v4 };
  "project.add": { params: Params_project_add_v4; result: Result_project_add_v4 };
  "project.list": { params: Params_project_list_v4; result: Result_project_list_v4 };
  "project.refresh": { params: Params_project_refresh_v4; result: Result_project_refresh_v4 };
  "project.remove": { params: Params_project_remove_v4; result: Result_project_remove_v4 };
  "provider.add": { params: Params_provider_add_v4; result: Result_provider_add_v4 };
  "provider.list": { params: Params_provider_list_v4; result: Result_provider_list_v4 };
  "provider.refresh": { params: Params_provider_refresh_v4; result: Result_provider_refresh_v4 };
  "provider.remove": { params: Params_provider_remove_v4; result: Result_provider_remove_v4 };
  "settings.get": { params: Params_settings_get_v4; result: Result_settings_get_v4 };
  "settings.update": { params: Params_settings_update_v4; result: Result_settings_update_v4 };
  "system.capabilities": { params: Params_system_capabilities_v4; result: Result_system_capabilities_v4 };
  "system.handshake": { params: Params_system_handshake_v4; result: Result_system_handshake_v4 };
  "system.health": { params: Params_system_health_v4; result: Result_system_health_v4 };
  "system.shutdown": { params: Params_system_shutdown_v4; result: Result_system_shutdown_v4 };
  "system.snapshot": { params: Params_system_snapshot_v4; result: Result_system_snapshot_v4 };
  "system.uninstall_inspect": { params: Params_system_uninstall_inspect_v4; result: Result_system_uninstall_inspect_v4 };
  "system.uninstall_prepare": { params: Params_system_uninstall_prepare_v4; result: Result_system_uninstall_prepare_v4 };
  "terminal.close": { params: Params_terminal_close_v4; result: Result_terminal_close_v4 };
  "terminal.input": { params: Params_terminal_input_v4; result: Result_terminal_input_v4 };
  "terminal.open": { params: Params_terminal_open_v4; result: Result_terminal_open_v4 };
  "terminal.resize": { params: Params_terminal_resize_v4; result: Result_terminal_resize_v4 };
  "thread.compact": { params: Params_thread_compact_v4; result: Result_thread_compact_v4 };
  "thread.configure": { params: Params_thread_configure_v4; result: Result_thread_configure_v4 };
  "thread.context": { params: Params_thread_context_v4; result: Result_thread_context_v4 };
  "thread.create": { params: Params_thread_create_v4; result: Result_thread_create_v4 };
  "thread.delete": { params: Params_thread_delete_v4; result: Result_thread_delete_v4 };
  "thread.get": { params: Params_thread_get_v4; result: Result_thread_get_v4 };
  "thread.list": { params: Params_thread_list_v4; result: Result_thread_list_v4 };
  "thread.navigate": { params: Params_thread_navigate_v4; result: Result_thread_navigate_v4 };
  "thread.next_turn": { params: Params_thread_next_turn_v4; result: Result_thread_next_turn_v4 };
  "thread.rename": { params: Params_thread_rename_v4; result: Result_thread_rename_v4 };
  "thread.reorder": { params: Params_thread_reorder_v4; result: Result_thread_reorder_v4 };
  "thread.set_archived": { params: Params_thread_set_archived_v4; result: Result_thread_set_archived_v4 };
  "thread.set_pinned": { params: Params_thread_set_pinned_v4; result: Result_thread_set_pinned_v4 };
  "thread.set_read": { params: Params_thread_set_read_v4; result: Result_thread_set_read_v4 };
  "thread.tree": { params: Params_thread_tree_v4; result: Result_thread_tree_v4 };
  "tools.search_test": { params: Params_tools_search_test_v4; result: Result_tools_search_test_v4 };
  "turn.cancel": { params: Params_turn_cancel_v4; result: Result_turn_cancel_v4 };
  "turn.follow_up": { params: Params_turn_follow_up_v4; result: Result_turn_follow_up_v4 };
  "turn.start": { params: Params_turn_start_v4; result: Result_turn_start_v4 };
  "turn.steer": { params: Params_turn_steer_v4; result: Result_turn_steer_v4 };
  "workspace.list": { params: Params_workspace_list_v4; result: Result_workspace_list_v4 };
  "workspace.roots": { params: Params_workspace_roots_v4; result: Result_workspace_roots_v4 };
}
export type RuntimeMethod = keyof RuntimeRpcMethodMap;
export type RuntimeRpcParams<M extends RuntimeMethod> = RuntimeRpcMethodMap[M]["params"];
export type RuntimeRpcResult<M extends RuntimeMethod> = RuntimeRpcMethodMap[M]["result"];

export interface RuntimePluginMethodMap {
  "plugin.cloud.account_login": { params: Params_plugin_cloud_account_login_v4; result: Result_plugin_cloud_account_status_v4 };
  "plugin.cloud.account_logout": { params: Params_plugin_cloud_account_logout_v4; result: Result_plugin_cloud_account_status_v4 };
  "plugin.cloud.account_status": { params: Params_plugin_cloud_account_status_v4; result: Result_plugin_cloud_account_status_v4 };
  "plugin.knowledge.index": { params: Params_plugin_knowledge_index_v4; result: Result_plugin_knowledge_index_v4 };
  "plugin.knowledge.search": { params: Params_plugin_knowledge_search_v4; result: Result_plugin_knowledge_search_v4 };
  "plugin.knowledge.stat": { params: Params_plugin_knowledge_stat_v4; result: Result_plugin_knowledge_stat_v4 };
  "plugin.memory.forget": { params: Params_plugin_memory_forget_v4; result: Result_plugin_memory_forget_v4 };
  "plugin.memory.recall": { params: Params_plugin_memory_recall_v4; result: Result_plugin_memory_recall_v4 };
  "plugin.memory.remember": { params: Params_plugin_memory_remember_v4; result: Result_plugin_memory_remember_v4 };
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
  | (RuntimeEventBase & { name: "capabilities.updated"; payload: Event_capabilities_updated_v4 })
  | (RuntimeEventBase & { name: "content.completed"; payload: Event_content_completed_v4 })
  | (RuntimeEventBase & { name: "content.delta"; payload: Event_content_delta_v4 })
  | (RuntimeEventBase & { name: "content.started"; payload: Event_content_started_v4 })
  | (RuntimeEventBase & { name: "content.updated"; payload: Event_content_updated_v4 })
  | (RuntimeEventBase & { name: "log.entry"; payload: Event_log_entry_v4 })
  | (RuntimeEventBase & { name: "operation.cancelled"; payload: Event_operation_cancelled_v4 })
  | (RuntimeEventBase & { name: "operation.cancelling"; payload: Event_operation_cancelling_v4 })
  | (RuntimeEventBase & { name: "operation.completed"; payload: Event_operation_completed_v4 })
  | (RuntimeEventBase & { name: "operation.failed"; payload: Event_operation_failed_v4 })
  | (RuntimeEventBase & { name: "operation.queued"; payload: Event_operation_queued_v4 })
  | (RuntimeEventBase & { name: "operation.started"; payload: Event_operation_started_v4 })
  | (RuntimeEventBase & { name: "plugin.<id>.*"; payload: Event_plugin_id_v4 })
  | (RuntimeEventBase & { name: "plugin.cloud.account_updated"; payload: Event_plugin_cloud_account_updated_v4 })
  | (RuntimeEventBase & { name: "provider.updated"; payload: Event_provider_updated_v4 })
  | (RuntimeEventBase & { name: "queue.updated"; payload: Event_queue_updated_v4 })
  | (RuntimeEventBase & { name: "retry.completed"; payload: Event_retry_completed_v4 })
  | (RuntimeEventBase & { name: "retry.started"; payload: Event_retry_started_v4 })
  | (RuntimeEventBase & { name: "settings.updated"; payload: Event_settings_updated_v4 })
  | (RuntimeEventBase & { name: "system.shutdown"; payload: Event_system_shutdown_v4 })
  | (RuntimeEventBase & { name: "terminal.exit"; payload: Event_terminal_exit_v4 })
  | (RuntimeEventBase & { name: "terminal.opened"; payload: Event_terminal_opened_v4 })
  | (RuntimeEventBase & { name: "terminal.output"; payload: Event_terminal_output_v4 })
  | (RuntimeEventBase & { name: "thread.compacted"; payload: Event_thread_compacted_v4 })
  | (RuntimeEventBase & { name: "thread.navigated"; payload: Event_thread_navigated_v4 })
  | (RuntimeEventBase & { name: "thread.renamed"; payload: Event_thread_renamed_v4 })
  | (RuntimeEventBase & { name: "tool.completed"; payload: Event_tool_completed_v4 })
  | (RuntimeEventBase & { name: "tool.started"; payload: Event_tool_started_v4 })
  | (RuntimeEventBase & { name: "tool.updated"; payload: Event_tool_updated_v4 })
  | (RuntimeEventBase & { name: "turn.created"; payload: Event_turn_created_v4 })
  | (RuntimeEventBase & { name: "usage.updated"; payload: Event_usage_updated_v4 })
;
export type RuntimeEventName = RuntimeEvent["name"];

export const RUNTIME_RPC_PROTOCOL = "aeloon-rpc" as const;
export const RUNTIME_RPC_VERSION = "4.0.0" as const;
export const RUNTIME_RPC_MAX_FRAME_BYTES = 41943040 as const;
