# Breaking runtime rewrite

The Electron runtime rewrite is intentionally incompatible with prior desktop transports and
data. There is no automatic protocol, Session, or configuration migration path.

The supported desktop integration is now:

- `aeloon rpc serve --socket <path> [--browser-runtime-socket <path>]` for the Bun Workbench;
- `aeloon-rpc-v1` for Workbench-to-Core requests and events;
- `browser-runtime-v1` for Core Browser Use execution;
- one UI thread UUID used directly as the Core Session and Browser scope ID.

Start with the new Electron application data directory. Old desktop data is neither read nor
deleted. Python API consumers can continue to compose `RuntimeService` directly, but removed CLI
commands and transports have no compatibility aliases.
