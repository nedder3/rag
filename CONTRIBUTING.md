# Contributing

1. **Conventional Commits:** Every commit must follow `<type>(<scope>): <description>` (`feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`).
2. **TDD First:** Add failing unit or integration tests in `tests/` before implementing changes.
3. **Verification:** Run `cargo test` against PostgreSQL 17 (`pgvector`) before opening a PR.
