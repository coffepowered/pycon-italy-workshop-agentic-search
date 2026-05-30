
You can also use this repo as a "skill" for the purpose of the worksop.

# How to test the server?
> uv run fastmcp dev inspector server.py

it also enables auto-reload!

# How to add the mcp server to Codex?

> codex mcp add search_my_files -- uv --directory "$(pwd)" run python server.py

# How do I see my token consuption?

> bunx ccusage codex session --color --since 2026-05-29