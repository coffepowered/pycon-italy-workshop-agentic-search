Material for the talk "Search in the era of AI agents".
You can also use this repo as a "skill" for the purpose of the worksop.

A few handy commands to get started

## Configuration

Install dependencies (requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/)):
> uv sync

Download the sample KB and drop it into the `documents/` folder (or bring your own):
> https://drive.google.com/file/d/1f4skbLlesutOawKFX6_xQYxJh4fO-0eO/view?usp=sharing

Install the MCP client harness ([codex cli](https://developers.openai.com/codex/cli)):
> npm install -g @openai/codex

## Optional: track token usage with ccusage
Install [bun](https://bun.sh/) (provides `bunx`, used for token tracking):
> npm install -g bun


## How to test the server?
> uv run fastmcp dev inspector server.py

it also enables auto-reload!

## How to add the mcp server to Codex?

> codex mcp add search_my_files -- uv --directory "$(pwd)" run python server.py

## How do I see my token consuption?

> bunx ccusage codex session --color --since 2026-05-30