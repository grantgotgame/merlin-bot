# RBOS File System Rules

*Where things go. Every agent checks this.*

---

## Directory Rules

| Directory | What goes here | What does NOT go here |
|-----------|---------------|----------------------|
| `core/` | Operating files (STATE, RUNTIME, BOOTSTRAP) | Drafts, logs, personal content |
| `briefing/` | One file per day (`YYYY-MM-DD.md`) | Anything not a daily briefing |
| `lifestyle/` | Rituals, routines, health protocols | Project work |
| `skills/` | Agent workflows (markdown protocols) | Code, scripts |
| `memory/` | Long-term context (people, projects, prefs) | Ephemeral notes |
| `inbox/` | Unsorted input — temporary holding | Permanent files |
| `queue/` | Night shift queue (YAML) | Anything else |
| `workspace/` | Active drafts, research, working docs | Completed work |
| `log/` | Sprint docs, changelog, shift logs | Active work |
| `product/` | Product specs, roadmap, backlog | Personal content |
| `archive/` | Completed/old work | Active work |
| `scripts/` | Shell scripts, Python utilities | Markdown docs |
| `merlin/` | Merlin config, briefing JSONs, personality | RBOS system files |

## File Rules

- No binary files in RBOS (except screenshots in `inbox/`)
- No code projects in RBOS — keep those in `~/Code/` or `~/Documents/Code/`
- No virtual environments in RBOS
- Secrets go in `.env` (git-ignored)
- One idea per file. If a file is doing two things, split it.

## Naming

- System files: `UPPER_CASE.md`
- Dated content: `YYYY-MM-DD_kebab-case.md`
- Working docs: `kebab-case.md`
- Underscores separate structural parts. Hyphens separate words.
