# Skills

Skills are agent workflows — structured protocols the AI follows when triggered.

## How Skills Work

Each skill is a markdown file that the agent reads and executes. You trigger them by saying the trigger phrase, and the agent follows the protocol.

## Available Skills

| Skill | Trigger | What it does |
|-------|---------|-------------|
| `checkpoint.md` | "Checkpoint" / "Save state" | Saves all work: git commit + open loop capture + state update |

## Creating New Skills

Create a new `.md` file in this directory with:
1. A clear trigger phrase
2. Step-by-step instructions the agent can follow
3. A definition of "done"

The agent will read the file when triggered and execute the steps.

## Tips

- Keep skills under 1 page
- Be specific about what "done" looks like
- Include the git commit step if the skill changes files
- Skills should be idempotent — running twice shouldn't break anything
