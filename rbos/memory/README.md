# Memory

Long-term context that persists across sessions. The agent reads these to understand who you are, what you're working on, and how to help.

## What Goes Here

- **People:** Key people in your life/work with context the agent needs
- **Projects:** Active project details, decisions made, status
- **Preferences:** How you like to work, what to avoid, what works
- **References:** Links to external resources, accounts, tools

## File Naming

- `user_[topic].md` — About you (role, preferences, background)
- `project_[name].md` — Project details
- `feedback_[topic].md` — Things the agent should remember (corrections, preferences)
- `reference_[topic].md` — External resources and pointers

## How It Works

The agent creates memory files as it learns about you. You can also create them manually. Each file should have a clear, specific purpose.

Don't put ephemeral stuff here — this is for things that matter across sessions, not within a single session.
