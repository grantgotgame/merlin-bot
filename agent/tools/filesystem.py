"""Filesystem tools — read, write, list files within the RBOS sandbox."""

from pathlib import Path
from typing import Optional
from .base import BaseTool
from ..config import RBOS_ROOT, MAX_FILE_CHARS


def _safe_resolve(user_path: str) -> Optional[Path]:
    """Resolve a user path against RBOS_ROOT. Returns None if it escapes the sandbox."""
    try:
        if user_path.startswith("/"):
            return None
        resolved = (RBOS_ROOT / user_path).resolve()
        if not str(resolved).startswith(str(RBOS_ROOT)):
            return None
        return resolved
    except Exception:
        return None


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file in the RBOS filesystem."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to RBOS root (e.g. 'core/STATE.md')",
                },
            },
            "required": ["path"],
        }

    def execute(self, **kwargs) -> str:
        path = kwargs.get("path", "")
        resolved = _safe_resolve(path)
        if resolved is None:
            return f"Error: path '{path}' is outside the RBOS sandbox."

        if not resolved.exists():
            return f"Error: file '{path}' does not exist."
        if not resolved.is_file():
            return f"Error: '{path}' is not a file (maybe a directory?)."

        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
            if len(text) > MAX_FILE_CHARS:
                return text[:MAX_FILE_CHARS] + f"\n\n[truncated — {len(text)} total chars, showing first {MAX_FILE_CHARS}]"
            return text
        except Exception as e:
            return f"Error reading '{path}': {e}"


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file in the RBOS filesystem. Creates parent directories if needed."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to RBOS root",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["path", "content"],
        }

    def execute(self, **kwargs) -> str:
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        resolved = _safe_resolve(path)
        if resolved is None:
            return f"Error: path '{path}' is outside the RBOS sandbox."

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"
        except Exception as e:
            return f"Error writing '{path}': {e}"


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List files and directories at a path in the RBOS filesystem."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to RBOS root (default: root)",
                    "default": ".",
                },
            },
            "required": [],
        }

    def execute(self, **kwargs) -> str:
        path = kwargs.get("path", ".")
        resolved = _safe_resolve(path)
        if resolved is None:
            return f"Error: path '{path}' is outside the RBOS sandbox."

        if not resolved.exists():
            return f"Error: '{path}' does not exist."
        if not resolved.is_dir():
            return f"Error: '{path}' is not a directory."

        try:
            entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            lines = []
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    lines.append(f"[dir]  {entry.name}/")
                else:
                    size = entry.stat().st_size
                    if size >= 1024 * 1024:
                        size_str = f"{size / (1024*1024):.1f} MB"
                    elif size >= 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size} B"
                    lines.append(f"[file] {entry.name} ({size_str})")
            return "\n".join(lines) if lines else "(empty directory)"
        except Exception as e:
            return f"Error listing '{path}': {e}"


def default_tools() -> list[BaseTool]:
    """Return all filesystem tools."""
    return [ReadFileTool(), WriteFileTool(), ListDirectoryTool()]
