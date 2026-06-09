# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool definitions for fixer agent in pandas compatibility pipeline.

Provides functions for:
- Reading/writing files in worktree
- Running shell commands
- Searching code with patterns
- Tool definitions for LLM function calling
"""

import logging
import subprocess
from pathlib import Path
from typing import Any

from ..utils.patch_validator import validate_patch

logger = logging.getLogger(__name__)


def read_file(path: str) -> str:
    """Read a file from worktree.

    Args:
        path: File path relative to repository root

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file does not exist
        PermissionError: If read access denied
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        raise


def write_file(path: str, content: str) -> dict[str, Any]:
    """Write a file to worktree with patch validation.

    Args:
        path: File path relative to repository root
        content: File contents to write

    Returns:
        Dictionary with:
        - "path": str - the file path
        - "success": bool - whether write succeeded
        - "bytes": int - number of bytes written
        - "validation": dict - patch validation result

    Raises:
        ValueError: If patch validation fails
        PermissionError: If write access denied
    """
    file_path = Path(path)

    # Validate patch before writing
    validation_result = validate_patch([path])
    if not validation_result.approved:
        error_msg = f"Patch validation failed: {validation_result.reason}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        bytes_written = file_path.write_text(content, encoding="utf-8")

        logger.info(f"Successfully wrote {bytes_written} bytes to {path}")

        return {
            "path": path,
            "success": True,
            "bytes": bytes_written,
            "validation": {
                "approved": validation_result.approved,
                "reason": validation_result.reason,
                "classification": validation_result.classification,
            },
        }
    except Exception as e:
        logger.error(f"Error writing {path}: {e}")
        return {
            "path": path,
            "success": False,
            "error": str(e),
            "validation": {
                "approved": False,
                "reason": validation_result.reason,
            },
        }


def run_command(
    cmd: str,
    cwd: str | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run a shell command with timeout and capture output.

    Args:
        cmd: Command string to execute
        cwd: Working directory for command (defaults to current)
        timeout: Command timeout in seconds (default: 300)

    Returns:
        Dictionary with:
        - "cmd": str - the command executed
        - "stdout": str - captured stdout
        - "stderr": str - captured stderr
        - "returncode": int - exit code
        - "success": bool - whether returncode == 0
        - "timed_out": bool - whether command timed out

    Raises:
        Exception: Only on internal errors, timeouts are returned in dict
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "cmd": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
            "timed_out": False,
        }

    except subprocess.TimeoutExpired as e:
        logger.warning(f"Command timed out after {timeout}s: {cmd}")
        return {
            "cmd": cmd,
            "stdout": e.stdout.decode() if e.stdout else "",
            "stderr": e.stderr.decode() if e.stderr else "",
            "returncode": None,
            "success": False,
            "timed_out": True,
            "error": f"Command timed out after {timeout}s",
        }

    except Exception as e:
        logger.error(f"Error running command: {e}")
        return {
            "cmd": cmd,
            "returncode": None,
            "success": False,
            "error": str(e),
        }


def search_code(pattern: str, path: str) -> str:
    """Search for code pattern in files using ripgrep or grep.

    Args:
        pattern: Regex pattern to search for
        path: Directory or file path to search in

    Returns:
        String with search results (one match per line)

    Raises:
        ValueError: If pattern is invalid
    """
    try:
        # Try ripgrep first (faster), fall back to grep
        try:
            result = subprocess.run(
                ["rg", "--no-heading", "-n", "-C", "2", pattern, path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode in (0, 1):  # 0 = match found, 1 = no match
                return result.stdout or "(no matches found)"
        except FileNotFoundError:
            # ripgrep not available, use grep
            result = subprocess.run(
                ["grep", "-r", "-n", "-C", "2", pattern, path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode in (0, 1):
                return result.stdout or "(no matches found)"

        logger.warning(f"Search returned non-zero code: {result.returncode}")
        return result.stderr or "(search failed)"

    except subprocess.TimeoutExpired:
        return "(search timed out after 30 seconds)"
    except Exception as e:
        logger.error(f"Error searching code: {e}")
        return f"(search error: {e})"


def get_tools() -> list[dict[str, Any]]:
    """Get tool definitions for LLM function calling.

    Returns:
        List of tool definitions in JSON Schema format for litellm
    """
    return [
        {
            "name": "read_file",
            "description": "Read contents of a file from the worktree",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to repository root",
                    }
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file in the worktree with validation",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to repository root",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "run_command",
            "description": "Execute a shell command and capture output",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (optional)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 300)",
                        "default": 300,
                    },
                },
                "required": ["cmd"],
            },
        },
        {
            "name": "search_code",
            "description": "Search for a regex pattern in code files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file path to search in",
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    ]
