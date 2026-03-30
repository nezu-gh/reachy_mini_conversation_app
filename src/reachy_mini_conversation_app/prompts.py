import re
import sys
import logging
from pathlib import Path

from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config


logger = logging.getLogger(__name__)


PROMPTS_LIBRARY_DIRECTORY = Path(__file__).parent / "prompts"
INSTRUCTIONS_FILENAME = "instructions.txt"
INSTRUCTIONS_LEAN_FILENAME = "instructions_lean.txt"
VOICE_FILENAME = "voice.txt"


def _expand_prompt_includes(content: str) -> str:
    """Expand [<name>] placeholders with content from prompts library files.

    Args:
        content: The template content with [<name>] placeholders

    Returns:
        Expanded content with placeholders replaced by file contents

    """
    # Pattern to match [<name>] where name is a valid file stem (alphanumeric, underscores, hyphens)
    # pattern = re.compile(r'^\[([a-zA-Z0-9_-]+)\]$')
    # Allow slashes for subdirectories
    pattern = re.compile(r'^\[([a-zA-Z0-9/_-]+)\]$')

    lines = content.split('\n')
    expanded_lines = []

    for line in lines:
        stripped = line.strip()
        match = pattern.match(stripped)

        if match:
            # Extract the name from [<name>]
            template_name = match.group(1)
            template_file = PROMPTS_LIBRARY_DIRECTORY / f"{template_name}.txt"

            try:
                if template_file.exists():
                    template_content = template_file.read_text(encoding="utf-8").rstrip()
                    expanded_lines.append(template_content)
                    logger.debug("Expanded template: [%s]", template_name)
                else:
                    logger.warning("Template file not found: %s, keeping placeholder", template_file)
                    expanded_lines.append(line)
            except Exception as e:
                logger.warning("Failed to read template '%s': %s, keeping placeholder", template_name, e)
                expanded_lines.append(line)
        else:
            expanded_lines.append(line)

    return '\n'.join(expanded_lines)


def get_session_instructions() -> str:
    """Get session instructions, loading from REACHY_MINI_CUSTOM_PROFILE if set.

    When PIPELINE_MODE=lean, loads the lean prompt variant
    (instructions_lean.txt) from the active profile if it exists,
    falling back to a built-in minimal prompt.
    """
    import os
    lean_mode = os.environ.get("PIPELINE_MODE", "full").lower() == "lean"

    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        logger.info(f"Loading default prompt from {PROMPTS_LIBRARY_DIRECTORY / 'default_prompt.txt'}")
        instructions_file = PROMPTS_LIBRARY_DIRECTORY / "default_prompt.txt"
    else:
        if config.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.info(
                "Loading prompt from external profile '%s' (root=%s)",
                profile,
                config.PROFILES_DIRECTORY,
            )
        else:
            logger.info(f"Loading prompt from profile '{profile}'")
        instructions_file = config.PROFILES_DIRECTORY / profile / INSTRUCTIONS_FILENAME

    # In lean mode, prefer the lean prompt variant if available
    if lean_mode and profile:
        lean_file = config.PROFILES_DIRECTORY / profile / INSTRUCTIONS_LEAN_FILENAME
        if lean_file.exists():
            logger.info("Pipeline mode: lean — loading %s", lean_file.name)
            instructions_file = lean_file
        else:
            logger.info("Pipeline mode: lean — no lean prompt found, using standard")
    elif lean_mode:
        logger.info("Pipeline mode: lean — no profile set, using default prompt")

    try:
        if instructions_file.exists():
            instructions = instructions_file.read_text(encoding="utf-8").strip()
            if instructions:
                # Expand [<name>] placeholders with content from prompts library
                expanded_instructions = _expand_prompt_includes(instructions)
                return expanded_instructions
            logger.error(f"Profile '{profile}' has empty {INSTRUCTIONS_FILENAME}")
            sys.exit(1)
        logger.error(f"Profile {profile} has no {INSTRUCTIONS_FILENAME}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load instructions from profile '{profile}': {e}")
        sys.exit(1)


def get_session_voice(default: str = "cedar") -> str:
    """Resolve the voice to use for the session.

    If a custom profile is selected and contains a voice.txt, return its
    trimmed content; otherwise return the provided default ("cedar").
    """
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        return default
    try:
        voice_file = config.PROFILES_DIRECTORY / profile / VOICE_FILENAME
        if voice_file.exists():
            voice = voice_file.read_text(encoding="utf-8").strip()
            return voice or default
    except Exception:
        pass
    return default
