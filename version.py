#!/usr/bin/env python3
"""
Version utility for the FinGPT AI Day Trading System.

This module provides utilities for handling semantic versioning according to SemVer 2.0.0.
"""

import re
import os
from dataclasses import dataclass
from typing import Optional, Union


# Current version of the FinGPT AI Day Trading System
__version__ = "1.0.0"


@dataclass


class Version:
    """ Represents a semantic version with optional prerelease and build
        metadata.
    """

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        """Convert version to string representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: 'Version') -> bool:
        """Compare versions according to SemVer precedence rules."""
        if not isinstance(other, Version):
            return NotImplemented

        # Compare major, minor, patch
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch

        # A version with prerelease has lower precedence than without
        if self.prerelease is None and other.prerelease is not None:
            return False
        if self.prerelease is not None and other.prerelease is None:
            return True

        # Compare prerelease identifiers
        if self.prerelease and other.prerelease:
            self_parts = self.prerelease.split('.')
            other_parts = other.prerelease.split('.')

            for i in range(min(len(self_parts), len(other_parts))):
                if self_parts[i].isdigit() and other_parts[i].isdigit():
                    if int(self_parts[i]) != int(other_parts[i]):
                        return int(self_parts[i]) < int(other_parts[i])
                elif self_parts[i].isdigit():
                    return True  # Numeric has lower precedence
                elif other_parts[i].isdigit():
                    return False  # Numeric has lower precedence
                elif self_parts[i] != other_parts[i]:
                    return self_parts[i] < other_parts[i]

            return len(self_parts) < len(other_parts)

        # Build metadata does not affect precedence
        return False

    def __eq__(self, other: object) -> bool:
        """Check if versions are equal (ignoring build metadata)."""
        if not isinstance(other, Version):
            return NotImplemented

        return (self.major == other.major and
                self.minor == other.minor and
                self.patch == other.patch and
                self.prerelease == other.prerelease)


def parse_version(version_str: str) -> Version:
    """
    Parse a version string into a Version object.

    Args:
        version_str: A string following SemVer format (e.g., "1.2.3-alpha+build.1")

    Returns:
        A Version object

    Raises:
        ValueError: If the version string does not follow SemVer format
    """
    pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
    match = re.match(pattern, version_str)

    if not match:
        raise ValueError(f"Invalid version string: {version_str}")

    major, minor, patch, prerelease, build = match.groups()

    return Version(
        major=int(major),
        minor=int(minor),
        patch=int(patch),
        prerelease=prerelease,
        build=build
    )


def get_current_version() -> Version:
    """
    Get the current version of the FinGPT AI Day Trading System.

    Returns:
        A Version object representing the current version
    """
    return parse_version(__version__)


def increment_version(
    version: Union[str, Version],
    major: bool = False,
    minor: bool = False,
    patch: bool = False
) -> Version:
    """
    Increment a version according to SemVer rules.

    Args:
        version: The version to increment (string or Version object)
        major: Whether to increment the major version
        minor: Whether to increment the minor version
        patch: Whether to increment the patch version

    Returns:
        A new Version object with the incremented version

    Raises:
        ValueError: If no increment type is specified
    """
    if isinstance(version, str):
        version = parse_version(version)

    if not any([major, minor, patch]):
        raise ValueError("At least one increment type must be specified")

    if major:
        return Version(version.major + 1, 0, 0)
    elif minor:
        return Version(version.major, version.minor + 1, 0)
    elif patch:
        return Version(version.major, version.minor, version.patch + 1)


def is_compatible(
    current: Union[str, Version],
    required: Union[str, Version]
) -> bool:
    """
    Check if the current version is compatible with the required version.

    According to SemVer:
    - Major version changes indicate incompatible API changes
    - Minor and patch changes should be backward compatible

    Args:
        current: The current version (string or Version object)
        required: The required version (string or Version object)

    Returns:
        True if compatible, False otherwise
    """
    if isinstance(current, str):
        current = parse_version(current)
    if isinstance(required, str):
        required = parse_version(required)

    # Same major version is required for compatibility
    if current.major != required.major:
        return False

    # Current minor/patch must be >= required
    if current.minor < required.minor:
        return False
    if current.minor == required.minor and current.patch < required.patch:
        return False

    return True


def update_version_files(new_version: Union[str, Version]) -> None:
    """
    Update version references in project files.

    Args:
        new_version: The new version (string or Version object)
    """
    if isinstance(new_version, Version):
        new_version = str(new_version)

    # Update this file
    update_version_in_file('version.py', r'__version__ = "[^"]+"', f'__version__ = "{new_version}"')

    # Update fingpt/__init__.py
    update_version_in_file('fingpt/__init__.py', r'__version__ = "[^"]+"', f'__version__ = "{new_version}"')

    print(f"Updated version to {new_version} in version files.")


def update_version_in_file(filepath: str, pattern: str, replacement: str) -> None:
    """
    Update a version string in a file based on a pattern.

    Args:
        filepath: Path to the file to update
        pattern: Regex pattern to match the version line
        replacement: Replacement string with the new version
    """
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist, skipping version update.")
        return

    with open(filepath, 'r') as f:
        content = f.read()

    new_content = re.sub(pattern, replacement, content)

    with open(filepath, 'w') as f:
        f.write(new_content)


if __name__ == "__main__":
    # Command-line interface for version management
    import argparse

    parser = argparse.ArgumentParser(description="FinGPT AI Day Trading System version utility")
    parser.add_argument('--get', action='store_true', help="Get the current version")
    parser.add_argument('--increment', choices=['major', 'minor', 'patch'],
                       help="Increment the specified version component")
    parser.add_argument('--update-files', action='store_true',
                       help="Update version in project files (use with --increment)")

    args = parser.parse_args()

    if args.get:
        print(get_current_version())

    if args.increment:
        current = get_current_version()
        if args.increment == 'major':
            new_version = increment_version(current, major=True)
        elif args.increment == 'minor':
            new_version = increment_version(current, minor=True)
        else:  # patch
            new_version = increment_version(current, patch=True)

        print(f"Incremented {args.increment} version: {current} -> {new_version}")

        if args.update_files:
            update_version_files(new_version)
