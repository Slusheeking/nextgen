#!/usr/bin/env python3
"""
Release automation script for the FinGPT AI Day Trading System.

This script helps automate the release process by:
1. Incrementing version numbers
2. Updating version files
3. Creating tagged commits
4. Generating release notes
"""

import argparse
import datetime
import re
import subprocess
import sys

from version import get_current_version, increment_version, update_version_files


def confirm(message):
    """Ask for user confirmation."""
    response = input(f"{message} (y/n): ").lower().strip()
    return response == "y" or response == "yes"


def run_command(command, capture_output=True):
    """Run a shell command and return the output."""
    print(f"Executing: {command}")

    if capture_output:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.stdout and not capture_output:
            print(f"Output:\n{result.stdout}")

        if result.stderr and result.returncode != 0:
            print(f"Error:\n{result.stderr}")
            if not confirm("Command failed. Continue anyway?"):
                sys.exit(1)

        return result
    else:
        # Run command and stream output directly
        return subprocess.run(command, shell=True)


def get_changes_since_last_tag():
    """Get list of changes since the last tag."""
    # Get the last tag
    result = run_command("git describe --tags --abbrev=0 2>/dev/null || echo 'none'")
    last_tag = result.stdout.strip()

    if last_tag == 'none':
        # No previous tags, get all commits
        result = run_command("git log --pretty=format:'%h - %s (%an)' | head -n 15")
    else:
        # Get commits since the last tag
        result = run_command(f"git log {last_tag}..HEAD --pretty=format:'%h - %s (%an)' | head -n 15")

    return result.stdout.strip().split('\n')


def create_tag_message(version, changes):
    """Create a tag message with the changes."""
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    message = f"Version {version} ({current_date})\n\n"

    if changes:
        message += "Changes:\n"
        for change in changes:
            if change:  # Skip empty lines
                message += f"- {change}\n"

    return message


def update_version_md(version, changes):
    """Update VERSION.md with the new version and changes."""
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    entry = f"\n### {version} ({current_date})\n\n"

    if changes:
        entry += "**Changes:**\n"
        for change in changes:
            if change:  # Skip empty lines
                entry += f"- {change}\n"
        entry += "\n"

    # Read the current VERSION.md
    with open('VERSION.md', 'r') as f:
        content = f.read()

    # Find the "Version History" section
    version_history_match = re.search(r'## Version History\s+', content)
    if version_history_match:
        # Insert the new entry after the "Version History" header
        insert_position = version_history_match.end()
        updated_content = content[:insert_position] + entry + content[insert_position:]

        # Write the updated content back
        with open('VERSION.md', 'w') as f:
            f.write(updated_content)
    else:
        print("Warning: Could not find 'Version History' section in VERSION.md")


def create_release(increment_type):
    """Create a new release."""
    # Check if there are uncommitted changes
    result = run_command("git status --porcelain")
    if result.stdout.strip():
        if not confirm("There are uncommitted changes. Continue anyway?"):
            sys.exit(1)

    # Get the current version
    current_version = get_current_version()
    print(f"Current version: {current_version}")

    # Increment the version
    if increment_type == "major":
        new_version = increment_version(current_version, major=True)
    elif increment_type == "minor":
        new_version = increment_version(current_version, minor=True)
    else:  # patch
        new_version = increment_version(current_version, patch=True)

    print(f"New version: {new_version}")

    # Confirm the version increment
    if not confirm(f"Increment version from {current_version} to {new_version}?"):
        sys.exit(0)

    # Update version files
    update_version_files(str(new_version))

    # Get changes since the last tag
    changes = get_changes_since_last_tag()

    # Show changes and ask for confirmation
    print("\nChanges since last release:")
    for change in changes:
        print(f"  {change}")

    # Update VERSION.md
    update_version_md(str(new_version), changes)

    if not confirm("\nUpdate VERSION.md and create release commit?"):
        run_command("git checkout -- VERSION.md version.py fingpt/__init__.py")
        sys.exit(0)

    # Create a commit with the version changes
    commit_msg = f"Release version {new_version}"
    run_command('git add VERSION.md version.py fingpt/__init__.py')
    run_command(f'git commit -m "{commit_msg}"')

    # Create tag message
    tag_message = create_tag_message(str(new_version), changes)

    # Create an annotated tag
    tag_name = f"v{new_version}"
    tag_cmd = f'git tag -a {tag_name} -m "{tag_message}"'
    run_command(tag_cmd)

    # Ask to push
    if confirm("Push the new release to the remote repository?"):
        run_command("git push")
        run_command(f"git push origin {tag_name}")
        print(f"\nSuccessfully released version {new_version}")
    else:
        print(f"\nVersion {new_version} prepared locally.")
        print(f"To push later, run: git push && git push origin {tag_name}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="FinGPT AI Day Trading System release tool")
    parser.add_argument('increment', choices=['major', 'minor', 'patch'],
                        help="Which version number to increment")

    args = parser.parse_args()
    create_release(args.increment)


if __name__ == "__main__":
    main()
