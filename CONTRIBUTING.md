# Contributing to the FinGPT AI Day Trading System

Thank you for your interest in contributing to the FinGPT AI Day Trading System. This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. **Fork the Repository**
   - Fork the repository on GitHub to your own account
   - Clone your fork locally: `git clone https://github.com/YOUR-USERNAME/nextgen.git`
   - Add the original repository as an upstream remote: `git remote add upstream https://github.com/slusheeking/nextgen.git`

2. **Set Up Development Environment**
   - Install all dependencies: `pip install -r requirements.txt`
   - Copy `.env.example` to `.env` and configure your environment variables

## Making Changes

1. **Create a Branch**
   - Create a new branch for your changes: `git checkout -b feature/your-feature-name`
   - Keep your changes focused on a single feature or bug fix

2. **Code Standards**
   - Follow PEP 8 style guidelines
   - Write docstrings for all functions, classes, and modules
   - Include type hints where appropriate
   - Add unit tests for new functionality

3. **Commit Changes**
   - Use meaningful commit messages
   - Keep commits atomic and focused
   - Reference issue numbers in commit messages when applicable

## Using the Automated Git Script

We've included a helpful script `push_git.py` to simplify Git operations:

```bash
# Run the script directly
./push_git.py
```

This script will:
1. Check git status
2. Add all changes
3. Commit with an automatically generated timestamp message
4. Push to the remote repository

You can use this for quick updates, but for significant contributions, please follow the standard pull request process.

## Pull Request Process

1. **Update Your Fork**
   - Before submitting a PR, update your fork: `git fetch upstream && git rebase upstream/main`
   - Resolve any conflicts

2. **Create a Pull Request**
   - Push your branch to your fork: `git push origin feature/your-feature-name`
   - Create a pull request from your branch to the main repository
   - Provide a clear description of the changes and any relevant issue numbers

3. **Code Review**
   - Address any feedback from reviewers
   - Make requested changes and push to your branch
   - The PR will be updated automatically

4. **Merge**
   - Once approved, your PR will be merged into the main branch
   - Delete your branch after it's been merged

## Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Include detailed steps to reproduce any bugs
- Include environment details (Python version, OS, etc.)

## Project Structure

Please maintain the existing project structure:

- Core FinGPT components should be placed in the appropriate subdirectory under `/fingpt/`
- Infrastructure components should have their own top-level directory
- Keep related files together in the same module

Thank you for contributing to the FinGPT AI Day Trading System!
