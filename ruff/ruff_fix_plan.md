# Ruff Issues Remediation Plan

## Objective

Systematically address the 315 Ruff linting issues to improve code quality, prevent potential runtime errors, and establish consistent style across the codebase.

## Implementation Strategy

### Phase 1: Critical Fixes (1-2 days)

**Target**: Address all high-priority issues that could cause runtime errors.

1. **Fix Undefined Names (F821)** - 3 issues
   - Investigate and fix the undefined reference to `updated_weights` in `nextgen_models/nextgen_decision/decision_model.py`
   - Address the other two undefined name errors
   - Add tests to verify corrections

2. **Apply Safe Automatic Fixes** - 58 issues
   ```bash
   cd /home/ubuntu/nextgen
   ruff check --fix .
   ```
   This will automatically fix:
   - F541: F-string missing placeholders (7 issues)
   - F811: Redefined while unused (1 issue)
   - Many F401: Unused imports

3. **Verify Fixes**
   - Run tests to ensure no regression
   - Commit changes with descriptive message

### Phase 2: Imports Restructuring (1 week)

**Target**: Address the most numerous issue category - imports not at the top of files.

1. **Create Import Organization Guidelines**
   - Document patterns for conditional imports
   - Define approach for circular import dependencies
   - Specify import order conventions

2. **Fix E402 Violations by Module Group**
   
   a. **Data MCP Modules**
   - Reorganize imports in data-related MCP modules
   - Use Ruff's `--select=E402` option to focus on these issues

   b. **Test Modules**
   - Restructure imports in test files
   - Lower priority since these don't impact production code

   c. **Core Modules**
   - Address imports in critical business logic modules

3. **Review and Test After Each Module Group**
   - Run tests after each set of changes
   - Document any import dependencies that can't be easily fixed

### Phase 3: Code Style Improvements (3-5 days)

**Target**: Address stylistic issues that impact readability.

1. **Fix Multiple Statements (E701)** - 43 issues
   - Convert single-line statements with colons to proper multi-line format
   - Example:
     ```python
     # From
     if condition: do_something()
     
     # To
     if condition:
         do_something()
     ```

2. **Apply Automated Unsafe Fixes for Unused Variables** - 30 issues
   - Review each instance of F841 (unused variables)
   - Consider whether the variable should be:
     - Used (logic error)
     - Removed (unnecessary)
     - Prefixed with `_` (intentionally unused)
   - Use `ruff --unsafe-fixes` for viable cases

3. **Address Other Minor Issues** - 4 issues
   - Fix E741 (ambiguous variable name)
   - Fix E722 (bare except)

### Phase 4: Workflow Integration (2-3 days)

**Target**: Prevent future linting issues by integrating Ruff into the development workflow.

1. **Create Ruff Configuration**
   - Create `pyproject.toml` with Ruff settings
   ```toml
   [tool.ruff]
   # Enable flake8-bugbear (`B`) rules.
   select = ["E", "F", "B"]
   ignore = []
   
   # Allow autofix for all enabled rules (when `--fix`) is provided.
   fixable = ["ALL"]
   unfixable = []
   
   # Exclude a variety of commonly ignored directories.
   exclude = [
       ".git",
       ".ruff_cache",
       "build",
       "dist",
   ]
   line-length = 100
   ```

2. **Set Up Pre-commit Hooks**
   - Create `.pre-commit-config.yaml`
   ```yaml
   repos:
   - repo: https://github.com/charliermarsh/ruff-pre-commit
     rev: v0.0.292
     hooks:
       - id: ruff
         args: [--fix]
   ```
   - Install and configure pre-commit

3. **Add Ruff to CI Pipeline**
   - Configure CI to run Ruff checks
   - Set appropriate error levels (warning vs. failure)

### Phase 5: Documentation and Training (1-2 days)

**Target**: Ensure team understanding and adoption of code quality standards.

1. **Create Style Guide**
   - Document Ruff rules being enforced
   - Provide examples of common patterns and fixes

2. **Team Knowledge Sharing**
   - Present findings and solutions
   - Demonstrate Ruff usage in development workflow

## Prioritized File List

Address files in order of issue density and importance to the application:

1. **mcp_tools/test_nextgen_models.py** (34 issues)
2. **mcp_tools/time_series_mcp/time_series_mcp.py** (27 issues)
3. **nextgen_models/nextgen_trader/trade_model.py** (24 issues)
4. **nextgen_models/nextgen_select/select_model.py** (22 issues)
5. **nextgen_models/nextgen_risk_assessment/risk_assessment_model.py** (20 issues)

## Testing Strategy

For each phase:

1. **Unit Tests**
   - Run existing tests after each set of changes
   - Add tests for any fixed logical issues

2. **Integration Tests**
   - Verify that modules work together after import restructuring

3. **Manual Testing**
   - Test critical workflows after all changes

## Metrics for Success

1. **Zero Critical Issues**
   - No undefined names (F821)
   - No bare except clauses (E722)

2. **Significant Reduction in Overall Issues**
   - At least 90% reduction in total issue count

3. **Consistent Code Style**
   - All new code follows the established style guide
   - Linting integrated into development workflow

4. **Developer Adoption**
   - Ruff integrated into local development environments
   - Team members familiar with standards

## Timeline

| Phase | Description | Timeline | Resources |
|-------|-------------|----------|-----------|
| 1 | Critical Fixes | 1-2 days | 1 developer |
| 2 | Imports Restructuring | 1 week | 1-2 developers |
| 3 | Code Style Improvements | 3-5 days | 1 developer |
| 4 | Workflow Integration | 2-3 days | 1 developer + DevOps |
| 5 | Documentation & Training | 1-2 days | 1 developer |

**Total Timeline**: Approximately 2-3 weeks

## Monitoring and Maintenance

1. **Regular Linting Reports**
   - Generate weekly reports on linting status
   - Track progress over time

2. **Rule Refinement**
   - Review and adjust Ruff rules based on team feedback
   - Consider adding more rules as code quality improves

3. **Continuous Improvement**
   - Regularly review new patterns of issues
   - Update style guide as needed

This plan provides a structured approach to addressing the 315 Ruff issues while prioritizing the most critical fixes and establishing long-term code quality practices.