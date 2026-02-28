# Claude Code Instructions

## Git Commits

- Do NOT add `Co-Authored-By` trailer to commit messages
- Do NOT add Claude as a co-author in any form

## Git Workflow

- NEVER commit directly to `main`
- NEVER push to `main` — the ONLY allowed operation on `main` is `git pull`
- ALWAYS create a separate feature branch for any changes
- Feature branches MUST be pushed to remote immediately after committing — do NOT wait for user to ask
- Branch naming: `feature/<short-description>` or `fix/<short-description>`
- After pushing a feature branch, ALWAYS provide the PR creation link in chat
- Format: `https://github.com/<owner>/<repo>/pull/new/<branch-name>`
- Extract owner/repo from the git remote URL automatically
