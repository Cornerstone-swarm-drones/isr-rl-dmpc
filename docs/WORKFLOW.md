# Git Workflow Guide

## Before You Start Coding

1. **Update local repository:**
   ```bash
   git pull origin develop
   ```

2. Create feature branch from develop:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

3. Push branch to GitHub:

```bash
git push -u origin feature/your-feature-name
```

## While Coding
1. Make small, logical commits:

```bash
git add src/module/file.py
git commit -m "feat: Add function description

- Detailed explanation of change
- Include reasoning if non-obvious"
```

2. Commit message format:

```text
<type>: <subject>

<body>

Types: feat, fix, docs, style, refactor, test, chore
```

3. Push regularly:

```bash
git push origin feature/your-feature-name
```

## When Ready to Merge
1. Pull latest develop:

```bash
git pull origin develop
```

2. Resolve any conflicts locally

3. Run all tests locally:

```bash
pytest tests/
```

4. Create Pull Request on GitHub:

- Go to your branch on GitHub

- Click "Compare & pull request"

- Add description:

```text
## Description
What does this PR do?

## Related Issue
Closes #123

## Type of Change
- [ ] Feature
- [ ] Bug fix
- [ ] Documentation

## Testing
How was this tested?
```
- Request code review from team members

5. Address review comments

6. Team Lead merges after approval:

```bash
git checkout develop
git pull origin develop
git merge feature/your-feature-name
git push origin develop
```
7. Delete feature branch:

```bash
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

## Syncing Between Team Members
Every morning, update your branch:

```bash
git checkout develop
git pull origin develop
git checkout feature/your-feature-name
git merge develop
```

This pulls in any changes from teammates overnight.
