# Architecture Review

Assessment PRs that touch `SKILL.md`, `skills/`, `frameworks/`, or
`.claude/skills/` should be
reviewed against `.claude/skills/torch-air-architecture-review/checklist.md`
before merge. It's a reference checklist (skill structure, framework
nesting, scoring consistency, dispatch logic) — not a form to fill in.

Run it with the `torch-air-architecture-review` skill:

```
/torch-air-architecture-review <pr-number-or-url-or-branch>
```

This checks the PR's diff against the checklist and writes up whatever's
actually wrong as a fresh, problems-only review, organized by category, with
a final Recommendation (**Approve** / **Request Changes** / **Needs
Discussion**). Categories with nothing wrong are omitted — a clean PR gets a
short review, not a wall of passing checkmarks. By default it's a dry run —
nothing is posted to GitHub. Add `--post` to have it post the review to the
PR (inline comments per finding, verdict in the review body) after you
confirm the rendered output:

```
/torch-air-architecture-review 16 --post
```

To review manually instead, read
`.claude/skills/torch-air-architecture-review/checklist.md` and write up
findings the same way. See
[issue #17](https://github.com/TorchedHat/torch-air/issues/17) for the
tracked follow-up (a GitHub Actions wrapper to trigger this automatically on
PR open/sync).
