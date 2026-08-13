<!-- Release PR: staging -> main. Per the Hyde SDLC standard (OPS-230), complete
every item before merge. Open with:
https://github.com/Hyde-Inc/autoresearch/compare/main...staging?template=release.md -->

## Release vX.Y.Z

### Sign-off checklist

- [ ] Staging verified for these commits: installed with
      `uv tool install --force git+https://github.com/Hyde-Inc/autoresearch@staging`
      at commit `<sha>`.
- [ ] Smoke checks passed: `autoresearch doctor`, one demo run reaches a scored
      experiment, no new errors.
- [ ] Config or migration notes (breaking task-config/CLI changes), in order:
- [ ] Rollback plan: previous tag `vX.Y.(Z-1)` -
      `uv tool install --force git+https://github.com/Hyde-Inc/autoresearch@<previous-tag>`.
- [ ] Linear tickets in this release:
- [ ] Owner available after the release: @

### After merge

- [ ] Tag: `git tag vX.Y.Z <merge-sha> && git push origin vX.Y.Z`
- [ ] GitHub Release created with short notes and Linear links.
