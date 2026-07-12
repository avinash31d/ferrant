# Skills Design

## Goal

Add Codex/Claude-compatible skills to Ferrant. An agent discovers skill packages, sees compact metadata for each package, loads a relevant `SKILL.md` on demand, and can read the package's supporting text resources. Skills never execute bundled code.

## Package format

A skill is a directory containing `SKILL.md`. The file begins with YAML frontmatter containing at least `name` and `description`, followed by Markdown instructions. Unknown frontmatter fields are preserved so Ferrant remains compatible with extensions to the Codex and Claude formats.

Supporting files may live anywhere below the skill directory. Ferrant exposes them as inert bytes or UTF-8 text through an explicit resource-read operation. It never interprets a resource as executable code.

## Sources and discovery

`SkillSource` supports:

- a local directory, recursively searched for `SKILL.md` files;
- a GitHub repository URL with an optional Git reference and subdirectory.

GitHub repositories are cloned into a caller-configurable cache. A stable source identity determines the cache entry, and a refresh operation updates an existing clone. Discovery happens only inside the configured local root or repository subdirectory.

Malformed skills, duplicate names, inaccessible sources, and invalid Git references produce explicit errors rather than being silently skipped.

## Core API

The `skills` module provides:

- `SkillMetadata`: required name and description plus preserved extension fields;
- `Skill`: metadata, instructions, package root, and source identity;
- `SkillSource`: local and GitHub source configuration;
- `SkillCatalog`: source loading, discovery, name lookup, summaries, and bounded resource reads;
- typed skill errors for parsing, source, duplicate-name, size-limit, and path-boundary failures.

`AgentBuilder::skills(catalog)` attaches the catalog to an agent. Building the agent adds compact skill name/description summaries to its existing system instructions and registers internal skill tools.

## Runtime behavior

The model initially receives only the catalog summaries and guidance to load a skill when its description matches the user's task. This avoids placing every skill's full instructions in the context.

The internal `load_skill` tool accepts a catalog name and returns that skill's full Markdown instructions. The internal `read_skill_resource` tool accepts a skill name and relative resource path and returns bounded text content. Tool results enter the normal Ferrant reasoning loop, so provider implementations require no changes.

If the agent has no existing instructions, Ferrant creates a system message containing skill guidance. If instructions already exist, it appends a clearly delimited skills section without otherwise changing them.

## Security boundaries

Resource paths must be relative, canonicalize beneath the selected skill root, and remain there after symlink resolution. Absolute paths, parent traversal, and symlink escape are rejected. Configurable per-file limits bound instruction and resource size.

Ferrant does not run scripts, invoke shells, install dependencies declared by a skill, or grant additional tools. Script files can only be returned as inert text when explicitly requested and within the configured size limit.

GitHub fetching uses the local `git` executable without embedding credentials. Authentication and repository access remain the host application's responsibility. Repository URLs and refs are passed as process arguments rather than shell-interpolated strings.

## Errors

Source and parse failures identify the affected source or path. Duplicate skill names report both packages. Tool-facing errors are concise and do not expose content outside the skill root. A failed skill load prevents construction of that catalog; it does not produce a partially populated catalog.

## Testing

Tests cover YAML parsing, preservation of extension metadata, recursive local discovery, duplicate names, malformed files, prompt summary generation, on-demand instruction loading, resource reads, file-size limits, traversal attempts, and symlink escape where supported.

GitHub behavior is tested offline with temporary local Git repositories exercising clone, ref selection, subdirectory discovery, cache reuse, and refresh. Agent integration tests use a deterministic fake model to verify tool registration and progressive disclosure without provider calls.

## Documentation

The README will include local-directory and GitHub examples and state the no-execution security model. A runnable example will demonstrate catalog construction and attachment through `AgentBuilder::skills`.

## Delivery

Implementation is made on the `skills` branch. The full Rust test suite and formatting checks must pass before the branch is pushed and a draft pull request is opened against the repository's default branch.
