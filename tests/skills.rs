use ferrant::{SkillCatalog, SkillError, SkillLimits, SkillSource};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("ferrant-skills-{}-{id}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_skill(root: &Path, relative: &str, contents: &str) {
    let directory = root.join(relative);
    fs::create_dir_all(&directory).unwrap();
    fs::write(directory.join("SKILL.md"), contents).unwrap();
}

fn git(cwd: &Path, args: &[&str]) {
    let output = Command::new("git")
        .current_dir(cwd)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn git_stdout(cwd: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .current_dir(cwd)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

struct GitFixture {
    _temp: TempDir,
    work: PathBuf,
    url: String,
    cache: PathBuf,
    bare: PathBuf,
}

impl GitFixture {
    fn new() -> Self {
        let temp = TempDir::new();
        let work = temp.path().join("work");
        let bare = temp.path().join("origin.git");
        fs::create_dir_all(&work).unwrap();
        git(&work, &["init", "-b", "main"]);
        git(&work, &["config", "user.email", "tests@example.invalid"]);
        git(&work, &["config", "user.name", "Ferrant Tests"]);
        write_skill(
            &work,
            "skills/main",
            "---\nname: main-skill\ndescription: Main\n---\nv1\n",
        );
        git(&work, &["add", "."]);
        git(&work, &["commit", "-m", "main skill"]);
        git(&work, &["tag", "v1"]);
        git(&work, &["checkout", "-b", "alternate"]);
        write_skill(
            &work,
            "skills/alternate",
            "---\nname: alternate-skill\ndescription: Alternate\n---\nbranch\n",
        );
        git(&work, &["add", "."]);
        git(&work, &["commit", "-m", "alternate skill"]);
        git(&work, &["checkout", "main"]);
        let bare_text = bare.to_string_lossy().into_owned();
        git(
            temp.path(),
            &[
                "clone",
                "--bare",
                work.to_string_lossy().as_ref(),
                &bare_text,
            ],
        );
        git(&work, &["remote", "add", "origin", &bare_text]);
        let url = format!(
            "file:///{}",
            bare.to_string_lossy()
                .replace('\\', "/")
                .trim_start_matches('/')
        );
        let cache = temp.path().join("cache");
        Self {
            _temp: temp,
            work,
            url,
            cache,
            bare,
        }
    }

    fn source(&self, git_ref: Option<&str>, subdirectory: Option<&str>) -> SkillSource {
        SkillSource::GitHub {
            repository: self.url.clone(),
            git_ref: git_ref.map(str::to_owned),
            subdirectory: subdirectory.map(PathBuf::from),
            cache_dir: self.cache.clone(),
        }
    }

    fn update_main(&self) {
        fs::write(
            self.work.join("skills/main/SKILL.md"),
            "---\nname: main-skill\ndescription: Main\n---\nv2\n",
        )
        .unwrap();
        git(&self.work, &["add", "."]);
        git(&self.work, &["commit", "-m", "update"]);
        git(&self.work, &["push", "origin", "main"]);
    }
}

#[test]
fn github_selects_tag_commit_sha_and_qualified_ref() {
    let fixture = GitFixture::new();
    let sha = git_stdout(&fixture.work, &["rev-parse", "main"]);
    for git_ref in ["v1", sha.as_str(), "refs/heads/main", "refs/tags/v1"] {
        let catalog = SkillCatalog::load(
            vec![fixture.source(Some(git_ref), Some("skills/main"))],
            SkillLimits::default(),
        )
        .unwrap();
        assert_eq!(
            catalog.skill("main-skill").unwrap().instructions,
            "v1\n",
            "ref {git_ref}"
        );
    }
}

#[test]
fn github_failed_initial_materialization_leaves_no_cache_entry() {
    let fixture = GitFixture::new();
    assert!(SkillCatalog::load(
        vec![fixture.source(Some("does-not-exist"), None)],
        SkillLimits::default(),
    )
    .is_err());
    assert!(fs::read_dir(&fixture.cache).unwrap().next().is_none());
}

#[test]
fn github_failed_refresh_preserves_valid_cached_checkout() {
    let fixture = GitFixture::new();
    let source = fixture.source(Some("main"), Some("skills/main"));
    SkillCatalog::load(vec![source.clone()], SkillLimits::default()).unwrap();
    let unavailable = fixture.bare.with_extension("unavailable");
    fs::rename(&fixture.bare, &unavailable).unwrap();
    assert!(SkillCatalog::refresh(vec![source.clone()], SkillLimits::default()).is_err());
    let cached = SkillCatalog::load(vec![source], SkillLimits::default()).unwrap();
    assert_eq!(cached.skill("main-skill").unwrap().instructions, "v1\n");
}

#[test]
fn github_clones_selects_ref_and_scopes_subdirectory() {
    let fixture = GitFixture::new();
    let main = SkillCatalog::load(
        vec![fixture.source(Some("main"), Some("skills/main"))],
        SkillLimits::default(),
    )
    .unwrap();
    assert_eq!(main.skill("main-skill").unwrap().instructions, "v1\n");
    assert!(main.skill("alternate-skill").is_none());
    let alternate = SkillCatalog::load(
        vec![fixture.source(Some("alternate"), Some("skills/alternate"))],
        SkillLimits::default(),
    )
    .unwrap();
    assert_eq!(
        alternate.skill("alternate-skill").unwrap().instructions,
        "branch\n"
    );
}

#[test]
fn github_load_reuses_cache_and_refresh_updates_it() {
    let fixture = GitFixture::new();
    let source = fixture.source(Some("main"), Some("skills/main"));
    assert_eq!(
        SkillCatalog::load(vec![source.clone()], SkillLimits::default())
            .unwrap()
            .skill("main-skill")
            .unwrap()
            .instructions,
        "v1\n"
    );
    fixture.update_main();
    assert_eq!(
        SkillCatalog::load(vec![source.clone()], SkillLimits::default())
            .unwrap()
            .skill("main-skill")
            .unwrap()
            .instructions,
        "v1\n"
    );
    assert_eq!(
        SkillCatalog::refresh(vec![source], SkillLimits::default())
            .unwrap()
            .skill("main-skill")
            .unwrap()
            .instructions,
        "v2\n"
    );
}

#[test]
fn github_rejects_invalid_ref() {
    let fixture = GitFixture::new();
    assert!(SkillCatalog::load(
        vec![fixture.source(Some("missing"), None)],
        SkillLimits::default()
    )
    .is_err());
}

#[test]
fn github_rejects_missing_subdirectory() {
    let fixture = GitFixture::new();
    assert!(SkillCatalog::load(
        vec![fixture.source(Some("main"), Some("absent"))],
        SkillLimits::default()
    )
    .is_err());
}

fn load(root: &Path, max_instruction_bytes: usize) -> Result<SkillCatalog, SkillError> {
    SkillCatalog::load(
        vec![SkillSource::Local {
            root: root.to_path_buf(),
        }],
        SkillLimits {
            max_instruction_bytes,
            ..SkillLimits::default()
        },
    )
}

fn load_with_resource_limit(root: &Path, max_resource_bytes: usize) -> SkillCatalog {
    SkillCatalog::load(
        vec![SkillSource::Local {
            root: root.to_path_buf(),
        }],
        SkillLimits {
            max_resource_bytes,
            ..SkillLimits::default()
        },
    )
    .unwrap()
}

fn resource_catalog(limit: usize) -> (TempDir, SkillCatalog) {
    let temp = TempDir::new();
    write_skill(
        temp.path(),
        "package",
        "---\nname: reader\ndescription: Read resources\n---\nRead.\n",
    );
    let catalog = load_with_resource_limit(temp.path(), limit);
    (temp, catalog)
}

#[test]
fn resource_reads_nested_utf8_file() {
    let (temp, catalog) = resource_catalog(1024);
    let nested = temp.path().join("package/assets");
    fs::create_dir_all(&nested).unwrap();
    fs::write(nested.join("guide.txt"), "héllo").unwrap();
    assert_eq!(
        catalog
            .read_resource("reader", Path::new("assets/guide.txt"))
            .unwrap(),
        "héllo"
    );
}

#[test]
fn resource_rejects_unknown_skill() {
    let (_temp, catalog) = resource_catalog(1024);
    assert!(matches!(
        catalog.read_resource("missing", Path::new("x")),
        Err(SkillError::UnknownSkill { .. })
    ));
}

#[test]
fn resource_reports_missing_file() {
    let (_temp, catalog) = resource_catalog(1024);
    assert!(matches!(
        catalog.read_resource("reader", Path::new("missing.txt")),
        Err(SkillError::Io { .. })
    ));
}

#[test]
fn resource_rejects_absolute_path() {
    let (temp, catalog) = resource_catalog(1024);
    assert!(matches!(
        catalog.read_resource("reader", &temp.path().join("outside")),
        Err(SkillError::InvalidResourcePath { .. })
    ));
}

#[test]
fn resource_rejects_parent_traversal() {
    let (_temp, catalog) = resource_catalog(1024);
    assert!(matches!(
        catalog.read_resource("reader", Path::new("../outside")),
        Err(SkillError::InvalidResourcePath { .. })
    ));
}

#[test]
fn resource_rejects_non_utf8_content() {
    let (temp, catalog) = resource_catalog(1024);
    fs::write(temp.path().join("package/binary"), [0xff]).unwrap();
    assert!(matches!(
        catalog.read_resource("reader", Path::new("binary")),
        Err(SkillError::ResourceNotUtf8 { .. })
    ));
}

#[test]
fn resource_enforces_byte_limit() {
    let (temp, catalog) = resource_catalog(4);
    fs::write(temp.path().join("package/large"), b"12345").unwrap();
    assert!(matches!(
        catalog.read_resource("reader", Path::new("large")),
        Err(SkillError::ResourceTooLarge { limit: 4, .. })
    ));
}

#[cfg(unix)]
#[test]
fn resource_rejects_symlink_escaping_package_root() {
    use std::os::unix::fs::symlink;
    let (temp, catalog) = resource_catalog(1024);
    let outside = temp.path().join("outside");
    fs::write(&outside, "secret").unwrap();
    symlink(&outside, temp.path().join("package/link")).unwrap();
    assert!(matches!(
        catalog.read_resource("reader", Path::new("link")),
        Err(SkillError::ResourceOutsideRoot { .. })
    ));
}

#[cfg(windows)]
#[test]
fn resource_rejects_symlink_escaping_package_root_when_permitted() {
    use std::os::windows::fs::symlink_file;
    let (temp, catalog) = resource_catalog(1024);
    let outside = temp.path().join("outside");
    fs::write(&outside, "secret").unwrap();
    if let Err(error) = symlink_file(&outside, temp.path().join("package/link")) {
        if error.kind() == std::io::ErrorKind::PermissionDenied
            || error.raw_os_error() == Some(1314)
        {
            return;
        }
        panic!("failed to create test symlink: {error}");
    }
    assert!(matches!(
        catalog.read_resource("reader", Path::new("link")),
        Err(SkillError::ResourceOutsideRoot { .. })
    ));
}

#[test]
fn local_discovers_recursively_and_extracts_body() {
    let temp = TempDir::new();
    write_skill(
        temp.path(),
        "nested/package",
        "---\nname: deploy\ndescription: Deploy safely\n---\n# Instructions\n\nShip it.\n",
    );

    let catalog = load(temp.path(), 1024).unwrap();
    let skill = catalog.skill("deploy").unwrap();
    assert_eq!(skill.instructions, "# Instructions\n\nShip it.\n");
    assert_eq!(
        skill.root,
        temp.path().join("nested/package").canonicalize().unwrap()
    );
}

#[test]
fn local_preserves_extension_metadata() {
    let temp = TempDir::new();
    write_skill(temp.path(), "one", "---\nname: review\ndescription: Review code\nlicense: MIT\nmetadata:\n  audience: rust\n---\nDo it.\n");
    let catalog = load(temp.path(), 1024).unwrap();
    let extensions = &catalog.skill("review").unwrap().metadata.extensions;
    assert_eq!(extensions["license"].as_str(), Some("MIT"));
    assert_eq!(extensions["metadata"]["audience"].as_str(), Some("rust"));
}

#[test]
fn local_rejects_duplicate_names_deterministically() {
    let temp = TempDir::new();
    write_skill(
        temp.path(),
        "b",
        "---\nname: same\ndescription: B\n---\nB\n",
    );
    write_skill(
        temp.path(),
        "a",
        "---\nname: same\ndescription: A\n---\nA\n",
    );
    let error = load(temp.path(), 1024).unwrap_err();
    match error {
        SkillError::DuplicateName {
            name,
            first,
            second,
        } => {
            assert_eq!(name, "same");
            assert!(first.ends_with("a"));
            assert!(second.ends_with("b"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn local_reports_malformed_frontmatter() {
    let temp = TempDir::new();
    write_skill(
        temp.path(),
        "bad",
        "---\nname: [broken\ndescription: nope\n---\nBody\n",
    );
    assert!(matches!(
        load(temp.path(), 1024),
        Err(SkillError::Parse { .. })
    ));
}

#[test]
fn local_requires_nonempty_name_and_description() {
    let temp = TempDir::new();
    write_skill(
        temp.path(),
        "bad",
        "---\nname: '  '\ndescription: present\n---\nBody\n",
    );
    assert!(matches!(
        load(temp.path(), 1024),
        Err(SkillError::Parse { .. })
    ));
}

#[test]
fn local_enforces_instruction_byte_limit() {
    let temp = TempDir::new();
    write_skill(
        temp.path(),
        "large",
        "---\nname: large\ndescription: Too large\n---\n0123456789\n",
    );
    assert!(matches!(
        load(temp.path(), 20),
        Err(SkillError::InstructionTooLarge { limit: 20, .. })
    ));
}
