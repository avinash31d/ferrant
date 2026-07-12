use ferrant::{SkillCatalog, SkillError, SkillLimits, SkillSource};
use std::fs;
use std::path::{Path, PathBuf};
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
