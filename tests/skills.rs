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
