use serde_yaml::Value;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct SkillMetadata {
    pub name: String,
    pub description: String,
    pub extensions: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Skill {
    pub metadata: SkillMetadata,
    pub instructions: String,
    pub root: PathBuf,
    pub source: SkillSource,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkillSource {
    Local { root: PathBuf },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkillLimits {
    pub max_instruction_bytes: usize,
    pub max_resource_bytes: usize,
}

impl Default for SkillLimits {
    fn default() -> Self {
        Self {
            max_instruction_bytes: 256 * 1024,
            max_resource_bytes: 1024 * 1024,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("skill source or file error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid skill at {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("skill instructions at {path} exceed {limit} bytes")]
    InstructionTooLarge { path: PathBuf, limit: usize },
    #[error("duplicate skill name '{name}' in {first} and {second}")]
    DuplicateName {
        name: String,
        first: PathBuf,
        second: PathBuf,
    },
    #[error("unknown skill '{name}'")]
    UnknownSkill { name: String },
    #[error("invalid resource path {path}")]
    InvalidResourcePath { path: PathBuf },
    #[error("resource {path} resolves outside skill root {root}")]
    ResourceOutsideRoot { path: PathBuf, root: PathBuf },
    #[error("resource at {path} is not a regular file")]
    ResourceNotFile { path: PathBuf },
    #[error("skill resource at {path} exceeds {limit} bytes")]
    ResourceTooLarge { path: PathBuf, limit: usize },
    #[error("skill resource at {path} is not UTF-8: {source}")]
    ResourceNotUtf8 {
        path: PathBuf,
        #[source]
        source: std::string::FromUtf8Error,
    },
}

#[derive(Clone, Debug, Default)]
pub struct SkillCatalog {
    skills: BTreeMap<String, Skill>,
    limits: SkillLimits,
}

impl SkillCatalog {
    pub fn load(sources: Vec<SkillSource>, limits: SkillLimits) -> Result<Self, SkillError> {
        let mut files = Vec::new();
        for source in &sources {
            match source {
                SkillSource::Local { root } => discover(root, source, &mut files)?,
            }
        }
        files.sort_by(|a, b| a.0.cmp(&b.0));

        let mut skills: BTreeMap<String, Skill> = BTreeMap::new();
        for (path, source) in files {
            let skill = parse_skill(&path, source, limits.max_instruction_bytes)?;
            if let Some(first) = skills.get(&skill.metadata.name) {
                return Err(SkillError::DuplicateName {
                    name: skill.metadata.name.clone(),
                    first: first.root.clone(),
                    second: skill.root.clone(),
                });
            }
            skills.insert(skill.metadata.name.clone(), skill);
        }
        Ok(Self { skills, limits })
    }

    pub fn skill(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    pub fn read_resource(&self, skill: &str, relative: &Path) -> Result<String, SkillError> {
        let skill = self
            .skills
            .get(skill)
            .ok_or_else(|| SkillError::UnknownSkill {
                name: skill.to_owned(),
            })?;
        if relative.as_os_str().is_empty()
            || relative.is_absolute()
            || relative
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err(SkillError::InvalidResourcePath {
                path: relative.to_path_buf(),
            });
        }

        let joined = skill.root.join(relative);
        let target = joined.canonicalize().map_err(|source| SkillError::Io {
            path: joined.clone(),
            source,
        })?;
        if !target.starts_with(&skill.root) {
            return Err(SkillError::ResourceOutsideRoot {
                path: target,
                root: skill.root.clone(),
            });
        }
        let metadata = fs::metadata(&target).map_err(|source| SkillError::Io {
            path: target.clone(),
            source,
        })?;
        if !metadata.is_file() {
            return Err(SkillError::ResourceNotFile { path: target });
        }
        let limit = self.limits.max_resource_bytes;
        if metadata.len() > limit as u64 {
            return Err(SkillError::ResourceTooLarge {
                path: target,
                limit,
            });
        }
        let mut bytes = Vec::new();
        File::open(&target)
            .map_err(|source| SkillError::Io {
                path: target.clone(),
                source,
            })?
            .take(limit.saturating_add(1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|source| SkillError::Io {
                path: target.clone(),
                source,
            })?;
        if bytes.len() > limit {
            return Err(SkillError::ResourceTooLarge {
                path: target,
                limit,
            });
        }
        String::from_utf8(bytes).map_err(|source| SkillError::ResourceNotUtf8 {
            path: target,
            source,
        })
    }
}

fn discover(
    root: &Path,
    source: &SkillSource,
    files: &mut Vec<(PathBuf, SkillSource)>,
) -> Result<(), SkillError> {
    let entries = fs::read_dir(root).map_err(|source_error| SkillError::Io {
        path: root.to_path_buf(),
        source: source_error,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source_error| SkillError::Io {
            path: root.to_path_buf(),
            source: source_error,
        })?;
        let file_type = entry.file_type().map_err(|source_error| SkillError::Io {
            path: entry.path(),
            source: source_error,
        })?;
        if file_type.is_dir() {
            discover(&entry.path(), source, files)?;
        } else if file_type.is_file() && entry.file_name() == "SKILL.md" {
            files.push((entry.path(), source.clone()));
        }
    }
    Ok(())
}

fn parse_skill(path: &Path, source: SkillSource, limit: usize) -> Result<Skill, SkillError> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|source_error| SkillError::Io {
            path: path.to_path_buf(),
            source: source_error,
        })?
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|source_error| SkillError::Io {
            path: path.to_path_buf(),
            source: source_error,
        })?;
    if bytes.len() > limit {
        return Err(SkillError::InstructionTooLarge {
            path: path.to_path_buf(),
            limit,
        });
    }
    let text = String::from_utf8(bytes).map_err(|error| SkillError::Parse {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let (yaml, instructions) = split_frontmatter(&text).ok_or_else(|| SkillError::Parse {
        path: path.to_path_buf(),
        message: "expected leading YAML frontmatter delimited by ---".into(),
    })?;
    let mut extensions: BTreeMap<String, Value> =
        serde_yaml::from_str(yaml).map_err(|error| SkillError::Parse {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let name = required_string(&mut extensions, "name", path)?;
    let description = required_string(&mut extensions, "description", path)?;
    let root = path
        .parent()
        .expect("SKILL.md has a parent")
        .canonicalize()
        .map_err(|source_error| SkillError::Io {
            path: path.to_path_buf(),
            source: source_error,
        })?;
    Ok(Skill {
        metadata: SkillMetadata {
            name,
            description,
            extensions,
        },
        instructions: instructions.to_owned(),
        root,
        source,
    })
}

fn split_frontmatter(text: &str) -> Option<(&str, &str)> {
    let rest = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))?;
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed == "---" {
            return Some((&rest[..offset], &rest[offset + line.len()..]));
        }
        offset += line.len();
    }
    None
}

fn required_string(
    map: &mut BTreeMap<String, Value>,
    key: &str,
    path: &Path,
) -> Result<String, SkillError> {
    match map.remove(key) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value),
        _ => Err(SkillError::Parse {
            path: path.to_path_buf(),
            message: format!("'{key}' must be a non-empty string"),
        }),
    }
}
