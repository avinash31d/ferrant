use serde::{Deserialize, Serialize};

/// The role of a message participant in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single request made by the model to invoke a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Provider-neutral text, image, audio, or file content carried by a message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    ImageData {
        data: String,
        media_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    Audio {
        data: String,
        format: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        transcript: Option<String>,
    },
    File {
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            url: url.into(),
            detail: None,
        }
    }

    pub fn image_data(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::ImageData {
            data: data.into(),
            media_type: media_type.into(),
            detail: None,
        }
    }

    pub fn audio(data: impl Into<String>, format: impl Into<String>) -> Self {
        Self::Audio {
            data: data.into(),
            format: format.into(),
            transcript: None,
        }
    }
}

/// A single message in the conversation history that is fed to the model
/// and produced by it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub content_parts: Vec<ContentPart>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<ToolCall>,
    /// Present only on `Role::Tool` messages: the id of the call being answered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Present only on `Role::Tool` messages: the tool name (helps providers/debugging).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            content_parts: vec![],
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            content_parts: vec![],
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }

    /// Build a user message containing any mixture of text, images, audio,
    /// and files.
    pub fn user_parts(content_parts: Vec<ContentPart>) -> Self {
        Self {
            role: Role::User,
            content: None,
            content_parts,
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            content_parts: vec![],
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant_parts(content_parts: Vec<ContentPart>) -> Self {
        let content = content_parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        Self {
            role: Role::Assistant,
            content: (!content.is_empty()).then_some(content),
            content_parts,
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            content_parts: vec![],
            tool_calls,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            content_parts: vec![],
            tool_calls: vec![],
            tool_call_id: Some(tool_call_id.into()),
            name: Some(name.into()),
        }
    }
}
