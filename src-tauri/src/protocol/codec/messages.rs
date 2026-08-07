//! `messages_to_chat_v1` — Anthropic Messages → OpenAI Chat Completions.
//!
//! Covers request encoding, non-stream response decoding, and the streaming
//! (SSE) response decoding.  Extracts the strict conversion previously living
//! in `protocol::anthropic` without lowering its rejection policy: invalid tool
//! arguments fail, unknown roles/blocks are rejected, prompt-cache annotations
//! are stripped only when lossless, and usage is taken from real upstream
//! values.

use super::error::{FeatureKind, UnsupportedFeatures};
use super::registry::{NonStreamDecoder, StreamDecoder};
use super::report::{ConversionContext, Usage};
use super::request;
use super::sse;
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Encode an Anthropic Messages request into an OpenAI Chat Completions
/// request.  `model` is the mapped upstream model decided by the caller.
pub fn encode_messages_to_chat(
    body: &Value,
    model: &str,
) -> Result<(Value, ConversionContext), UnsupportedFeatures> {
    let mut out = Vec::new();
    // Fail-open drops/transforms recorded as JSON pointers for the report.
    let mut normalized: Vec<String> = Vec::new();

    // Top-level fields we can map 1:1 between Messages and Chat.  `top_k` is
    // deliberately absent: OpenAI Chat has no top_k, so it cannot be preserved
    // and must be rejected rather than silently dropped.
    const SUPPORTED_TOP_LEVEL: &[&str] = &[
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "stop_sequences",
        "stream",
        "stream_options",
        "tools",
        "tool_choice",
        "system",
        "user",
    ];

    // Native Anthropic features with no Chat equivalent are rejected here,
    // before any upstream access.  Anything not in the supported whitelist is
    // rejected with a concrete JSON pointer (never silently dropped), matching
    // the chat_to_messages_v1 top-level scan.  Two classes are exceptions,
    // both fail-open by decision (T13):
    //   - `thinking`/`output_config` are *mapped* to `reasoning_effort` below
    //     (CLIProxyAPI semantics); they are never rejected.
    //   - `container`/`context_management`/`context_management_config` have no
    //     Chat equivalent and are dropped, recorded on the report's
    //     `normalized` list rather than rejected.
    if let Some(obj) = body.as_object() {
        for (key, value) in obj.iter() {
            if !SUPPORTED_TOP_LEVEL.contains(&key.as_str()) {
                match key.as_str() {
                    "thinking" | "output_config" => {
                        // Mapped to `reasoning_effort` in the assembly section.
                        let _ = value;
                    }
                    "container" | "context_management" | "context_management_config" => {
                        normalized.push(format!("/{key}"));
                        let _ = value;
                    }
                    other => {
                        request::reject(
                            &mut out,
                            FeatureKind::UnsupportedField,
                            format!("/{other}"),
                            format!("Messages field {other:?} has no Chat Completions equivalent"),
                        );
                        let _ = value;
                    }
                }
            }
        }
    }

    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            let mut rejections = out.clone();
            request::reject(
                &mut rejections,
                FeatureKind::UnknownRole,
                "/messages",
                "Messages request requires a messages array",
            );
            UnsupportedFeatures::new(rejections)
        })?;

    // system -> single system message (order preserved, annotations stripped).
    let mut system_text: Option<String> = None;
    if let Some(sys) = body.get("system") {
        match request::anthropic_system_to_chat(sys, "/system", &mut normalized) {
            Ok(text) => {
                if !text.is_empty() {
                    system_text = Some(text);
                }
            }
            Err(e) => out.extend(e.fields),
        }
    }

    let mut chat_messages: Vec<Value> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        let mp = format!("/messages/{i}");
        match convert_anthropic_message_to_chat(msg, &mp, &mut normalized) {
            Ok(mut msgs) => chat_messages.append(&mut msgs),
            Err(e) => out.extend(e.fields),
        }
    }

    if !out.is_empty() {
        return Err(UnsupportedFeatures::new(out));
    }

    let mut chat = Map::new();
    chat.insert("model".to_string(), Value::String(model.to_string()));
    if let Some(sys) = system_text {
        chat.insert(
            "messages".to_string(),
            Value::Array(
                std::iter::once(serde_json::json!({"role": "system", "content": sys}))
                    .chain(chat_messages.into_iter())
                    .collect(),
            ),
        );
    } else {
        chat.insert("messages".to_string(), Value::Array(chat_messages));
    }
    chat.insert(
        "max_tokens".to_string(),
        body.get("max_tokens")
            .and_then(Value::as_u64)
            .map(Value::from)
            .unwrap_or(Value::from(4096u64)),
    );
    chat.insert("stream".to_string(), Value::Bool(stream));
    if let Some(t) = body.get("temperature") {
        chat.insert("temperature".to_string(), t.clone());
    }
    if let Some(t) = body.get("top_p") {
        chat.insert("top_p".to_string(), t.clone());
    }
    if let Some(stop) = body.get("stop_sequences") {
        match stop {
            Value::Array(_) => {
                chat.insert("stop".to_string(), stop.clone());
            }
            _ => {
                return Err(UnsupportedFeatures::single(
                    FeatureKind::UnsupportedField,
                    "/stop_sequences",
                    "stop_sequences must be an array of strings",
                ))
            }
        }
    }
    // tools
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let mut chat_tools = Vec::new();
        for (i, tool) in tools.iter().enumerate() {
            let tp = format!("/tools/{i}");
            match convert_anthropic_tool_to_chat(tool, &tp) {
                Ok(t) => chat_tools.push(t),
                Err(e) => out.extend(e.fields),
            }
        }
        if !out.is_empty() {
            return Err(UnsupportedFeatures::new(out));
        }
        if !chat_tools.is_empty() {
            chat.insert("tools".to_string(), Value::Array(chat_tools));
        }
    }
    // tool_choice
    if let Some(tc) = body.get("tool_choice") {
        let v = anthropic_tool_choice_to_chat(tc, "/tool_choice")?;
        chat.insert("tool_choice".to_string(), v);
    }
    if body
        .pointer("/tool_choice/disable_parallel_tool_use")
        .and_then(Value::as_bool)
        == Some(true)
    {
        chat.insert("parallel_tool_calls".to_string(), Value::Bool(false));
    }
    if stream {
        let mut options = body
            .get("stream_options")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        if !options.is_object() {
            return Err(UnsupportedFeatures::single(
                FeatureKind::UnsupportedField,
                "/stream_options",
                "stream_options must be an object",
            ));
        }
        options["include_usage"] = Value::Bool(true);
        chat.insert("stream_options".to_string(), options);
    }

    // thinking / output_config -> reasoning_effort (fail-open mapping, CPA
    // semantics).  Only present when the downstream asked for thinking; absent
    // thinking leaves `reasoning_effort` unset so the upstream applies its own
    // default.  The upstream (not us) adjudicates whether the model supports it.
    if let Some(effort) = anthropic_thinking_to_reasoning_effort(body) {
        chat.insert("reasoning_effort".to_string(), Value::String(effort));
    }

    let request_id = body
        .get("id")
        .and_then(Value::as_str)
        .map(String::from)
        .unwrap_or_else(|| format!("msg_{}", uuid::Uuid::new_v4().simple()));

    let mut context = ConversionContext::new(request_id, model.to_string(), stream);
    context.normalized = normalized;
    Ok((Value::Object(chat), context))
}

/// Map an Anthropic `thinking` config to an OpenAI `reasoning_effort` value,
/// following CLIProxyAPI's `ConvertClaudeRequestToOpenAI`.  Returns `None`
/// when the downstream did not ask for thinking (or asked for an unrecognized
/// type), in which case `reasoning_effort` is left unset.
fn anthropic_thinking_to_reasoning_effort(body: &Value) -> Option<String> {
    let thinking = body.get("thinking")?;
    if !thinking.is_object() {
        return None;
    }
    let ty = thinking.get("type").and_then(Value::as_str)?;
    match ty {
        "enabled" => {
            // budget_tokens present -> ConvertBudgetToLevel; absent -> auto.
            match thinking.get("budget_tokens").and_then(Value::as_i64) {
                Some(budget) => crate::protocol::thinking::budget_to_level(budget).map(String::from),
                None => Some("auto".to_string()),
            }
        }
        "adaptive" | "auto" => {
            // Explicit output_config.effort passes through (lowercased); else xhigh.
            match body
                .get("output_config")
                .and_then(|oc| oc.get("effort"))
                .and_then(Value::as_str)
            {
                Some(effort) if !effort.trim().is_empty() => {
                    Some(effort.trim().to_ascii_lowercase())
                }
                _ => Some("xhigh".to_string()),
            }
        }
        "disabled" => Some("none".to_string()),
        _ => None,
    }
}

/// Convert one Anthropic message (content array or string) into zero or more
/// Chat messages.  A `tool_result` user message is split into the preceding
/// text (as a user message) and a `role: tool` message; an assistant message
/// with tool_use blocks becomes one assistant message with `tool_calls`.
fn convert_anthropic_message_to_chat(
    msg: &Value,
    pointer: &str,
    normalized: &mut Vec<String>,
) -> Result<Vec<Value>, UnsupportedFeatures> {
    let role = msg.get("role").and_then(Value::as_str).ok_or_else(|| {
        UnsupportedFeatures::single(
            FeatureKind::UnknownRole,
            format!("{pointer}/role"),
            "Messages message missing role",
        )
    })?;
    match role {
        "user" | "assistant" | "system" => {}
        other => {
            return Err(UnsupportedFeatures::single(
                FeatureKind::UnknownRole,
                format!("{pointer}/role"),
                format!("unsupported Messages role {other:?}"),
            ))
        }
    }

    let content = msg.get("content");
    if role == "system" {
        let content = content.ok_or_else(|| {
            UnsupportedFeatures::single(
                FeatureKind::UnknownBlock,
                format!("{pointer}/content"),
                "system message missing content",
            )
        })?;
        let text =
            request::anthropic_system_to_chat(content, &format!("{pointer}/content"), normalized)?;
        return Ok(vec![serde_json::json!({"role": "system", "content": text})]);
    }

    let content_arr = content.and_then(Value::as_array);
    if let Some(items) = content_arr {
        let mut user_parts: Vec<Value> = Vec::new();
        let mut out = Vec::new();
        // Buffer the tool messages produced by tool_result blocks so they can be
        // inserted in order.  OpenAI requires a `role: tool` message to follow
        // the assistant tool_calls immediately; when a user message mixes
        // tool_result with preceding text (`[tool_result, text]` vs
        // `[text, tool_result]`), we must not let the text push a tool message
        // away from its assistant.  We therefore collect tool messages and emit
        // them ahead of any buffered text (option B in the review).
        let mut tool_messages: Vec<Value> = Vec::new();
        let flush_user = |parts: &mut Vec<Value>, out: &mut Vec<Value>| {
            if !parts.is_empty() {
                // Chat accepts a plain string when the user content is a single
                // text part; richer arrays are preserved as arrays.
                let content = if parts.len() == 1
                    && parts[0].get("type").and_then(Value::as_str) == Some("text")
                {
                    Value::String(
                        parts[0]
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    )
                } else {
                    Value::Array(std::mem::take(parts))
                };
                out.push(serde_json::json!({"role": "user", "content": content}));
            }
        };
        let mut assistant_text: Vec<String> = Vec::new();
        let mut assistant_reasoning = String::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        for (bi, block) in items.iter().enumerate() {
            let bp = format!("{pointer}/content/{bi}");
            let bt = block.get("type").and_then(Value::as_str).unwrap_or("");
            match bt {
                "text" => {
                    let t = block.get("text").and_then(Value::as_str).unwrap_or("");
                    match role {
                        "user" => {
                            if !t.is_empty() {
                                user_parts.push(serde_json::json!({"type": "text", "text": t}));
                            }
                        }
                        _ => {
                            if !t.is_empty() {
                                assistant_text.push(t.to_string());
                            }
                        }
                    }
                }
                "image" => {
                    if role != "user" {
                        return Err(UnsupportedFeatures::single(
                            FeatureKind::Media,
                            bp,
                            "assistant image blocks have no safe Chat representation",
                        ));
                    }
                    let img = request::anthropic_image_to_chat(block, &bp)?;
                    user_parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": img,
                    }));
                }
                "tool_use" => {
                    if role != "assistant" {
                        return Err(UnsupportedFeatures::single(
                            FeatureKind::UnknownBlock,
                            bp,
                            "tool_use blocks must be in an assistant message",
                        ));
                    }
                    let id = block
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            UnsupportedFeatures::single(
                                FeatureKind::MissingToolField,
                                format!("{bp}/id"),
                                "tool_use block missing id",
                            )
                        })?;
                    let name = block
                        .get("name")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            UnsupportedFeatures::single(
                                FeatureKind::MissingToolField,
                                format!("{bp}/name"),
                                "tool_use block missing name",
                            )
                        })?;
                    // A tool_use block without `input` is malformed: never
                    // fabricate `{}`.  `input: {}` is fine only when explicitly
                    // present.
                    let input = block.get("input").ok_or_else(|| {
                        UnsupportedFeatures::single(
                            FeatureKind::MissingToolField,
                            format!("{bp}/input"),
                            "tool_use block missing input",
                        )
                    })?;
                    if !input.is_object() {
                        return Err(UnsupportedFeatures::single(
                            FeatureKind::InvalidToolArguments,
                            format!("{bp}/input"),
                            "tool_use input must be a JSON object",
                        ));
                    }
                    let input = input.clone();
                    let arguments = serde_json::to_string(&input).map_err(|e| {
                        UnsupportedFeatures::single(
                            FeatureKind::InvalidToolArguments,
                            format!("{bp}/input"),
                            format!("tool_use input could not be serialized: {e}"),
                        )
                    })?;
                    tool_calls.push(serde_json::json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments}
                    }));
                }
                "tool_result" => {
                    if role != "user" {
                        return Err(UnsupportedFeatures::single(
                            FeatureKind::UnknownBlock,
                            bp,
                            "tool_result blocks must be in a user message",
                        ));
                    }
                    let tool_use_id = block
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| {
                            UnsupportedFeatures::single(
                                FeatureKind::MissingToolField,
                                format!("{bp}/tool_use_id"),
                                "tool_result missing tool_use_id",
                            )
                        })?;
                    let (text, is_error) = tool_result_to_chat_content(block, &bp)?;
                    let content = if is_error {
                        format!("Tool execution error:\n{text}")
                    } else {
                        text
                    };
                    tool_messages.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": content
                    }));
                }
                "thinking" => {
                    // Fail-open: assistant reasoning is carried into the Chat
                    // message as `reasoning_content` (OpenAI non-stream field);
                    // reasoning on any other role is dropped — we never inject
                    // thinking into a user/system channel.  `redacted_thinking`
                    // has no readable text and is ignored.
                    if role == "assistant" {
                        if let Some(t) = block.get("thinking").and_then(Value::as_str) {
                            if !t.is_empty() {
                                assistant_reasoning.push_str(t);
                            }
                        }
                        normalized.push(bp);
                    }
                }
                "redacted_thinking" => {
                    // No readable text; nothing to forward.
                    normalized.push(bp);
                }
                "cache_control" => {
                    return Err(UnsupportedFeatures::single(
                        FeatureKind::PromptCache,
                        bp,
                        "cache_control blocks have no Chat equivalent",
                    ))
                }
                other => {
                    return Err(UnsupportedFeatures::single(
                        FeatureKind::UnknownBlock,
                        bp,
                        format!("unsupported Messages content block type {other:?}"),
                    ))
                }
            }
        }
        if role == "assistant" {
            let content = if assistant_text.is_empty() {
                Value::Null
            } else {
                Value::String(assistant_text.join(""))
            };
            // Reasoning content extracted from assistant `thinking` blocks
            // (fail-open mapping to OpenAI's non-stream reasoning_content).
            let reasoning = if assistant_reasoning.is_empty() {
                None
            } else {
                Some(assistant_reasoning)
            };
            if tool_calls.is_empty() && content.is_null() && reasoning.is_none() {
                return Err(UnsupportedFeatures::single(
                    FeatureKind::UnknownBlock,
                    pointer,
                    "assistant message is empty",
                ));
            }
            let mut assistant = serde_json::json!({"role": "assistant", "content": content});
            if let Some(r) = reasoning {
                assistant["reasoning_content"] = Value::String(r);
            }
            if !tool_calls.is_empty() {
                assistant["tool_calls"] = Value::Array(tool_calls);
            }
            out.push(assistant);
        } else {
            // Emit tool messages ahead of the buffered text so they stay
            // adjacent to the assistant tool_calls message they answer.
            out.append(&mut tool_messages);
            flush_user(&mut user_parts, &mut out);
        }
        Ok(out)
    } else if let Some(s) = content.and_then(Value::as_str) {
        Ok(vec![serde_json::json!({"role": role, "content": s})])
    } else if content.map(|c| c.is_null()).unwrap_or(true) {
        // Anthropic allows null content on assistant messages carrying only
        // tool_use elsewhere; a null-content message with nothing else is
        // dropped.
        Ok(vec![])
    } else {
        Err(UnsupportedFeatures::single(
            FeatureKind::UnknownBlock,
            format!("{pointer}/content"),
            "Messages content must be a string or an array of blocks",
        ))
    }
}

/// Reduce a `tool_result` block to Chat text + error flag.
fn tool_result_to_chat_content(
    block: &Value,
    pointer: &str,
) -> Result<(String, bool), UnsupportedFeatures> {
    let is_error = block
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let content = block.get("content");
    match content {
        None | Some(Value::Null) => Ok((String::new(), is_error)),
        Some(Value::String(s)) => Ok((s.clone(), is_error)),
        Some(Value::Array(items)) => {
            let mut text = String::new();
            for (i, item) in items.iter().enumerate() {
                let ip = format!("{pointer}/content/{i}");
                match item.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        text.push_str(item.get("text").and_then(Value::as_str).unwrap_or(""))
                    }
                    Some("image") => {
                        return Err(UnsupportedFeatures::single(
                            FeatureKind::Media,
                            ip,
                            "tool_result images are not representable in Chat for this version",
                        ))
                    }
                    _ => {
                        return Err(UnsupportedFeatures::single(
                            FeatureKind::UnknownBlock,
                            ip,
                            "tool_result content block must be text",
                        ))
                    }
                }
            }
            Ok((text, is_error))
        }
        _ => Err(UnsupportedFeatures::single(
            FeatureKind::UnknownBlock,
            format!("{pointer}/content"),
            "tool_result content must be a string or text blocks",
        )),
    }
}

/// Convert an Anthropic tool to the Chat `tools` entry.
fn convert_anthropic_tool_to_chat(
    tool: &Value,
    pointer: &str,
) -> Result<Value, UnsupportedFeatures> {
    let ty = tool.get("type").and_then(Value::as_str).unwrap_or("custom");
    match ty {
        "custom" | "" => {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    UnsupportedFeatures::single(
                        FeatureKind::MissingToolField,
                        format!("{pointer}/name"),
                        "tool is missing name",
                    )
                })?;
            let description = tool
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("");
            let input_schema = tool.get("input_schema").ok_or_else(|| {
                UnsupportedFeatures::single(
                    FeatureKind::InvalidToolArguments,
                    format!("{pointer}/input_schema"),
                    format!("tool {name:?} is missing input_schema"),
                )
            })?;
            let parameters = request::anthropic_schema_to_chat_parameters(
                input_schema,
                &format!("{pointer}/input_schema"),
            )?;
            Ok(serde_json::json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            }))
        }
        _ => Err(UnsupportedFeatures::single(
            FeatureKind::BuiltinTool,
            format!("{pointer}/type"),
            format!("Anthropic built-in tool {ty:?} has no Chat equivalent"),
        )),
    }
}

/// Convert an Anthropic tool_choice to a Chat tool_choice.
fn anthropic_tool_choice_to_chat(tc: &Value, pointer: &str) -> Result<Value, UnsupportedFeatures> {
    if let Some(s) = tc.as_str() {
        // Anthropic accepts the bare strings auto/any; OpenAI only accepts
        // auto/none/required, so map (never pass through verbatim).
        return match s {
            "auto" => Ok(Value::String("auto".to_string())),
            "any" => Ok(Value::String("required".to_string())),
            // "tool" as a bare string carries no tool name; reject rather than
            // emit an empty-named Chat tool_choice.
            "tool" => Err(UnsupportedFeatures::single(
                FeatureKind::MissingToolField,
                format!("{pointer}/name"),
                "bare string tool_choice \"tool\" requires an explicit name (use the object form)",
            )),
            other => Err(UnsupportedFeatures::single(
                FeatureKind::UnsupportedField,
                pointer,
                format!("unsupported tool_choice string {other:?}"),
            )),
        };
    }
    let ty = tc.get("type").and_then(Value::as_str).ok_or_else(|| {
        UnsupportedFeatures::single(
            FeatureKind::UnsupportedField,
            pointer,
            "tool_choice must have a type",
        )
    })?;
    match ty {
        "auto" => Ok(Value::String("auto".to_string())),
        "any" => Ok(Value::String("required".to_string())),
        "tool" => {
            let name = tc
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    UnsupportedFeatures::single(
                        FeatureKind::MissingToolField,
                        format!("{pointer}/name"),
                        "tool_choice type=tool missing name",
                    )
                })?;
            Ok(serde_json::json!({
                "type": "function",
                "function": {"name": name}
            }))
        }
        other => Err(UnsupportedFeatures::single(
            FeatureKind::UnsupportedField,
            pointer,
            format!("unsupported tool_choice type {other:?}"),
        )),
    }
}

// ===========================================================================
// Non-stream response decoding: Messages JSON -> Chat Completions JSON.
// ===========================================================================

pub struct NonStreamResponseDecoder {
    context: ConversionContext,
}

impl NonStreamResponseDecoder {
    pub fn boxed(context: &ConversionContext) -> Box<dyn NonStreamDecoder + Send + Sync> {
        Box::new(NonStreamResponseDecoder {
            context: context.clone(),
        })
    }
}

impl NonStreamDecoder for NonStreamResponseDecoder {
    fn decode(&self, body: &Value) -> Result<Value, UnsupportedFeatures> {
        decode_messages_response_to_chat(body, &self.context)
    }
}

/// Decode a non-stream Anthropic Messages response into Chat Completions.
pub fn decode_messages_response_to_chat(
    body: &Value,
    context: &ConversionContext,
) -> Result<Value, UnsupportedFeatures> {
    if body.get("type").and_then(Value::as_str) != Some("message") {
        return Err(UnsupportedFeatures::single(
            FeatureKind::UnknownEvent,
            "/type",
            "Messages response must have type=message",
        ));
    }
    let content = body
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            UnsupportedFeatures::single(
                FeatureKind::UnknownEvent,
                "/content",
                "Messages response missing content array",
            )
        })?;

    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    for (i, block) in content.iter().enumerate() {
        let bp = format!("/content/{i}");
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                text.push_str(block.get("text").and_then(Value::as_str).unwrap_or(""));
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        UnsupportedFeatures::single(
                            FeatureKind::MissingToolField,
                            format!("{bp}/id"),
                            "tool_use block missing id",
                        )
                    })?;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        UnsupportedFeatures::single(
                            FeatureKind::MissingToolField,
                            format!("{bp}/name"),
                            "tool_use block missing name",
                        )
                    })?;
                // A tool_use without `input` is malformed: never fabricate `{}`.
                let input = block.get("input").ok_or_else(|| {
                    UnsupportedFeatures::single(
                        FeatureKind::MissingToolField,
                        format!("{bp}/input"),
                        "tool_use block missing input",
                    )
                })?;
                if !input.is_object() {
                    return Err(UnsupportedFeatures::single(
                        FeatureKind::InvalidToolArguments,
                        format!("{bp}/input"),
                        "tool_use input must be a JSON object",
                    ));
                }
                let input = input.clone();
                let arguments = serde_json::to_string(&input).map_err(|e| {
                    UnsupportedFeatures::single(
                        FeatureKind::InvalidToolArguments,
                        format!("{bp}/input"),
                        format!("tool_use input could not be serialized: {e}"),
                    )
                })?;
                tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments}
                }));
            }
            Some("thinking") => {
                // Fail-open: reasoning is surfaced as OpenAI `reasoning_content`
                // (string), never rejected.  Only the visible text is kept; the
                // signature/encrypted forms are dropped.
                if let Some(t) = block.get("thinking").and_then(Value::as_str) {
                    reasoning.push_str(t);
                }
            }
            Some("redacted_thinking") => {
                // No usable text; skip.
            }
            Some(other) => {
                return Err(UnsupportedFeatures::single(
                    FeatureKind::UnknownBlock,
                    format!("{bp}/type"),
                    format!("unsupported Messages response block type {other:?}"),
                ))
            }
            None => {
                return Err(UnsupportedFeatures::single(
                    FeatureKind::UnknownBlock,
                    format!("{bp}/type"),
                    "response content block missing type",
                ))
            }
        }
    }

    let stop_reason = body.get("stop_reason").and_then(Value::as_str);
    let finish_reason = match stop_reason {
        Some("end_turn") => "stop",
        Some("max_tokens") => "length",
        Some("tool_use") => "tool_calls",
        Some("refusal") | Some("refusal_message") => "content_filter",
        Some(other) => {
            return Err(UnsupportedFeatures::single(
                FeatureKind::UnknownFinishReason,
                "/stop_reason",
                format!("unknown Messages stop_reason {other:?}"),
            ))
        }
        None => {
            // Missing stop_reason: only safe when tool_use was emitted.
            if !tool_calls.is_empty() {
                "tool_calls"
            } else {
                return Err(UnsupportedFeatures::single(
                    FeatureKind::UnknownFinishReason,
                    "/stop_reason",
                    "Messages response missing stop_reason",
                ));
            }
        }
    };

    let usage = usage_from_messages(body);
    let response_id = body
        .get("id")
        .and_then(Value::as_str)
        .map(String::from)
        .unwrap_or_else(|| format!("chatcmpl_{}", uuid::Uuid::new_v4().simple()));

    let mut message = serde_json::json!({
        "role": "assistant",
        "content": if text.is_empty() { Value::Null } else { Value::String(text) },
    });
    if !reasoning.is_empty() {
        message["reasoning_content"] = Value::String(reasoning);
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }

    Ok(serde_json::json!({
        "id": response_id,
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": context.upstream_model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
            "prompt_tokens_details": {
                "cached_tokens": usage.cache_read_input_tokens
            },
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
        }
    }))
}

/// Extract real usage from a Messages response.  Cache tokens are surfaced in
/// OpenAI `usage` details without double-counting into input_tokens.
pub fn usage_from_messages(body: &Value) -> Usage {
    let input = body.pointer("/usage/input_tokens").and_then(Value::as_u64);
    let output = body.pointer("/usage/output_tokens").and_then(Value::as_u64);
    let cache_creation = body
        .pointer("/usage/cache_creation_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read = body
        .pointer("/usage/cache_read_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Usage {
        input_tokens: input.unwrap_or(0),
        output_tokens: output.unwrap_or(0),
        cache_creation_input_tokens: cache_creation,
        cache_read_input_tokens: cache_read,
        usage_unknown: input.is_none() || output.is_none(),
    }
}

// ===========================================================================
// Streaming: Messages SSE -> Chat SSE.
// ===========================================================================

#[derive(Default)]
struct MsgToolAccum {
    index: usize,
    id: String,
    name: String,
    arguments: String,
    started: bool,
    completed: bool,
}

/// Per-request state for the Messages SSE → Chat SSE decoder.
#[derive(Default)]
pub struct MessagesSseState {
    pending: Vec<u8>,
    started: bool,
    ended: bool,
    current_text_index: Option<usize>,
    text_content_index: Option<usize>,
    tools: BTreeMap<usize, MsgToolAccum>,
    next_tool_index: usize,
    stop_reason: Option<String>,
    usage: Usage,
    message_id: String,
    current_block: Option<String>,
    /// The mapped upstream model to emit in the synthesized Chat `role` frame.
    pub model: String,
}

impl MessagesSseState {
    /// Create the per-request state with the caller-provided model.
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            ..Default::default()
        }
    }

    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<String>, UnsupportedFeatures> {
        self.pending.extend_from_slice(bytes);
        let mut events = Vec::new();
        while let Some(end) = sse::record_end(&self.pending) {
            let record: Vec<u8> = self.pending.drain(..end).collect();
            let payload = sse::parse_data_payload(&record)?;
            if payload.is_empty() || payload == "[DONE]" {
                continue;
            }
            let json: Value = serde_json::from_str(&payload).map_err(|e| {
                UnsupportedFeatures::single(
                    FeatureKind::UnknownEvent,
                    "/",
                    format!("Anthropic upstream emitted invalid SSE JSON: {e}"),
                )
            })?;
            self.consume_json(json, &mut events)?;
        }
        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<String>, UnsupportedFeatures> {
        let mut events = Vec::new();
        if !self.pending.is_empty() {
            let record = std::mem::take(&mut self.pending);
            let payload = sse::parse_data_payload(&record)?;
            if !payload.is_empty() && payload != "[DONE]" {
                let json: Value = serde_json::from_str(&payload).map_err(|e| {
                    UnsupportedFeatures::single(
                        FeatureKind::UnknownEvent,
                        "/",
                        format!("Anthropic upstream emitted invalid SSE JSON: {e}"),
                    )
                })?;
                self.consume_json(json, &mut events)?;
            }
        }
        self.emit_final(&mut events)?;
        Ok(events)
    }

    pub fn usage(&self) -> Usage {
        self.usage
    }

    fn consume_json(
        &mut self,
        json: Value,
        events: &mut Vec<String>,
    ) -> Result<(), UnsupportedFeatures> {
        let ty = json.get("type").and_then(Value::as_str).ok_or_else(|| {
            UnsupportedFeatures::single(
                FeatureKind::UnknownEvent,
                "/type",
                "Anthropic SSE frame missing type",
            )
        })?;
        match ty {
            "message_start" => {
                if !self.started {
                    self.started = true;
                    if let Some(msg) = json.get("message") {
                        if let Some(id) = msg.get("id").and_then(Value::as_str) {
                            self.message_id = id.to_string();
                        }
                        if let Some(u) = msg.get("usage") {
                            self.update_usage(u);
                        }
                    }
                    // Emit the Chat `role` frame now.
                    events.push(sse::data_frame(serde_json::json!({
                        "id": self.message_id,
                        "object": "chat.completion.chunk",
                        "created": chrono::Utc::now().timestamp(),
                        "model": self.model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
                    })));
                }
                // a second message_start is ignored.
            }
            "content_block_start" => {
                let index = json.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let block = json.get("content_block").unwrap_or(&Value::Null);
                let bt = block.get("type").and_then(Value::as_str);
                self.current_block = Some(bt.unwrap_or("").to_string());
                match bt {
                    Some("text") => {
                        self.text_content_index = Some(index);
                    }
                    Some("thinking") | Some("redacted_thinking") => {
                        // Fail-open: reasoning is forwarded as OpenAI
                        // `reasoning_content` deltas.  `redacted_thinking` has
                        // no visible text (signature only) — its deltas are
                        // ignored but not rejected.  current_block is already
                        // recorded above, so text_delta-like handling below
                        // routes on the delta type, not the block type.
                    }
                    Some("tool_use") => {
                        // id and name are mandatory on a tool_use block; never
                        // emit an empty-id/empty-name tool call (R22).
                        let id = block
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                UnsupportedFeatures::single(
                                    FeatureKind::MissingToolField,
                                    "/content_block_start/content_block/id",
                                    "tool_use block missing id",
                                )
                            })?
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .ok_or_else(|| {
                                UnsupportedFeatures::single(
                                    FeatureKind::MissingToolField,
                                    "/content_block_start/content_block/name",
                                    "tool_use block missing name",
                                )
                            })?
                            .to_string();
                        let tool_index = self.next_tool_index;
                        self.next_tool_index += 1;
                        self.tools.insert(
                            index,
                            MsgToolAccum {
                                index: tool_index,
                                id: id.clone(),
                                name: name.clone(),
                                arguments: String::new(),
                                started: true,
                                completed: false,
                            },
                        );
                        // Emit the Chat tool_calls delta immediately (id + name +
                        // empty arguments) so consumers see the call id early.
                        events.push(sse::data_frame(serde_json::json!({
                            "choices": [{"index": 0, "delta": {"tool_calls": [{
                                "index": tool_index,
                                "id": id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""}
                            }]}, "finish_reason": null}]
                        })));
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let index = json.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let delta = json.get("delta").unwrap_or(&Value::Null);
                let dt = delta.get("type").and_then(Value::as_str);
                match dt {
                    Some("text_delta") => {
                        let text = delta.get("text").and_then(Value::as_str).unwrap_or("");
                        if !text.is_empty() {
                            events.push(sse::data_frame(serde_json::json!({
                                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                            })));
                        }
                    }
                    Some("input_json_delta") => {
                        let partial = delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        if let Some(tool) = self.tools.get_mut(&index) {
                            if !tool.completed {
                                tool.arguments.push_str(partial);
                            }
                        }
                    }
                    Some("thinking_delta") => {
                        let reasoning = delta
                            .get("thinking")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        if !reasoning.is_empty() {
                            events.push(sse::data_frame(serde_json::json!({
                                "choices": [{"index": 0, "delta": {"reasoning_content": reasoning}, "finish_reason": null}]
                            })));
                        }
                    }
                    Some("signature_delta") => {
                        // Encrypted/reference signature — no usable text for
                        // the Chat downstream; drop fail-open.
                    }
                    _ => {
                        return Err(UnsupportedFeatures::single(
                            FeatureKind::UnknownEvent,
                            "/delta/type",
                            format!("unknown content_block_delta type {dt:?}"),
                        ))
                    }
                }
            }
            "content_block_stop" => {
                let index = json.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                if let Some(tool) = self.tools.get_mut(&index) {
                    if !tool.completed {
                        // complete the tool call: emit the full accumulated
                        // arguments only when it is valid JSON (object).
                        let input: Value = serde_json::from_str(&tool.arguments).map_err(|e| {
                            UnsupportedFeatures::single(
                                FeatureKind::InvalidToolArguments,
                                "/content_block_delta/partial_json",
                                format!("tool arguments did not form valid JSON: {e}"),
                            )
                        })?;
                        if !input.is_object() {
                            return Err(UnsupportedFeatures::single(
                                FeatureKind::InvalidToolArguments,
                                "/content_block_delta/partial_json",
                                "tool arguments must decode to a JSON object",
                            ));
                        }
                        // Emit the remainder (if any) then nothing else needed:
                        // consumers already saw id/name; we must not re-send id.
                        // Send an arguments-only delta with full args.
                        events.push(sse::data_frame(serde_json::json!({
                            "choices": [{"index": 0, "delta": {"tool_calls": [{
                                "index": tool.index,
                                "function": {"arguments": tool.arguments}
                            }]}, "finish_reason": null}]
                        })));
                        tool.completed = true;
                    }
                }
            }
            "message_delta" => {
                if let Some(delta) = json.get("delta") {
                    if let Some(reason) = delta.get("stop_reason").and_then(Value::as_str) {
                        self.stop_reason = Some(reason.to_string());
                    }
                }
                if let Some(u) = json.get("usage") {
                    self.update_usage(u);
                }
            }
            "message_stop" => {
                // exactly-once termination handled by emit_final
            }
            "ping" => {}
            "error" => {
                return Err(UnsupportedFeatures::single(
                    FeatureKind::UnknownEvent,
                    "/type",
                    format!("Anthropic upstream error event: {}", json),
                ))
            }
            other => {
                return Err(UnsupportedFeatures::single(
                    FeatureKind::UnknownEvent,
                    "/type",
                    format!("unknown Anthropic SSE event type {other:?}"),
                ))
            }
        }
        Ok(())
    }

    fn update_usage(&mut self, u: &Value) {
        let input = u.get("input_tokens").and_then(Value::as_u64);
        let output = u.get("output_tokens").and_then(Value::as_u64);
        if let Some(i) = input {
            self.usage.input_tokens = i;
        }
        if let Some(o) = output {
            self.usage.output_tokens = o;
        }
        if let Some(c) = u.get("cache_creation_input_tokens").and_then(Value::as_u64) {
            self.usage.cache_creation_input_tokens = c;
        }
        if let Some(c) = u.get("cache_read_input_tokens").and_then(Value::as_u64) {
            self.usage.cache_read_input_tokens = c;
        }
        if input.is_none()
            && output.is_none()
            && self.usage.input_tokens == 0
            && self.usage.output_tokens == 0
        {
            self.usage.usage_unknown = true;
        }
    }

    fn emit_final(&mut self, events: &mut Vec<String>) -> Result<(), UnsupportedFeatures> {
        if self.ended {
            return Ok(());
        }
        if !self.started {
            // The upstream stream never delivered a message_start frame.  This
            // is a codec error (not an empty success) so the gateway can fail
            // over before committing the downstream response.
            self.ended = true;
            return Err(UnsupportedFeatures::single(
                FeatureKind::UnknownEvent,
                "/",
                "Anthropic upstream stream ended before any first frame (no message_start)",
            ));
        }
        // Validate any in-progress tool calls: Anthropic may send
        // content_block_stop for a tool_use with no deltas yet (empty object).
        for tool in self.tools.values() {
            if !tool.completed {
                // A tool_use that never saw content_block_stop is malformed.
                return Err(UnsupportedFeatures::single(
                    FeatureKind::MissingToolField,
                    "/content_block_start/content_block",
                    "Anthropic stream ended with an incomplete tool call",
                ));
            }
        }
        let finish_reason = match self.stop_reason.as_deref() {
            Some("end_turn") => "stop",
            Some("max_tokens") => "length",
            Some("tool_use") => "tool_calls",
            Some("refusal") | Some("refusal_message") => "content_filter",
            Some(other) => {
                return Err(UnsupportedFeatures::single(
                    FeatureKind::UnknownFinishReason,
                    "/message_delta/delta/stop_reason",
                    format!("unknown Messages stop_reason {other:?}"),
                ))
            }
            None => {
                return Err(UnsupportedFeatures::single(
                    FeatureKind::UnknownFinishReason,
                    "/message_delta/delta/stop_reason",
                    "Anthropic stream ended without a stop_reason",
                ))
            }
        };
        events.push(sse::data_frame(serde_json::json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
        })));
        events.push(sse::data_frame(serde_json::json!({
            "usage": {
                "prompt_tokens": self.usage.input_tokens,
                "completion_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.input_tokens + self.usage.output_tokens,
                "prompt_tokens_details": {"cached_tokens": self.usage.cache_read_input_tokens},
                "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": self.usage.cache_read_input_tokens,
            }
        })));
        events.push(sse::data_frame(Value::String("[DONE]".to_string())));
        self.ended = true;
        Ok(())
    }
}

pub struct MessagesStreamDecoder {
    state: MessagesSseState,
}

impl MessagesStreamDecoder {
    pub fn boxed(context: &ConversionContext) -> Box<dyn StreamDecoder + Send + Sync> {
        Box::new(MessagesStreamDecoder {
            state: MessagesSseState::new(&context.upstream_model),
        })
    }
}

impl super::registry::StreamDecoder for MessagesStreamDecoder {
    fn feed(&mut self, bytes: &[u8]) -> Result<Vec<String>, UnsupportedFeatures> {
        self.state.feed(bytes)
    }
    fn finish(&mut self) -> Result<Vec<String>, UnsupportedFeatures> {
        self.state.finish()
    }
    fn usage(&self) -> Option<Usage> {
        Some(self.state.usage)
    }
}
