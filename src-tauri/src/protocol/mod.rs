pub mod anthropic;
pub mod codec;
pub mod responses;
pub mod thinking;

use serde_json::Value;

/// Extract API key from either `Authorization: Bearer xxx` or `x-api-key: xxx` header.
pub fn extract_api_key(headers: &axum::http::HeaderMap) -> Option<String> {
    // Try Authorization: Bearer xxx first
    if let Some(auth) = headers.get("authorization").and_then(|h| h.to_str().ok()) {
        if let Some(key) = auth.strip_prefix("Bearer ") {
            let trimmed = key.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    // Fall back to x-api-key
    if let Some(key) = headers.get("x-api-key").and_then(|h| h.to_str().ok()) {
        let trimmed = key.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// Detect if a request is in Anthropic format by checking headers and body.
#[allow(dead_code)]
pub fn is_anthropic_request(headers: &axum::http::HeaderMap, body: &Value) -> bool {
    // Check for anthropic-version header
    if headers.contains_key("anthropic-version") {
        return true;
    }
    // Check for x-api-key without Authorization Bearer
    if headers.contains_key("x-api-key") && !headers.contains_key("authorization") {
        return true;
    }
    // Check body: Anthropic format uses "max_tokens" but not "messages" with OpenAI structure
    // Actually both use "messages", so rely on headers primarily.
    // As a fallback, check if body has "max_tokens" but not "model" (unlikely to help).
    // The header-based detection is the primary signal.
    let _ = body;
    false
}

/// Detect if a request targets the Responses API format.
#[allow(dead_code)]
pub fn is_responses_request(body: &Value) -> bool {
    // Responses API uses "input" instead of "messages"
    body.get("input").is_some() && body.get("messages").is_none()
}

/// Convert OpenAI Chat Completions response to Responses API format.
pub fn openai_to_responses(openai_resp: &Value, model: &str) -> Value {
    let choice = openai_resp
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first());

    let message = choice.and_then(|ch| ch.get("message"));

    let content = message
        .and_then(|msg| msg.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");

    let finish_reason = choice
        .and_then(|ch| ch.get("finish_reason"))
        .and_then(|f| f.as_str())
        .unwrap_or("stop");

    let prompt_tokens = openai_resp
        .get("usage")
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|t| t.as_u64())
        .unwrap_or(0);
    let completion_tokens = openai_resp
        .get("usage")
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|t| t.as_u64())
        .unwrap_or(0);

    // Build output array: message + function_call items
    let mut output = Vec::new();

    // Add function_call outputs for tool_calls
    if let Some(tool_calls) = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(|t| t.as_array())
    {
        for tc in tool_calls {
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let arguments = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("");
            let call_id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");
            output.push(serde_json::json!({
                "id": format!("fc_{}", uuid::Uuid::new_v4().simple()),
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "status": "completed"
            }));
        }
    }

    // Add text message output (always include, even if empty when tool_calls present)
    if !content.is_empty() || output.is_empty() {
        output.push(serde_json::json!({
            "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": content
            }],
            "status": "completed"
        }));
    }

    serde_json::json!({
        "id": openai_resp.get("id").cloned().unwrap_or(Value::String(format!("resp_{}", uuid::Uuid::new_v4()))),
        "object": "response",
        "created_at": chrono::Utc::now().timestamp(),
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        },
        "status": "completed",
        "finish_reason": finish_reason
    })
}

/// Convert Responses API request to OpenAI Chat Completions format.
pub fn responses_to_openai(body: &Value) -> Value {
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_string();

    // Convert input array to messages array
    let messages = if let Some(input) = body.get("input") {
        convert_responses_input_to_messages(input)
    } else {
        Value::Array(vec![])
    };

    // max_output_tokens -> max_tokens
    let max_tokens = body
        .get("max_output_tokens")
        .and_then(|m| m.as_u64())
        .unwrap_or(4096);

    let stream = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    let mut openai_body = serde_json::json!({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    });

    // Pass through temperature if present
    if let Some(temp) = body.get("temperature") {
        openai_body["temperature"] = temp.clone();
    }
    // Pass through top_p if present
    if let Some(top_p) = body.get("top_p") {
        openai_body["top_p"] = top_p.clone();
    }
    // Convert Responses API tools to Chat Completions tools format.
    // Responses API uses flat format: { type: "function", name, parameters, description }
    // Chat Completions uses nested format: { type: "function", function: { name, parameters, description } }
    if let Some(tools) = body.get("tools") {
        if let Some(arr) = tools.as_array() {
            let openai_tools: Vec<Value> = arr
                .iter()
                .filter_map(|t| {
                    let tool_type = t.get("type").and_then(|ty| ty.as_str()).unwrap_or("");
                    match tool_type {
                        // Function tools: convert flat → nested
                        "function" => {
                            // Already in Chat Completions format (has "function" field) — pass through
                            if t.get("function").is_some() {
                                return Some(t.clone());
                            }
                            // Responses API flat format → convert to Chat Completions nested format.
                            // Chat Completions requires an object JSON schema.
                            let parameters = t.get("parameters").cloned().unwrap_or(Value::Null);
                            let parameters = if parameters.is_null() || !parameters.is_object() {
                                serde_json::json!({"type": "object", "properties": {}})
                            } else {
                                let mut params = parameters;
                                if params.get("type").is_none() {
                                    if let Some(obj) = params.as_object_mut() {
                                        obj.insert(
                                            "type".to_string(),
                                            Value::String("object".to_string()),
                                        );
                                    }
                                }
                                params
                            };
                            let func = serde_json::json!({
                                "name": t.get("name").cloned().unwrap_or(Value::Null),
                                "parameters": parameters,
                            });
                            let mut func_obj = func;
                            if let Some(desc) = t.get("description") {
                                func_obj["description"] = desc.clone();
                            }
                            if let Some(strict) = t.get("strict") {
                                func_obj["strict"] = strict.clone();
                            }
                            Some(serde_json::json!({
                                "type": "function",
                                "function": func_obj
                            }))
                        }
                        // Built-in tools (web_search, file_search, computer_use, etc.) — skip
                        _ => None,
                    }
                })
                .collect();
            if !openai_tools.is_empty() {
                openai_body["tools"] = Value::Array(openai_tools);
            }
        }
    }

    // Pass through tool_choice (format is the same between Responses and Chat Completions)
    if let Some(tc) = body.get("tool_choice") {
        openai_body["tool_choice"] = tc.clone();
    }

    // Pass through instructions as a system message if present
    if let Some(instructions) = body.get("instructions").and_then(|i| i.as_str()) {
        if !instructions.is_empty() {
            if let Some(msgs) = openai_body
                .get_mut("messages")
                .and_then(|m| m.as_array_mut())
            {
                msgs.insert(
                    0,
                    serde_json::json!({
                        "role": "system",
                        "content": instructions
                    }),
                );
            }
        }
    }

    openai_body
}

/// Convert Responses API `input` array to OpenAI `messages` array.
/// Handles: message, function_call (assistant tool call), function_call_output (tool result)
fn convert_responses_input_to_messages(input: &Value) -> Value {
    let messages = if let Some(arr) = input.as_array() {
        // First pass: collect all function_call call_ids and their matching outputs
        let mut call_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut output_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        // Map from original (possibly empty) call_id → fallback call_id
        let mut call_id_fallback: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        let mut fallback_counter = 0u32;
        for item in arr {
            let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match item_type {
                "function_call" => {
                    let cid = item
                        .get("call_id")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();
                    if cid.is_empty() {
                        let fallback = format!("call_{}", fallback_counter);
                        fallback_counter += 1;
                        call_id_fallback.insert(cid.clone(), fallback.clone());
                        call_ids.insert(fallback);
                    } else {
                        call_ids.insert(cid);
                    }
                }
                "function_call_output" => {
                    let cid = item
                        .get("call_id")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();
                    // Use fallback if one was generated for the corresponding function_call
                    let effective_cid = call_id_fallback.get(&cid).cloned().unwrap_or(cid);
                    output_ids.insert(effective_cid);
                }
                _ => {}
            }
        }

        let mut msgs = Vec::new();
        for item in arr {
            let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");

            match item_type {
                // function_call: assistant's tool call → OpenAI assistant message with tool_calls
                "function_call" => {
                    let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let arguments = item.get("arguments").and_then(|a| a.as_str()).unwrap_or("");
                    let original_call_id = item
                        .get("call_id")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();
                    // Use fallback call_id if the original was empty
                    let call_id = call_id_fallback
                        .get(&original_call_id)
                        .cloned()
                        .unwrap_or(original_call_id);
                    msgs.push(serde_json::json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments
                            }
                        }]
                    }));
                    // If this function_call has no matching output, synthesize an empty tool response
                    // to prevent upstream "tool_call_ids did not have response messages" errors
                    if !output_ids.contains(&call_id) {
                        msgs.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": ""
                        }));
                    }
                }

                // function_call_output: tool result → OpenAI tool message
                "function_call_output" => {
                    let original_call_id = item
                        .get("call_id")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();
                    // Use fallback call_id if one was generated for the corresponding function_call
                    let call_id = call_id_fallback
                        .get(&original_call_id)
                        .cloned()
                        .unwrap_or(original_call_id);
                    let output = item.get("output").and_then(|o| o.as_str()).unwrap_or("");
                    msgs.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output
                    }));
                }

                // message: standard chat message
                "message" | _ if item.get("role").is_some() => {
                    let role = item
                        .get("role")
                        .and_then(|r| r.as_str())
                        .unwrap_or("user")
                        .to_string();
                    // Map Roles that some providers don't recognize
                    // 'developer' is an OpenAI alias for 'system' (used by Codex/Responses API)
                    let role = match role.as_str() {
                        "developer" => "system".to_string(),
                        other => other.to_string(),
                    };
                    let content =
                        if let Some(content_arr) = item.get("content").and_then(|c| c.as_array()) {
                            // Extract text from content blocks
                            let texts: Vec<String> = content_arr
                                .iter()
                                .filter_map(|block| {
                                    // input_text, output_text, text
                                    block
                                        .get("text")
                                        .and_then(|t| t.as_str())
                                        .map(|s| s.to_string())
                                })
                                .collect();
                            Value::String(texts.join(""))
                        } else if let Some(text) = item.get("content").and_then(|c| c.as_str()) {
                            Value::String(text.to_string())
                        } else {
                            Value::String(String::new())
                        };
                    msgs.push(serde_json::json!({
                        "role": role,
                        "content": content,
                    }));
                }

                // Simple text item
                _ if item.get("text").is_some() => {
                    let text = item.get("text").and_then(|t| t.as_str()).unwrap_or("");
                    msgs.push(serde_json::json!({
                        "role": "user",
                        "content": text,
                    }));
                }

                // Raw string input
                _ => {
                    if let Some(s) = item.as_str() {
                        msgs.push(serde_json::json!({
                            "role": "user",
                            "content": s,
                        }));
                    }
                }
            }
        }
        msgs
    } else if let Some(s) = input.as_str() {
        // Simple string input
        vec![serde_json::json!({"role": "user", "content": s})]
    } else {
        vec![]
    };

    Value::Array(messages)
}

/// Convert an OpenAI Chat Completions response to Anthropic Messages format.
///
/// This deliberately fails instead of inventing tool input when an upstream
/// returns malformed function arguments. Claude Code uses those arguments to
/// execute local tools, so replacing bad JSON with `{}` is unsafe.
pub fn openai_to_anthropic(openai_resp: &Value, model: &str) -> Result<Value, String> {
    let choice = openai_resp
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first());

    let message = choice.and_then(|ch| ch.get("message"));

    let message = message
        .ok_or_else(|| "OpenAI response does not contain a completion message".to_string())?;
    // Fail-open (CPA semantics): upstream reasoning is surfaced as a Messages
    // `thinking` block, always kept (even when content is also present).  Only
    // the visible text is used; `{text: ...}` object form is unwrapped.
    let reasoning_text = message
        .get("reasoning_content")
        .and_then(|v| match v {
            Value::String(s) if !s.is_empty() => Some(s.clone()),
            Value::Object(m) => m
                .get("text")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(String::from),
            _ => None,
        })
        .or_else(|| match message.get("thinking") {
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            Some(Value::Object(m)) => m
                .get("thinking")
                .or_else(|| m.get("text"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(String::from),
            _ => None,
        });
    let content_text = match message.get("content") {
        None | Some(Value::Null) => "",
        Some(Value::String(value)) => value,
        Some(_) => {
            return Err("OpenAI response has unsupported non-text message content".to_string())
        }
    };

    let finish_reason = choice
        .and_then(|ch| ch.get("finish_reason"))
        .and_then(|f| f.as_str())
        .unwrap_or("");

    // Chat Completions normally sets `tool_calls`, but some compatible
    // upstreams omit it.  The tool-call payload is less ambiguous than a
    // missing finish reason, so do not report a completed tool turn as an
    // ordinary end_turn.
    let has_tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty());
    let stop_reason = match finish_reason {
        "stop" => "end_turn",
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        "content_filter" => "refusal",
        _ if message.get("refusal").is_some() => "refusal",
        _ if has_tool_calls => "tool_use",
        _ => "end_turn",
    };

    let input_tokens = openai_resp
        .get("usage")
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|t| t.as_u64())
        .unwrap_or(0);
    let output_tokens = openai_resp
        .get("usage")
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|t| t.as_u64())
        .unwrap_or(0);

    // Build content array: thinking block (if any) + text blocks + tool_use
    let mut content_blocks = Vec::new();

    // Add thinking block first (reasoning precedes visible text)
    if let Some(rt) = reasoning_text.as_ref().filter(|s| !s.is_empty()) {
        content_blocks.push(serde_json::json!({
            "type": "thinking",
            "thinking": rt
        }));
    }

    // Add text block if present
    if !content_text.is_empty() {
        content_blocks.push(serde_json::json!({
            "type": "text",
            "text": content_text
        }));
    }

    // Add tool_use blocks for tool_calls
    if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tool_calls {
            let id = tc
                .get("id")
                .and_then(|i| i.as_str())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| "OpenAI response tool call is missing its id".to_string())?;
            let func = tc.get("function");
            let name = func
                .and_then(|f| f.get("name").and_then(|n| n.as_str()))
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    "OpenAI response tool call is missing its function name".to_string()
                })?;
            let arguments_str = func
                .and_then(|f| f.get("arguments").and_then(|a| a.as_str()))
                .ok_or_else(|| {
                    "OpenAI response tool call is missing function arguments".to_string()
                })?;
            let input: Value = serde_json::from_str(arguments_str).map_err(|error| {
                format!(
                    "OpenAI response contained invalid tool arguments: {}",
                    error
                )
            })?;
            if !input.is_object() {
                return Err(
                    "OpenAI response tool arguments must decode to a JSON object".to_string(),
                );
            }

            content_blocks.push(serde_json::json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input
            }));
        }
    }

    // If no content blocks at all, add empty text
    if content_blocks.is_empty() {
        content_blocks.push(serde_json::json!({
            "type": "text",
            "text": ""
        }));
    }

    Ok(serde_json::json!({
        "id": openai_resp.get("id").cloned().unwrap_or(Value::String(format!("msg_{}", uuid::Uuid::new_v4().simple()))),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    }))
}

/// Convert an Anthropic Messages request to OpenAI Chat Completions.
///
/// This converter intentionally accepts only the intersection which can be
/// represented by Chat Completions. Native Anthropic channels must bypass it.
pub fn anthropic_to_openai(body: &Value) -> Result<Value, String> {
    // Fail-open (CLIProxyAPI semantics): thinking/output_config are mapped to
    // `reasoning_effort` below; container/context_management are dropped.  The
    // upstream provider adjudicates capability; we never reject thinking.
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .to_string();
    let messages = body
        .get("messages")
        .cloned()
        .unwrap_or(Value::Array(vec![]));
    let max_tokens = body
        .get("max_tokens")
        .and_then(|m| m.as_u64())
        .unwrap_or(4096);
    let stream = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    // Extract top-level system message and prepend it.
    let system = body
        .get("system")
        .map(anthropic_system_content_to_openai_text)
        .transpose()?;

    // Convert Anthropic message content (array format) to OpenAI string format
    let openai_messages = convert_anthropic_messages_to_openai(&messages, system)?;

    let mut openai_body = serde_json::json!({
        "model": model,
        "messages": openai_messages,
        "max_tokens": max_tokens,
        "stream": stream,
    });

    if let Some(temp) = body.get("temperature") {
        openai_body["temperature"] = temp.clone();
    }
    if let Some(top_p) = body.get("top_p") {
        openai_body["top_p"] = top_p.clone();
    }
    // Pass through top_k (OpenAI also supports this via some providers)
    if let Some(top_k) = body.get("top_k") {
        openai_body["top_k"] = top_k.clone();
    }
    // Pass through stop_sequences → stop
    if let Some(stop_seq) = body.get("stop_sequences") {
        openai_body["stop"] = stop_seq.clone();
    }
    if stream {
        let mut options = body
            .get("stream_options")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        if !options.is_object() {
            return Err("stream_options must be an object".to_string());
        }
        options["include_usage"] = Value::Bool(true);
        openai_body["stream_options"] = options;
    }
    // Convert Anthropic tools to OpenAI tools format
    // Anthropic: {"name": "xxx", "description": "xxx", "input_schema": {...}}
    // OpenAI: {"type": "function", "function": {"name": "xxx", "description": "xxx", "parameters": {...}}}
    // Also handles Anthropic built-in tools (web_search, computer_use, etc.) which are skipped.
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let mut openai_tools = Vec::new();
        for tool in tools {
            // `cache_control` on a custom tool is likewise an Anthropic
            // caching annotation and has no Chat Completions equivalent.
            // Get the tool type — Anthropic custom tools use "custom" or have no type field
            let tool_type = tool
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("custom");
            match tool_type {
                // Standard function tools (type "custom" or no type)
                "custom" | "" => {
                    let name = tool
                        .get("name")
                        .and_then(|n| n.as_str())
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| "Anthropic tool is missing its name".to_string())?;
                    let description = tool
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("");
                    let parameters = tool.get("input_schema").cloned().ok_or_else(|| {
                        format!("Anthropic tool '{}' is missing input_schema", name)
                    })?;
                    openai_tools.push(serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": parameters
                        }
                    }));
                }
                _ => {
                    return Err(
                        "Anthropic built-in tools require a native Anthropic Messages channel"
                            .to_string(),
                    )
                }
            }
        }
        if !openai_tools.is_empty() {
            openai_body["tools"] = Value::Array(openai_tools);
        }
    }

    // Convert tool_choice
    // Anthropic: {"type": "auto"} or {"type": "any"} or {"type": "tool", "name": "xxx"}
    // OpenAI: "auto" or "required" or {"type": "function", "function": {"name": "xxx"}}
    if let Some(tc) = body.get("tool_choice") {
        if let Some(tc_type) = tc.get("type").and_then(|t| t.as_str()) {
            let openai_tc = match tc_type {
                "auto" => Value::String("auto".to_string()),
                "any" => Value::String("required".to_string()),
                "tool" => {
                    let name = tc.get("name").and_then(|n| n.as_str()).filter(|s| !s.is_empty())
                        .ok_or_else(|| "Anthropic tool_choice type 'tool' is missing a name".to_string())?;
                    serde_json::json!({
                        "type": "function",
                        "function": {"name": name}
                    })
                }
                _ => return Err("unsupported Anthropic tool_choice requires a native Anthropic Messages channel".to_string()),
            };
            openai_body["tool_choice"] = openai_tc;
        } else if let Some(s) = tc.as_str() {
            let openai_tc = match s {
                "auto" => Value::String("auto".to_string()),
                "any" => Value::String("required".to_string()),
                "tool" => return Err("Anthropic tool_choice 'tool' requires a name".to_string()),
                _ => {
                    return Err(
                        "unsupported Anthropic tool_choice requires a native Anthropic Messages channel"
                            .to_string(),
                    )
                }
            };
            openai_body["tool_choice"] = openai_tc;
        } else {
            return Err("unsupported Anthropic tool_choice requires a native Anthropic Messages channel".to_string());
        }
    }

    if body
        .get("tool_choice")
        .and_then(|choice| choice.get("disable_parallel_tool_use"))
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        openai_body["parallel_tool_calls"] = Value::Bool(false);
    }

    // Fail-open thinking mapping: Anthropic `thinking` / `output_config` →
    // OpenAI `reasoning_effort` (CPA semantics).  Only set when the downstream
    // asked for thinking; otherwise leave unset so the upstream applies its own
    // default.  `container`/`context_management`/`context_management_config`
    // have no Chat equivalent and were dropped above (fail-open).
    if let Some(effort) = anthropic_thinking_to_reasoning_effort(body) {
        openai_body["reasoning_effort"] = Value::String(effort);
    }

    Ok(openai_body)
}

/// Map an Anthropic `thinking` config to an OpenAI `reasoning_effort` value.
///
/// `None` when the downstream did not ask for thinking (or asked for an
/// unrecognized type), in which case `reasoning_effort` is left unset.
fn anthropic_thinking_to_reasoning_effort(body: &Value) -> Option<String> {
    let thinking = body.get("thinking")?;
    if !thinking.is_object() {
        return None;
    }
    let ty = thinking.get("type").and_then(Value::as_str)?;
    match ty {
        "enabled" => match thinking.get("budget_tokens").and_then(Value::as_i64) {
            Some(budget) => crate::protocol::thinking::budget_to_level(budget).map(String::from),
            None => Some("auto".to_string()),
        },
        "adaptive" | "auto" => match body
            .get("output_config")
            .and_then(|oc| oc.get("effort"))
            .and_then(Value::as_str)
        {
            Some(effort) if !effort.trim().is_empty() => {
                Some(effort.trim().to_ascii_lowercase())
            }
            _ => Some("xhigh".to_string()),
        },
        "disabled" => Some("none".to_string()),
        _ => None,
    }
}

/// Estimate structured Anthropic request size for the optional count_tokens endpoint.
pub fn estimate_anthropic_input_tokens(body: &Value) -> u64 {
    fn estimate(value: &Value) -> u64 {
        match value {
            Value::String(text) => ((text.chars().count() as u64) + 3) / 4,
            Value::Array(values) => values.iter().map(estimate).sum(),
            Value::Object(object) => object
                .iter()
                // Image source data is base64, not prompt text. Counting it would
                // overestimate by orders of magnitude on OpenAI-only channels.
                .filter(|(key, _)| !matches!(key.as_str(), "model" | "stream" | "data"))
                .map(|(_, value)| estimate(value))
                .sum(),
            _ => 0,
        }
    }
    estimate(body).max(1)
}

fn tool_result_to_openai_content(block: &Value) -> Result<String, String> {
    match block.get("content") {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(items)) => {
            let mut text = String::new();
            for item in items {
                match item.get("type").and_then(|v| v.as_str()) {
                    Some("text") => text.push_str(item.get("text").and_then(|v| v.as_str()).unwrap_or("")),
                    Some("image") => return Err("tool_result images require a native Anthropic Messages channel".to_string()),
                    _ => return Err("unsupported tool_result content requires a native Anthropic Messages channel".to_string()),
                }
            }
            Ok(text)
        }
        _ => Err("tool_result content must be text or text blocks".to_string()),
    }
}

fn anthropic_system_content_to_openai_text(value: &Value) -> Result<String, String> {
    if let Some(str_val) = value.as_str() {
        Ok(str_val.to_string())
    } else if let Some(arr) = value.as_array() {
        let mut texts = Vec::new();
        for block in arr {
            // Prompt caching changes Anthropic billing/cache behavior but not
            // the text content of a Chat Completions request.  It is safe to
            // drop this annotation on the OpenAI bridge; native channels still
            // receive the original body unchanged.
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => texts.push(
                    block
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                ),
                Some("thinking") => {
                    // Fail-open: reasoning instructions on the system prompt
                    // are dropped (no Chat equivalent), not rejected.
                }
                Some("cache_control") => {
                    return Err(
                        "system cache_control blocks require a native Anthropic Messages channel"
                            .to_string(),
                    )
                }
                _ => {
                    return Err(
                        "unsupported non-text system content requires a native Anthropic Messages channel"
                            .to_string(),
                    )
                }
            }
        }
        Ok(texts.join(""))
    } else {
        Err("system must be text or an array of text blocks".to_string())
    }
}

/// Convert Anthropic messages array to OpenAI messages array.
/// Anthropic content can be string or array of content blocks.
/// Handles: text, tool_use (assistant), tool_result (user)
fn convert_anthropic_messages_to_openai(
    messages: &Value,
    system: Option<String>,
) -> Result<Value, String> {
    let mut msgs = Vec::new();

    // Prepend system message if present
    if let Some(sys) = system {
        msgs.push(serde_json::json!({"role": "system", "content": sys}));
    }

    if let Some(arr) = messages.as_array() {
        for msg in arr {
            let role = msg
                .get("role")
                .and_then(|r| r.as_str())
                .ok_or_else(|| "Anthropic message is missing role".to_string())?
                .to_string();
            if role != "user" && role != "assistant" && role != "system" {
                return Err("only user, assistant, and system Anthropic messages can be sent to OpenAI Chat Completions".to_string());
            }

            if role == "system" {
                let content = msg
                    .get("content")
                    .ok_or_else(|| "system message is missing content".to_string())?;
                msgs.push(serde_json::json!({
                    "role": "system",
                    "content": anthropic_system_content_to_openai_text(content)?,
                }));
                continue;
            }

            if let Some(content_arr) = msg.get("content").and_then(|c| c.as_array()) {
                let mut parts: Vec<Value> = Vec::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                let mut assistant_reasoning = String::new();
                let flush_user_parts = |parts: &mut Vec<Value>, msgs: &mut Vec<Value>| {
                    if !parts.is_empty() {
                        msgs.push(
                            serde_json::json!({"role": "user", "content": std::mem::take(parts)}),
                        );
                    }
                };
                for block in content_arr {
                    // Cache controls are annotations on otherwise supported
                    // blocks.  Strip them instead of rejecting an entire
                    // OpenAI-only Claude Code request.
                    match block.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                        "text" => parts.push(serde_json::json!({"type": "text", "text": block.get("text").and_then(|t| t.as_str()).unwrap_or("")})),
                        "image" => {
                            if role != "user" { return Err("OpenAI Chat Completions cannot safely encode assistant image blocks".to_string()); }
                            let source = block.get("source").ok_or_else(|| "Anthropic image block is missing source".to_string())?;
                            let url = match source.get("type").and_then(|v| v.as_str()) {
                                Some("url") => source.get("url").and_then(|v| v.as_str()).ok_or_else(|| "Anthropic image URL source is missing url".to_string())?.to_string(),
                                Some("base64") => format!("data:{};base64,{}", source.get("media_type").and_then(|v| v.as_str()).ok_or_else(|| "Anthropic base64 image is missing media_type".to_string())?, source.get("data").and_then(|v| v.as_str()).ok_or_else(|| "Anthropic base64 image is missing data".to_string())?),
                                _ => return Err("unsupported Anthropic image source requires a native channel".to_string()),
                            };
                            parts.push(serde_json::json!({"type": "image_url", "image_url": {"url": url}}));
                        }
                        "tool_use" => {
                            if role != "assistant" { return Err("tool_use blocks must be in an assistant message".to_string()); }
                            let id = block.get("id").and_then(|i| i.as_str()).filter(|s| !s.is_empty()).ok_or_else(|| "tool_use is missing id".to_string())?;
                            let name = block.get("name").and_then(|n| n.as_str()).filter(|s| !s.is_empty()).ok_or_else(|| "tool_use is missing name".to_string())?;
                            let input = block.get("input").ok_or_else(|| "tool_use is missing input".to_string())?;
                            if !input.is_object() {
                                return Err("tool_use input must be a JSON object".to_string());
                            }
                            let input = input.clone();
                            tool_calls.push(serde_json::json!({"id": id, "type": "function", "function": {"name": name, "arguments": serde_json::to_string(&input).map_err(|e| e.to_string())?}}));
                        }
                        "tool_result" => {
                            if role != "user" { return Err("tool_result blocks must be in a user message".to_string()); }
                            flush_user_parts(&mut parts, &mut msgs);
                            let tool_use_id = block.get("tool_use_id").and_then(|t| t.as_str()).filter(|s| !s.is_empty()).ok_or_else(|| "tool_result is missing tool_use_id".to_string())?;
                            let result_content = tool_result_to_openai_content(block)?;
                            let is_error = block.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false);
                            msgs.push(serde_json::json!({"role": "tool", "tool_call_id": tool_use_id, "content": if is_error { format!("Tool execution error:\n{}", result_content) } else { result_content }}));
                        }
                        "thinking" => {
                            // Fail-open: assistant reasoning is carried into
                            // the Chat message as `reasoning_content` (OpenAI
                            // non-stream field).  Reasoning on any other role is
                            // dropped — we never inject thinking into a
                            // user/system channel.
                            if role == "assistant" {
                                if let Some(t) = block.get("thinking").and_then(|t| t.as_str()) {
                                    assistant_reasoning.push_str(t);
                                }
                            }
                        }
                        "redacted_thinking" => {
                            // Encrypted/signature form — no usable text; drop.
                        }
                        "cache_control" => return Err("Anthropic cache controls require a native Anthropic Messages channel".to_string()),
                        _ => return Err("unsupported Anthropic content block requires a native Anthropic Messages channel".to_string()),
                    }
                }
                if role == "assistant" {
                    let content = if parts.is_empty() {
                        Value::Null
                    } else if parts
                        .iter()
                        .all(|part| part.get("type").and_then(|v| v.as_str()) == Some("text"))
                    {
                        Value::String(
                            parts
                                .iter()
                                .filter_map(|part| part.get("text").and_then(|v| v.as_str()))
                                .collect::<String>(),
                        )
                    } else {
                        Value::Array(parts)
                    };
                    // Reasoning content extracted from assistant `thinking`
                    // blocks (fail-open mapping to OpenAI `reasoning_content`).
                    let reasoning = if assistant_reasoning.is_empty() {
                        None
                    } else {
                        Some(assistant_reasoning)
                    };
                    if tool_calls.is_empty() && content.is_null() && reasoning.is_none() {
                        return Err("assistant message is empty".to_string());
                    }
                    let mut assistant =
                        serde_json::json!({"role": "assistant", "content": content});
                    if let Some(r) = reasoning {
                        assistant["reasoning_content"] = Value::String(r);
                    }
                    if !tool_calls.is_empty() {
                        assistant["tool_calls"] = Value::Array(tool_calls);
                    }
                    msgs.push(assistant);
                } else {
                    flush_user_parts(&mut parts, &mut msgs);
                }
            } else if let Some(s) = msg.get("content").and_then(|c| c.as_str()) {
                msgs.push(serde_json::json!({
                    "role": role,
                    "content": s.to_string(),
                }));
            } else {
                msgs.push(serde_json::json!({
                    "role": role,
                    "content": msg.get("content").cloned().unwrap_or(Value::String(String::new())),
                }));
            }
        }
    } else {
        return Err("Anthropic messages must be an array".to_string());
    }

    Ok(Value::Array(msgs))
}

#[cfg(test)]
mod anthropic_tests {
    use super::*;

    #[test]
    fn counts_structured_anthropic_input() {
        let body = serde_json::json!({
            "model": "test-model",
            "system": [{"type": "text", "text": "system prompt"}],
            "tools": [{"name": "read", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        });
        assert!(estimate_anthropic_input_tokens(&body) > 1);
    }

    #[test]
    fn maps_tools_parallel_control_and_mixed_tool_results() {
        let request = serde_json::json!({
            "model": "claude-compatible",
            "max_tokens": 32,
            "system": [{"type":"text", "text":"be concise"}],
            "tools": [{"name":"weather", "description":"weather", "input_schema":{"type":"object"}}],
            "tool_choice": {"type":"any", "disable_parallel_tool_use":true},
            "messages": [
                {"role":"assistant", "content":[{"type":"text","text":"checking"},{"type":"tool_use","id":"call_1","name":"weather","input":{"city":"Paris"}}]},
                {"role":"user", "content":[{"type":"tool_result","tool_use_id":"call_1","content":"sunny"},{"type":"text","text":"thanks"}]}
            ]
        });
        let converted = anthropic_to_openai(&request).unwrap();
        assert_eq!(converted["parallel_tool_calls"], false);
        assert_eq!(converted["tool_choice"], "required");
        assert_eq!(
            converted["tools"][0]["function"]["parameters"]["type"],
            "object"
        );
        assert_eq!(
            converted["messages"][1]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
        assert_eq!(converted["messages"][2]["role"], "tool");
        assert_eq!(converted["messages"][3]["content"][0]["text"], "thanks");
    }

    #[test]
    fn maps_mid_conversation_system_messages_to_chat_system_role() {
        let request = serde_json::json!({
            "model": "claude-compatible",
            "messages": [
                {"role":"user", "content":"use the strict profile"},
                {"role":"system", "content":[{"type":"text", "text":"strict profile active", "cache_control":{"type":"ephemeral"}}]},
                {"role":"assistant", "content":[{"type":"text", "text":"ack"}]}
            ]
        });
        let converted = anthropic_to_openai(&request).unwrap();
        assert_eq!(converted["messages"][0]["role"], "user");
        assert_eq!(converted["messages"][1]["role"], "system");
        assert_eq!(converted["messages"][1]["content"], "strict profile active");
        assert_eq!(converted["messages"][2]["role"], "assistant");
    }

    #[test]
    fn legacy_tool_choice_strings_map_or_reject() {
        for (input, expected) in [("auto", "auto"), ("any", "required")] {
            let request = serde_json::json!({
                "model": "model",
                "messages": [{"role":"user", "content":"hi"}],
                "tool_choice": input
            });
            let converted = anthropic_to_openai(&request).unwrap();
            assert_eq!(converted["tool_choice"], expected);
        }

        let request = serde_json::json!({
            "model": "model",
            "messages": [{"role":"user", "content":"hi"}],
            "tool_choice": "tool"
        });
        assert!(anthropic_to_openai(&request).is_err());

        let request = serde_json::json!({
            "model": "model",
            "messages": [{"role":"user", "content":"hi"}],
            "tool_choice": "bogus"
        });
        assert!(anthropic_to_openai(&request).is_err());
    }

    #[test]
    fn legacy_tool_use_requires_input_not_fabricated() {
        let request = serde_json::json!({
            "model": "model",
            "messages": [{"role":"assistant", "content":[{"type":"tool_use", "id":"call_1", "name":"run"}]}]
        });
        assert!(anthropic_to_openai(&request).is_err());

        let request = serde_json::json!({
            "model": "model",
            "messages": [{"role":"assistant", "content":[{"type":"tool_use", "id":"call_1", "name":"run", "input":[]}]}]
        });
        assert!(anthropic_to_openai(&request).is_err());

        let request = serde_json::json!({
            "model": "model",
            "messages": [{"role":"assistant", "content":[{"type":"tool_use", "id":"call_1", "name":"run", "input":{}}]}]
        });
        let converted = anthropic_to_openai(&request).unwrap();
        assert_eq!(
            converted["messages"][0]["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
    }

    #[test]
    fn rejects_invalid_openai_tool_arguments_without_inventing_input() {
        let response = serde_json::json!({"choices":[{"finish_reason":"tool_calls", "message":{"role":"assistant", "content":null, "tool_calls":[{"id":"call_1", "function":{"name":"run", "arguments":"{bad"}}]}}]});
        assert!(openai_to_anthropic(&response, "model").is_err());
    }

    #[test]
    fn rejects_non_object_openai_tool_arguments_and_strips_cache_controls() {
        let response = serde_json::json!({"choices":[{"message":{"role":"assistant", "tool_calls":[{"id":"call_1", "function":{"name":"run", "arguments":"[]"}}]}}]});
        assert!(openai_to_anthropic(&response, "model").is_err());

        let cache_in_system = serde_json::json!({"model":"model", "system":[{"type":"text", "text":"cached", "cache_control":{"type":"ephemeral"}}], "messages":[]});
        assert_eq!(
            anthropic_to_openai(&cache_in_system).unwrap()["messages"][0]["content"],
            "cached"
        );
        let cache_in_message = serde_json::json!({"model":"model", "messages":[{"role":"user", "content":[{"type":"text", "text":"cached", "cache_control":{"type":"ephemeral"}}]}]});
        assert_eq!(
            anthropic_to_openai(&cache_in_message).unwrap()["messages"][0]["content"][0]["text"],
            "cached"
        );
    }

    #[test]
    fn preserves_anthropic_response_shape_for_refusals_and_implicit_tools() {
        let refusal = serde_json::json!({"choices":[{"finish_reason":"content_filter", "message":{"role":"assistant", "content":null, "refusal":"no"}}]});
        let converted = openai_to_anthropic(&refusal, "model").unwrap();
        assert_eq!(converted["stop_reason"], "refusal");
        assert!(converted.get("stop_sequence").is_some());

        let implicit_tool = serde_json::json!({"choices":[{"finish_reason":null, "message":{"role":"assistant", "content":null, "tool_calls":[{"id":"call_1", "function":{"name":"run", "arguments":"{}"}}]}}]});
        assert_eq!(
            openai_to_anthropic(&implicit_tool, "model").unwrap()["stop_reason"],
            "tool_use"
        );
    }

    #[test]
    fn streaming_openai_requests_always_request_late_usage() {
        let request = serde_json::json!({"model":"model", "stream":true, "stream_options":{"include_usage":false, "custom":true}, "messages":[]});
        let converted = anthropic_to_openai(&request).unwrap();
        assert_eq!(converted["stream_options"]["include_usage"], true);
        assert_eq!(converted["stream_options"]["custom"], true);
    }

    #[test]
    fn anthropic_to_openai_maps_thinking_fail_open() {
        // thinking enabled + budget_tokens 1024 -> reasoning_effort "low".
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "u"}],
            "thinking": {"type": "enabled", "budget_tokens": 1024}
        });
        let converted = anthropic_to_openai(&body).unwrap();
        assert_eq!(converted["reasoning_effort"], "low");

        // adaptive + output_config.effort passthrough (lowercased).
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "u"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "HIGH"}
        });
        let converted = anthropic_to_openai(&body).unwrap();
        assert_eq!(converted["reasoning_effort"], "high");

        // container / context_management dropped fail-open (no error).
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "u"}],
            "container": {"type": "super_container"},
            "context_management": {"turns": 4}
        });
        let converted = anthropic_to_openai(&body).unwrap();
        assert!(converted.get("container").is_none());
        assert!(converted.get("context_management").is_none());

        // system thinking block dropped.
        let body = serde_json::json!({
            "model": "m",
            "system": [{"type": "thinking", "thinking": "instruct"}],
            "messages": []
        });
        let converted = anthropic_to_openai(&body).unwrap();
        assert_eq!(converted["messages"][0]["content"], "");

        // assistant thinking block -> reasoning_content; redacted dropped.
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "chain"},
                {"type": "redacted_thinking", "data": "sig"},
                {"type": "text", "text": "answer"}
            ]}]
        });
        let converted = anthropic_to_openai(&body).unwrap();
        assert_eq!(converted["messages"][0]["reasoning_content"], "chain");
        assert_eq!(converted["messages"][0]["content"], "answer");
    }

    #[test]
    fn openai_to_anthropic_maps_reasoning_fail_open() {
        // reasoning_content -> Messages thinking block, kept even with content.
        let response = serde_json::json!({"choices":[{"finish_reason":"stop", "message":{"role":"assistant", "reasoning_content":"chain", "content":"answer"}}]});
        let converted = openai_to_anthropic(&response, "model").unwrap();
        assert_eq!(converted["content"][0]["type"], "thinking");
        assert_eq!(converted["content"][0]["thinking"], "chain");
        assert_eq!(converted["content"][1]["type"], "text");
        assert_eq!(converted["content"][1]["text"], "answer");
    }
}
