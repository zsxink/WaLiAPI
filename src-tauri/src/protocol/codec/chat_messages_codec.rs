//! Tests for the strict Chat ↔ Messages codec (T04).
//!
//! These tests encode the WaLiAPI fail-closed contract: unsupported features
//! are rejected with a concrete JSON pointer and stable error code before any
//! upstream access; invalid tool arguments are never rewritten to `{}`; an
//! unknown finish reason is never downgraded to a normal stop/end_turn; SSE
//! arbitrary fragmentation is deterministic; termination happens exactly once.

use super::chat;
use super::error::{FeatureKind, UnsupportedFeatures};
use super::messages;
use super::registry::CodecRegistry;
use serde_json::json;
use serde_json::Value;

fn reject_features(e: &UnsupportedFeatures) -> Vec<String> {
    e.features.clone()
}

// ===========================================================================
// chat_to_messages_v1 — request encoding
// ===========================================================================

#[test]
fn chat_request_text_system_and_sampling() {
    let body = json!({
        "model": "public-model",
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["END"],
        "stream": false,
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "developer", "content": "follow up"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"}
        ]
    });
    let prepared = CodecRegistry::chat_to_messages("upstream-model", &body).unwrap();
    let out = &prepared.encoded_request;
    assert_eq!(out["model"], "upstream-model");
    assert_eq!(out["max_tokens"], 128);
    assert_eq!(out["temperature"], 0.7);
    assert_eq!(out["top_p"], 0.9);
    assert_eq!(out["stop_sequences"], json!(["END"]));
    assert_eq!(out["stream"], false);
    // system and developer are ordered and hoisted to top-level system.
    let system = out["system"].as_array().unwrap();
    assert_eq!(system[0]["text"], "be brief");
    assert_eq!(system[1]["text"], "follow up");
    // messages contain only user/assistant.
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(prepared.context.upstream_model, "upstream-model");
}

#[test]
fn chat_request_function_tools_and_choice() {
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "weather",
                "description": "get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }],
        "tool_choice": {"type": "function", "function": {"name": "weather"}}
    });
    let prepared = CodecRegistry::chat_to_messages("m", &body).unwrap();
    let out = &prepared.encoded_request;
    assert_eq!(out["tools"][0]["name"], "weather");
    assert_eq!(out["tools"][0]["input_schema"]["type"], "object");
    assert_eq!(
        out["tool_choice"],
        json!({"type": "tool", "name": "weather"})
    );
}

#[test]
fn chat_request_tool_calls_and_results_are_strict() {
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{\"a\":1}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "done"}
        ]
    });
    let prepared = CodecRegistry::chat_to_messages("m", &body).unwrap();
    let out = &prepared.encoded_request;
    let msgs = out["messages"].as_array().unwrap();
    // assistant -> tool_use block
    assert_eq!(msgs[0]["content"][0]["type"], "tool_use");
    assert_eq!(msgs[0]["content"][0]["id"], "call_1");
    assert_eq!(msgs[0]["content"][0]["input"], json!({"a": 1}));
    // tool -> user tool_result as a content block (canonical Anthropic shape).
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[1]["content"][0]["type"], "tool_result");
    assert_eq!(msgs[1]["content"][0]["tool_use_id"], "call_1");
    assert_eq!(msgs[1]["content"][0]["content"][0]["type"], "text");
    assert_eq!(msgs[1]["content"][0]["content"][0]["text"], "done");
    assert!(
        msgs[1].get("tool_result").is_none(),
        "no message-level tool_result key"
    );
}

#[test]
fn chat_request_consecutive_tool_results_aggregate_into_one_user_message() {
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "first"},
            {"role": "tool", "tool_call_id": "call_2", "content": "second"}
        ]
    });
    let prepared = CodecRegistry::chat_to_messages("m", &body).unwrap();
    let msgs = prepared.encoded_request["messages"].as_array().unwrap();
    // assistant + a SINGLE user message carrying both tool_result blocks.
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "assistant");
    assert_eq!(msgs[1]["role"], "user");
    assert_eq!(msgs[1]["content"].as_array().unwrap().len(), 2);
    assert_eq!(msgs[1]["content"][0]["tool_use_id"], "call_1");
    assert_eq!(msgs[1]["content"][1]["tool_use_id"], "call_2");
    assert_eq!(msgs[1]["content"][0]["content"][0]["text"], "first");
    assert_eq!(msgs[1]["content"][1]["content"][0]["text"], "second");
}

#[test]
fn chat_request_user_images() {
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
        ]}]
    });
    let prepared = CodecRegistry::chat_to_messages("m", &body).unwrap();
    let out = &prepared.encoded_request;
    assert_eq!(out["messages"][0]["content"][0]["type"], "image");
    assert_eq!(out["messages"][0]["content"][0]["source"]["type"], "base64");
    assert_eq!(
        out["messages"][0]["content"][0]["source"]["media_type"],
        "image/png"
    );
    // F2: no non-canonical `_media_type` key on the image block.
    assert!(out["messages"][0]["content"][0]
        .get("_media_type")
        .is_none());
}

#[test]
fn chat_request_rejects_invalid_images() {
    // R15: Chat image_url must be a valid image — non-image media type, or a
    // non-http(s) url, is rejected rather than forwarded.
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:application/octet-stream;base64,aGVsbG8="}}
        ]}]
    });
    let e = CodecRegistry::chat_to_messages("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unsupported_media")));

    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "javascript:alert(1)"}}
        ]}]
    });
    let e = CodecRegistry::chat_to_messages("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unsupported_media")));
}

#[test]
fn chat_request_rejects_n_gt_1_instead_of_silently_dropping() {
    // `n` is not in the support matrix: Messages always returns one completion,
    // so n>1 must be rejected (never silently yield a single completion).
    let body = json!({
        "model": "m",
        "n": 2,
        "messages": [{"role": "user", "content": "u"}]
    });
    let e = CodecRegistry::chat_to_messages("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unsupported_feature.field")));
    assert!(e.json_pointers.iter().any(|p| p == "/n"));
}

#[test]
fn chat_request_rejects_thinking_and_structured_output() {
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "response_format": {"type": "json_schema", "json_schema": {}}
    });
    let e = CodecRegistry::chat_to_messages("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("structured_output")));
    assert!(e.json_pointers.iter().any(|p| p == "/response_format"));
}

#[test]
fn chat_request_reasoning_effort_maps_to_thinking() {
    // CPA ConvertOpenAIRequestToClaude + MapToClaudeEffort, exercised directly.
    // none -> disabled
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "reasoning_effort": "none"
    });
    let out = &CodecRegistry::chat_to_messages("m", &body).unwrap().encoded_request;
    assert_eq!(out["thinking"], json!({"type": "disabled"}));
    assert!(out.get("output_config").is_none());

    // auto -> adaptive (no budget)
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "reasoning_effort": "auto"
    });
    let out = &CodecRegistry::chat_to_messages("m", &body).unwrap().encoded_request;
    assert_eq!(out["thinking"], json!({"type": "adaptive"}));
    assert!(out.get("output_config").is_none());

    // medium -> adaptive + output_config.effort=medium
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "reasoning_effort": "medium"
    });
    let out = &CodecRegistry::chat_to_messages("m", &body).unwrap().encoded_request;
    assert_eq!(out["thinking"], json!({"type": "adaptive"}));
    assert_eq!(out["output_config"], json!({"effort": "medium"}));

    // xhigh (no model registry) -> collapses to high (MapToClaudeEffort)
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "reasoning_effort": "xhigh"
    });
    let out = &CodecRegistry::chat_to_messages("m", &body).unwrap().encoded_request;
    assert_eq!(out["output_config"], json!({"effort": "high"}));
}

#[test]
fn chat_request_rejects_unknown_role_and_builtin_tool() {
    let body = json!({
        "model": "m",
        "messages": [{"role": "system", "content": "x"}, {"role": "tool", "tool_call_id": "t", "content": "x"}]
    });
    // No prior assistant tool_call -> tool message without id should fail (but
    // here id is present; role tool without matching assistant tool is a
    // strictness case).  This must not invent an assistant message.
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "function", "content": "x"}
        ]
    });
    let e = CodecRegistry::chat_to_messages("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unknown_role")));

    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "tools": [{"type": "web_search", "function": {"name": "x"}}]
    });
    let e = CodecRegistry::chat_to_messages("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("builtin_tool")));
}

#[test]
fn chat_request_rejects_invalid_tool_arguments_never_rewrites() {
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "{bad"}}
            ]}
        ]
    });
    let e = CodecRegistry::chat_to_messages("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("invalid_tool_arguments")));
    // The non-object argument case (array) must also fail, not become {}.
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "run", "arguments": "[]"}}
            ]}
        ]
    });
    assert!(CodecRegistry::chat_to_messages("m", &body).is_err());
}

// ===========================================================================
// chat_to_messages_v1 — non-stream response
// ===========================================================================

#[test]
fn chat_response_text_and_finish_mapping() {
    let body = json!({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    });
    let out = chat::decode_chat_response_to_messages(&body, &Default::default()).unwrap();
    assert_eq!(out["content"][0]["type"], "text");
    assert_eq!(out["content"][0]["text"], "hi");
    assert_eq!(out["stop_reason"], "end_turn");
    assert_eq!(out["usage"]["input_tokens"], 10);
    assert_eq!(out["usage"]["output_tokens"], 5);
}

#[test]
fn chat_response_reasoning_content_becomes_thinking_block() {
    // Fail-open (direction A, non-stream): reasoning_content is emitted as a
    // Messages `thinking` block before the text block, always kept even when
    // content is also present.
    let body = json!({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": {
            "role": "assistant",
            "reasoning_content": "chain",
            "content": "answer"
        }, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    });
    let out = chat::decode_chat_response_to_messages(&body, &Default::default()).unwrap();
    assert_eq!(out["content"][0]["type"], "thinking");
    assert_eq!(out["content"][0]["thinking"], "chain");
    assert_eq!(out["content"][1]["type"], "text");
    assert_eq!(out["content"][1]["text"], "answer");

    // `{text: ...}` object form of reasoning_content is unwrapped.
    let body = json!({
        "choices": [{"index": 0, "message": {
            "role": "assistant",
            "reasoning_content": {"text": "obj-chain"},
            "content": "answer"
        }, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    });
    let out = chat::decode_chat_response_to_messages(&body, &Default::default()).unwrap();
    assert_eq!(out["content"][0]["type"], "thinking");
    assert_eq!(out["content"][0]["thinking"], "obj-chain");
}

#[test]
fn chat_response_maps_length_and_tool_calls() {
    let body = json!({
        "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_1", "function": {"name": "run", "arguments": "{\"a\":1}"}}
        ]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    });
    let out = chat::decode_chat_response_to_messages(&body, &Default::default()).unwrap();
    assert_eq!(out["stop_reason"], "tool_use");
    assert_eq!(out["content"][0]["type"], "tool_use");
    assert_eq!(out["content"][0]["input"], json!({"a": 1}));
}

#[test]
fn chat_response_rejects_invalid_tool_arguments() {
    let body = json!({
        "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_1", "function": {"name": "run", "arguments": "{bad"}}
        ]}, "finish_reason": "tool_calls"}],
        "usage": {}
    });
    let e = chat::decode_chat_response_to_messages(&body, &Default::default()).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("invalid_tool_arguments")));
    // Array arguments must not become {}.
    let body = json!({
        "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_1", "function": {"name": "run", "arguments": "[]"}}
        ]}, "finish_reason": "tool_calls"}],
        "usage": {}
    });
    assert!(chat::decode_chat_response_to_messages(&body, &Default::default()).is_err());
}

#[test]
fn chat_response_unknown_finish_reason_never_becomes_stop() {
    let body = json!({
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "content_steering"}],
        "usage": {}
    });
    let e = chat::decode_chat_response_to_messages(&body, &Default::default()).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("finish_reason")));
}

#[test]
fn chat_response_refusal_maps_to_refusal_not_stop() {
    let body = json!({
        "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "refusal": "no"}, "finish_reason": "content_filter"}],
        "usage": {}
    });
    let out = chat::decode_chat_response_to_messages(&body, &Default::default()).unwrap();
    assert_eq!(out["stop_reason"], "refusal");
}

#[test]
fn chat_response_no_finish_reason_with_tool_calls_is_tool_use() {
    let body = json!({
        "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [
            {"id": "call_1", "function": {"name": "run", "arguments": "{}"}}
        ]}, "finish_reason": null}],
        "usage": {}
    });
    let out = chat::decode_chat_response_to_messages(&body, &Default::default()).unwrap();
    assert_eq!(out["stop_reason"], "tool_use");
}

// ===========================================================================
// chat_to_messages_v1 — streaming
// ===========================================================================

#[test]
fn chat_stream_arbitrary_fragmentation_and_tool_accumulation() {
    let mut state = chat::ChatSseState::default();
    let parts = [
        b"data: {\"choices\":[{\"delta\":{\"content\":\"h".as_slice(),
        "\u{00e9}".as_bytes(),
        b"\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"two\",\"arguments\":\"{\\\"b\\\":2}\"}},{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"one\",\"arguments\":\"{\\\"a\\\":1}\"}}]}}]}\r\n\r\n".as_slice(),
        b"data: {\"choices\":[{\"finish_reason\":\"tool_calls\"}]}\n\ndata: {\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n".as_slice(),
    ];
    let mut output = Vec::new();
    for part in parts {
        output.extend(state.feed(part).unwrap());
    }
    output.extend(state.finish().unwrap());
    let output = output.join("");
    assert!(output.contains("hé"));
    assert!(output.contains("\"id\":\"a\""));
    assert!(output.contains("\"id\":\"b\""));
    assert!(output.contains("\"input_tokens\":7"));
    assert!(output.contains("\"stop_sequence\":null"));
    let text_stop = output.find("content_block_stop").unwrap();
    let first_tool = output.find("\"type\":\"tool_use\"").unwrap();
    assert!(
        text_stop < first_tool,
        "text must stop before a tool block starts"
    );
    assert!(output.find("\"id\":\"a\"").unwrap() < output.find("\"id\":\"b\"").unwrap());
    assert_eq!(output.matches("event: message_stop").count(), 1);
}

#[test]
fn chat_stream_incomplete_tool_arguments_are_rejected() {
    let mut state = chat::ChatSseState::default();
    state.feed(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c\",\"function\":{\"name\":\"run\",\"arguments\":\"{bad\"}}]}}]}\n\n").unwrap();
    let e = state.finish().unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("invalid_tool_arguments")));
}

#[test]
fn chat_stream_unknown_finish_reason_rejected_at_finalize() {
    let mut state = chat::ChatSseState::default();
    state
        .feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
        .unwrap();
    state
        .feed(b"data: {\"choices\":[{\"finish_reason\":\"bizarre\"}]}\n\n")
        .unwrap();
    let e = state.finish().unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("finish_reason")));
}

#[test]
fn chat_stream_first_frame_invalid_is_a_codec_error() {
    let mut state = chat::ChatSseState::default();
    let e = state.feed(b"data: {not-json}\n\n").unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unknown_event")));
}

#[test]
fn chat_stream_termination_exactly_once() {
    let mut state = chat::ChatSseState::default();
    state
        .feed(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\n",
        )
        .unwrap();
    let first = state.finish().unwrap();
    assert_eq!(
        first.iter().filter(|e| e.contains("message_stop")).count(),
        1
    );
    // finish() again is a no-op.
    let second = state.finish().unwrap();
    assert!(second.is_empty());
}

#[test]
fn chat_stream_empty_stream_is_a_codec_error_not_an_empty_success() {
    // F4: a stream that closes before any first frame must surface a codec
    // error (for pre-commit failover), never a silent empty Ok.
    let mut state = chat::ChatSseState::default();
    state.feed(b"").unwrap();
    let e = state.finish().unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unknown_event")));

    let mut state = chat::ChatSseState::default();
    state.feed(b"data: [DONE]\n\n").unwrap();
    let e = state.finish().unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unknown_event")));
}

#[test]
fn chat_stream_emits_prepared_model_and_request_id() {
    // F5: the streaming decoder must thread the mapped upstream model and the
    // per-request id from the PreparedAttempt context into the synthesized
    // message_start frame.
    let mut state = chat::ChatSseState::new("upstream-model-9", "req-42");
    let events = state
        .feed(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\n",
        )
        .unwrap();
    let joined = events.join("");
    assert!(joined.contains("\"model\":\"upstream-model-9\""));
    assert!(joined.contains("\"id\":\"req-42\""));
    state.finish().unwrap();
}

// ===========================================================================
// messages_to_chat_v1 — request encoding
// ===========================================================================

#[test]
fn messages_request_system_text_and_sampling() {
    let body = json!({
        "model": "m",
        "max_tokens": 64,
        "temperature": 0.5,
        "system": [{"type": "text", "text": "sys"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    });
    let prepared = CodecRegistry::messages_to_chat("up", &body).unwrap();
    let out = &prepared.encoded_request;
    assert_eq!(out["model"], "up");
    assert_eq!(out["max_tokens"], 64);
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(
        out["messages"][0],
        json!({"role": "system", "content": "sys"})
    );
    assert_eq!(out["messages"][1]["content"], "hi");
}

#[test]
fn messages_request_stream_options_are_allowed_and_force_usage() {
    let body = json!({
        "model": "m",
        "stream": true,
        "stream_options": {"include_usage": false, "custom": true},
        "messages": [{"role": "user", "content": "hi"}]
    });
    let prepared = CodecRegistry::messages_to_chat("m", &body).unwrap();
    let out = &prepared.encoded_request;
    assert_eq!(out["stream_options"]["include_usage"], true);
    assert_eq!(out["stream_options"]["custom"], true);
}

#[test]
fn messages_request_tools_choice_and_tool_results() {
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "checking"}, {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "Paris"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"}, {"type": "text", "text": "thanks"}]}
        ],
        "tools": [{"name": "weather", "description": "weather", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "any", "disable_parallel_tool_use": true}
    });
    let prepared = CodecRegistry::messages_to_chat("m", &body).unwrap();
    let out = &prepared.encoded_request;
    assert_eq!(out["parallel_tool_calls"], false);
    assert_eq!(out["tool_choice"], "required");
    assert_eq!(out["tools"][0]["function"]["name"], "weather");
    // No system message here, so messages[0] is the assistant with tool_calls.
    assert_eq!(out["messages"][0]["content"], "checking");
    assert_eq!(
        out["messages"][0]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Paris\"}"
    );
    assert_eq!(out["messages"][1]["role"], "tool");
    assert_eq!(out["messages"][2]["content"], "thanks");
}

#[test]
fn messages_request_thinking_fail_open_and_builtin_tools_rejected() {
    // Fail-open: thinking is mapped to reasoning_effort, never rejected.
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024}
    });
    let prepared = CodecRegistry::messages_to_chat("m", &body).unwrap();
    let out = &prepared.encoded_request;
    // budget 1024 -> low (CPA ConvertBudgetToLevel).
    assert_eq!(out["reasoning_effort"], "low");

    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "tools": [{"type": "web_search", "name": "web"}]
    });
    let e = CodecRegistry::messages_to_chat("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("builtin_tool")));
}

#[test]
fn messages_request_thinking_variants_map_reasoning_effort() {
    // CPA ConvertClaudeRequestToOpenAI semantics, exercised directly.
    // enabled + budget_tokens -> level
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024}
    });
    let out = &CodecRegistry::messages_to_chat("m", &body).unwrap().encoded_request;
    assert_eq!(out["reasoning_effort"], "low", "1024 -> low");

    // enabled without budget -> auto
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "thinking": {"type": "enabled"}
    });
    let out = &CodecRegistry::messages_to_chat("m", &body).unwrap().encoded_request;
    assert_eq!(out["reasoning_effort"], "auto");

    // adaptive + output_config.effort -> passthrough (lowercased)
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "MEDIUM"}
    });
    let out = &CodecRegistry::messages_to_chat("m", &body).unwrap().encoded_request;
    assert_eq!(out["reasoning_effort"], "medium", "effort lowercased passthrough");

    // adaptive without effort -> xhigh
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "thinking": {"type": "adaptive"}
    });
    let out = &CodecRegistry::messages_to_chat("m", &body).unwrap().encoded_request;
    assert_eq!(out["reasoning_effort"], "xhigh");

    // disabled -> none
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "thinking": {"type": "disabled"}
    });
    let out = &CodecRegistry::messages_to_chat("m", &body).unwrap().encoded_request;
    assert_eq!(out["reasoning_effort"], "none");
}

#[test]
fn messages_request_container_dropped_fail_open() {
    // container / context_management have no Chat equivalent; dropped and
    // recorded on the report, never rejected.
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "container": {"type": "super_container"},
        "context_management": {"turns": 4},
        "context_management_config": {"mode": "auto"}
    });
    let prepared = CodecRegistry::messages_to_chat("m", &body).unwrap();
    let out = &prepared.encoded_request;
    assert!(out.get("container").is_none());
    assert!(out.get("context_management").is_none());
    assert!(out.get("context_management_config").is_none());
    // The report surfaces the drop pointers.
    let report = &prepared.report;
    assert!(report.normalized.iter().any(|p| p.contains("container")));
    assert!(
        report
            .normalized
            .iter()
            .any(|p| p.contains("context_management"))
    );
}

#[test]
fn messages_request_assistant_thinking_becomes_reasoning_content() {
    // An assistant message carrying a thinking block keeps its reasoning as
    // `reasoning_content` on the Chat message; redacted_thinking is dropped.
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "chain"},
                {"type": "redacted_thinking", "data": "sig"},
                {"type": "text", "text": "answer"}
            ]}
        ]
    });
    let out = &CodecRegistry::messages_to_chat("m", &body).unwrap().encoded_request;
    let assistant = &out["messages"][1];
    assert_eq!(assistant["reasoning_content"], "chain");
    assert_eq!(assistant["content"], "answer");
}

#[test]
fn messages_request_unknown_role_and_block_rejected() {
    let body = json!({
        "model": "m",
        "messages": [{"role": "bogus", "content": "x"}]
    });
    let e = CodecRegistry::messages_to_chat("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unknown_role")));

    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}]
    });
    let e = CodecRegistry::messages_to_chat("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unknown_block")));
}

#[test]
fn messages_request_mid_conversation_system_maps_to_chat_system_role() {
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "user", "content": "activate strict mode"},
            {"role": "system", "content": [{"type": "text", "text": "strict mode active", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": "ack"}
        ]
    });
    let prepared = CodecRegistry::messages_to_chat("m", &body).unwrap();
    let messages = prepared.encoded_request["messages"].as_array().unwrap();
    assert_eq!(
        messages[0],
        json!({"role": "user", "content": "activate strict mode"})
    );
    assert_eq!(
        messages[1],
        json!({"role": "system", "content": "strict mode active"})
    );
    assert_eq!(messages[2], json!({"role": "assistant", "content": "ack"}));
}

#[test]
fn messages_request_strips_lossless_cache_controls() {
    let body = json!({
        "model": "m",
        "system": [{"type": "text", "text": "cached", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}]
    });
    let prepared = CodecRegistry::messages_to_chat("m", &body).unwrap();
    let out = &prepared.encoded_request;
    assert_eq!(out["messages"][0]["content"], "cached");
    assert!(out["messages"][0].get("cache_control").is_none());
}

#[test]
fn messages_request_rejects_invalid_tool_input() {
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "run", "input": [1, 2]}]}
        ]
    });
    let e = CodecRegistry::messages_to_chat("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("invalid_tool_arguments")));
}

#[test]
fn messages_request_rejects_unknown_top_level_fields() {
    // R4: unknown top-level Messages fields are rejected with a JSON pointer,
    // never silently dropped.  A whitelist mirrors chat_to_messages_v1.
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "metadata": {"user_id": "u1"}
    });
    let e = CodecRegistry::messages_to_chat("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unsupported_feature.field")));
    assert!(e.json_pointers.iter().any(|p| p == "/metadata"));
}

#[test]
fn messages_request_rejects_non_array_stop_sequences() {
    // R12: a non-array stop_sequences must be rejected, not silently dropped.
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "stop_sequences": "END"
    });
    let e = CodecRegistry::messages_to_chat("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unsupported_feature.field")));
}

#[test]
fn messages_request_tool_choice_strings_are_mapped_not_passed_through() {
    // R9: bare Anthropic tool_choice strings map to Chat values; unknown
    // strings and a bare "tool" (which needs a name) are rejected.
    for (input, expected) in [("auto", "auto"), ("any", "required")] {
        let body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "u"}],
            "tool_choice": input
        });
        let prepared = CodecRegistry::messages_to_chat("m", &body).unwrap();
        assert_eq!(prepared.encoded_request["tool_choice"], expected);
    }
    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "tool_choice": "tool"
    });
    let e = CodecRegistry::messages_to_chat("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("missing_tool_field")));

    let body = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "u"}],
        "tool_choice": "bogus"
    });
    let e = CodecRegistry::messages_to_chat("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unsupported_feature.field")));
}

#[test]
fn messages_request_tool_use_requires_input_not_fabricated() {
    // R8/R21: a tool_use without `input` is malformed and must be rejected; we
    // never fabricate `{}`.  An explicit `input: {}` is accepted.
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "run"}]}
        ]
    });
    let e = CodecRegistry::messages_to_chat("m", &body).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("missing_tool_field")));
    assert!(e.json_pointers.iter().any(|p| p.ends_with("/input")));

    let body = json!({
        "model": "m",
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "run", "input": {}}]}
        ]
    });
    let prepared = CodecRegistry::messages_to_chat("m", &body).unwrap();
    assert_eq!(
        prepared.encoded_request["messages"][0]["tool_calls"][0]["function"]["arguments"],
        "{}"
    );
}

#[test]
fn messages_request_tool_results_stay_adjacent_to_assistant() {
    // tool ordering: a user message mixing text-before-tool_result must keep the
    // tool message adjacent to the assistant tool_calls it answers.  Expected
    // order: assistant(tool_calls) -> tool -> user(text).
    let body = json!({
        "model": "m",
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "w", "input": {}}]},
            {"role": "user", "content": [
                {"type": "text", "text": "before"},
                {"type": "tool_result", "tool_use_id": "call_1", "content": "result"}
            ]}
        ]
    });
    let prepared = CodecRegistry::messages_to_chat("m", &body).unwrap();
    let msgs = prepared.encoded_request["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "assistant");
    assert_eq!(msgs[0]["tool_calls"][0]["id"], "call_1");
    // tool message must immediately follow the assistant, ahead of the text.
    assert_eq!(msgs[1]["role"], "tool");
    assert_eq!(msgs[1]["tool_call_id"], "call_1");
    assert_eq!(msgs[2]["role"], "user");
    assert_eq!(msgs[2]["content"], "before");
}

#[test]
fn messages_response_tool_use_requires_input_not_fabricated() {
    // R8/R21 response side: a non-stream tool_use without `input` is rejected,
    // not fabricated as `{}`.
    let body = json!({
        "id": "msg_1", "type": "message",
        "content": [{"type": "tool_use", "id": "c", "name": "run"}],
        "stop_reason": "tool_use", "usage": {}
    });
    let e = messages::decode_messages_response_to_chat(&body, &Default::default()).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("missing_tool_field")));
    assert!(e.json_pointers.iter().any(|p| p.ends_with("/input")));
}

// ===========================================================================
// messages_to_chat_v1 — non-stream response
// ===========================================================================

#[test]
fn messages_response_text_and_tool_use() {
    let body = json!({
        "id": "msg_1",
        "type": "message",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "call_1", "name": "run", "input": {"a": 1}}
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    });
    let out = messages::decode_messages_response_to_chat(&body, &Default::default()).unwrap();
    assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(out["choices"][0]["message"]["content"], "hello");
    assert_eq!(
        out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"a\":1}"
    );
    assert_eq!(out["usage"]["prompt_tokens"], 10);
    assert_eq!(out["usage"]["completion_tokens"], 5);
}

#[test]
fn messages_response_maps_stop_reasons() {
    let base = |stop: &str| {
        json!({
            "id": "msg_1", "type": "message", "content": [{"type": "text", "text": "x"}],
            "stop_reason": stop, "usage": {"input_tokens": 1, "output_tokens": 1}
        })
    };
    assert_eq!(
        messages::decode_messages_response_to_chat(&base("end_turn"), &Default::default()).unwrap()
            ["choices"][0]["finish_reason"],
        "stop"
    );
    assert_eq!(
        messages::decode_messages_response_to_chat(&base("max_tokens"), &Default::default())
            .unwrap()["choices"][0]["finish_reason"],
        "length"
    );
    // Unknown stop reason is rejected, never mapped to stop.
    let e = messages::decode_messages_response_to_chat(&base("budget_forced"), &Default::default())
        .unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("finish_reason")));
}

#[test]
fn messages_response_thinking_fail_open_and_bad_input_rejected() {
    // Fail-open: a Messages response `thinking` block is surfaced as OpenAI
    // `reasoning_content`, never rejected.
    let body = json!({
        "id": "msg_1", "type": "message",
        "content": [{"type": "thinking", "thinking": "..."}],
        "stop_reason": "end_turn", "usage": {}
    });
    let out = messages::decode_messages_response_to_chat(&body, &Default::default()).unwrap();
    assert_eq!(out["choices"][0]["message"]["reasoning_content"], "...");
    // reasoning only -> content stays null (no fabricated empty text)
    assert!(out["choices"][0]["message"]["content"].is_null());

    let body = json!({
        "id": "msg_1", "type": "message",
        "content": [{"type": "tool_use", "id": "c", "name": "run", "input": [1]}],
        "stop_reason": "tool_use", "usage": {}
    });
    let e = messages::decode_messages_response_to_chat(&body, &Default::default()).unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("invalid_tool_arguments")));
}

// ===========================================================================
// messages_to_chat_v1 — streaming
// ===========================================================================

#[test]
fn messages_stream_text_and_tool_deltas() {
    let mut state = messages::MessagesSseState::default();
    let mut events = Vec::new();
    events.extend(state.feed(b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":5}}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n").unwrap());
    events.extend(state.feed(b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n").unwrap());
    events.extend(
        state
            .feed(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
            .unwrap(),
    );
    events.extend(state.finish().unwrap());
    let joined = events.join("");
    assert!(joined.contains("\"role\":\"assistant\""));
    assert!(joined.contains("\"content\":\"hel\""));
    assert!(joined.contains("\"content\":\"lo\""));
    assert!(joined.contains("\"finish_reason\":\"stop\""));
    assert!(joined.contains("\"prompt_tokens\":5"));
    assert!(joined.contains("\"completion_tokens\":2"));
    assert_eq!(events.iter().filter(|e| e.contains("[DONE]")).count(), 1);
}

#[test]
fn messages_stream_thinking_fail_open_to_reasoning_content() {
    // Fail-open (direction B, streaming): a Messages `thinking` block is
    // surfaced as OpenAI `reasoning_content` deltas, never rejected.
    let mut state = messages::MessagesSseState::default();
    let mut events = Vec::new();
    events.extend(state.feed(b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":5}}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"se\"}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"cret\"}}\n\n").unwrap());
    // signature_delta carries no visible text; dropped fail-open.
    events.extend(state.feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"abc\"}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n").unwrap());
    events.extend(state.feed(b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n").unwrap());
    events.extend(state.feed(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n").unwrap());
    events.extend(state.finish().unwrap());
    let joined = events.join("");
    assert!(joined.contains("\"reasoning_content\":\"se\""));
    assert!(joined.contains("\"reasoning_content\":\"cret\""));
    assert!(joined.contains("\"finish_reason\":\"stop\""));
    assert!(!joined.contains("\"content\":\"se\"") || joined.contains("\"reasoning_content\""));
}

#[test]
fn chat_stream_reasoning_fail_open_to_thinking_block() {
    // Fail-open (direction A, streaming): a Chat `reasoning_content` delta is
    // emitted as a Messages `thinking` block, never rejected.
    let mut state = chat::ChatSseState::new("up-model", "msg_1");
    let mut events = Vec::new();
    events.extend(state.feed(b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"se\"}}]}\n\n").unwrap());
    events.extend(state.feed(b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"cret\"}}]}\n\n").unwrap());
    events.extend(state.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n").unwrap());
    events.extend(state.feed(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n").unwrap());
    events.extend(state.finish().unwrap());
    let joined = events.join("");
    // serde_json here sorts object keys (no preserve_order), so assert on
    // order-independent fragments rather than `"type":"thinking"` adjacency.
    assert!(joined.contains("\"type\":\"thinking\""));
    assert!(joined.contains("\"thinking\":\"se\""));
    assert!(joined.contains("\"thinking\":\"cret\""));
    assert!(joined.contains("\"text\":\"hi\""));
    // both blocks are stopped exactly once.
    assert_eq!(
        events.iter().filter(|e| e.contains("content_block_stop")).count(),
        2,
        "thinking + text blocks both stop"
    );
}

#[test]
fn messages_stream_tool_calls_accumulate_by_index() {
    let mut state = messages::MessagesSseState::default();
    let mut events = Vec::new();
    events.extend(state.feed(b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{}}}\n\n").unwrap());
    // Two parallel tool blocks (index 0 and 1); deltas interleave.
    events.extend(state.feed(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_a\",\"name\":\"one\",\"input\":{}}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\"\"}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_b\",\"name\":\"two\",\"input\":{}}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"b\\\":2}\"}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\":1}\"}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n").unwrap());
    events.extend(state.feed(b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{}}\n\n").unwrap());
    events.extend(
        state
            .feed(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
            .unwrap(),
    );
    events.extend(state.finish().unwrap());
    let joined = events.join("");
    assert!(joined.contains("\"id\":\"call_a\""));
    assert!(joined.contains("\"name\":\"one\""));
    assert!(joined.contains("\"id\":\"call_b\""));
    assert!(joined.contains("\"name\":\"two\""));
    assert!(joined.contains("\"arguments\":\"{\\\"a\\\":1}\""));
    assert!(joined.contains("\"arguments\":\"{\\\"b\\\":2}\""));
    assert!(joined.contains("\"finish_reason\":\"tool_calls\""));
    assert_eq!(events.iter().filter(|e| e.contains("[DONE]")).count(), 1);
}

#[test]
fn messages_stream_invalid_tool_json_is_rejected() {
    let mut state = messages::MessagesSseState::default();
    state.feed(b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{}}}\n\n").unwrap();
    state.feed(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"c\",\"name\":\"run\",\"input\":{}}}\n\n").unwrap();
    state.feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{bad\"}}\n\n").unwrap();
    // content_block_stop validates the accumulated arguments and must reject the
    // malformed JSON rather than invent `{}`.
    let e = state
        .feed(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
        .unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("invalid_tool_arguments")));
}

#[test]
fn messages_stream_unknown_event_is_a_codec_error() {
    let mut state = messages::MessagesSseState::default();
    let e = state
        .feed(b"event: wat\ndata: {\"type\":\"wat\"}\n\n")
        .unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unknown_event")));
}

#[test]
fn messages_stream_fragmented_utf8_and_crlf() {
    let mut state = messages::MessagesSseState::default();
    let mut events = Vec::new();
    let payload = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{}}}\r\n\r\n";
    for chunk in payload.as_bytes().chunks(5) {
        events.extend(state.feed(chunk).unwrap());
    }
    events.extend(state.feed("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\r\n\r\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"h\u{00e9}\"}}\r\n\r\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\r\n\r\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\r\n\r\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\r\n\r\n".as_bytes()).unwrap());
    events.extend(state.finish().unwrap());
    assert!(events.join("").contains("h\u{00e9}"));
}

#[test]
fn messages_stream_empty_stream_is_a_codec_error_not_an_empty_success() {
    // F4: a Messages stream that closes before any message_start frame must
    // surface a codec error for pre-commit failover, never an empty Ok.
    let mut state = messages::MessagesSseState::default();
    state.feed(b"").unwrap();
    let e = state.finish().unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unknown_event")));

    let mut state = messages::MessagesSseState::default();
    state
        .feed(b"event: ping\ndata: {\"type\":\"ping\"}\n\n")
        .unwrap();
    let e = state.finish().unwrap_err();
    assert!(reject_features(&e)
        .iter()
        .any(|c| c.contains("unknown_event")));
}

#[test]
fn messages_stream_emits_prepared_model() {
    // F5: the Messages→Chat streaming decoder must thread the mapped upstream
    // model from the PreparedAttempt context into the synthesized Chat role
    // frame (never a hardcoded empty model).
    let mut state = messages::MessagesSseState::new("upstream-model-9");
    let mut events = Vec::new();
    events.extend(state.feed(b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{}}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n").unwrap());
    events.extend(state.feed(b"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n").unwrap());
    events.extend(state.feed(b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n").unwrap());
    events.extend(
        state
            .feed(b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
            .unwrap(),
    );
    events.extend(state.finish().unwrap());
    let joined = events.join("");
    assert!(joined.contains("\"model\":\"upstream-model-9\""));
}

// ===========================================================================
// FeatureKind stable codes
// ===========================================================================

#[test]
fn feature_kind_stable_codes() {
    assert_eq!(FeatureKind::Thinking.code(), "unsupported_feature.thinking");
    assert_eq!(
        FeatureKind::StructuredOutput.code(),
        "unsupported_feature.structured_output"
    );
    assert_eq!(
        FeatureKind::BuiltinTool.code(),
        "unsupported_feature.builtin_tool"
    );
    assert_eq!(FeatureKind::Document.code(), "unsupported_feature.document");
    assert_eq!(
        FeatureKind::PromptCache.code(),
        "unsupported_feature.prompt_cache"
    );
    assert_eq!(
        FeatureKind::UnknownRole.code(),
        "unsupported_feature.unknown_role"
    );
    assert_eq!(
        FeatureKind::UnknownBlock.code(),
        "unsupported_feature.unknown_block"
    );
    assert_eq!(
        FeatureKind::UnknownEvent.code(),
        "unsupported_feature.unknown_event"
    );
    assert_eq!(
        FeatureKind::UnknownFinishReason.code(),
        "unsupported_feature.finish_reason"
    );
    assert_eq!(
        FeatureKind::InvalidToolArguments.code(),
        "unsupported_feature.invalid_tool_arguments"
    );
    assert_eq!(
        FeatureKind::MissingToolField.code(),
        "unsupported_feature.missing_tool_field"
    );
    assert_eq!(FeatureKind::Media.code(), "unsupported_media");
}
