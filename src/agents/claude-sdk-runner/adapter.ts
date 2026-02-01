/**
 * Claude Agent SDK Adapter
 *
 * Drop-in replacement for runEmbeddedPiAgent using Claude Agent SDK.
 * Accepts RunEmbeddedPiAgentParams and returns EmbeddedPiRunResult.
 *
 * @see clawd-7xvm (BD task)
 * @see clawd-fbtq (BD epic)
 */

import os from "node:os";

import { query } from "@anthropic-ai/claude-agent-sdk";

import type { RunEmbeddedPiAgentParams } from "../pi-embedded-runner/run/params.js";
import type {
  EmbeddedPiRunResult,
  EmbeddedPiRunMeta,
  EmbeddedPiAgentMeta,
} from "../pi-embedded-runner/types.js";
import type { MessagingToolSend } from "../pi-embedded-messaging.js";
import { resolveHeartbeatPrompt } from "../../auto-reply/heartbeat.js";
import { resolveChannelCapabilities } from "../../config/channel-capabilities.js";
import { getMachineDisplayName } from "../../infra/machine-name.js";
import { resolveTelegramInlineButtonsScope } from "../../telegram/inline-buttons.js";
import { resolveTelegramReactionLevel } from "../../telegram/reaction-level.js";
import { normalizeMessageChannel } from "../../utils/message-channel.js";
import { isReasoningTagProvider } from "../../utils/provider-utils.js";
import { isCompactionFailureError, isContextOverflowError } from "../pi-embedded-helpers.js";
import { resolveBootstrapContextForRun } from "../bootstrap-files.js";
import { resolveOpenClawDocsPath } from "../docs-path.js";
import { resolveSessionAgentIds } from "../agent-scope.js";
import { listChannelSupportedActions } from "../channel-tools.js";
import { resolveDefaultModelForAgent } from "../model-selection.js";
import { createOpenClawCodingTools } from "../pi-tools.js";
import type { AnyAgentTool } from "../pi-tools.types.js";
import { buildModelAliasLines } from "../pi-embedded-runner/model.js";
import { buildEmbeddedSystemPrompt } from "../pi-embedded-runner/system-prompt.js";
import { buildSystemPromptParams } from "../system-prompt-params.js";
import { appendSessionTurns, buildSessionHistoryPrompt } from "./session-bridge.js";
import { createClawdbotMcpServerWithTools, createMessagingToolTracker } from "./tool-wrapper.js";
import { isSubagentSessionKey } from "../../routing/session-key.js";

const CLAUDE_SDK_PROVIDER = "claude-agent-sdk";
const CLAUDE_SDK_MODELS = new Set(["opus", "sonnet", "haiku"]);
const CLAUDE_SDK_MODEL_ALIASES = new Map<string, string>([
  ["opus-4.5", "opus"],
  ["opus-4-5", "opus"],
  ["sonnet-4.5", "sonnet"],
  ["sonnet-4-5", "sonnet"],
  ["claude-opus-4-20250514", "opus"],
  ["claude-opus-4-5", "opus"],
  ["claude-opus-4.5", "opus"],
  ["claude-sonnet-4-5-20250929", "sonnet"],
  ["claude-sonnet-4-5", "sonnet"],
  ["claude-sonnet-4.5", "sonnet"],
]);

// ============================================================================
// Main Adapter Function
// ============================================================================

/**
 * Run an agent turn using Claude Agent SDK.
 *
 * This is a drop-in replacement for runEmbeddedPiAgent.
 * Accepts the same params and returns the same result shape.
 */
export async function runClaudeSdkAgent(
  params: RunEmbeddedPiAgentParams,
): Promise<EmbeddedPiRunResult> {
  const startTime = Date.now();
  const payloads: NonNullable<EmbeddedPiRunResult["payloads"]> = [];
  const messagingTracker = createMessagingToolTracker();
  const provider = params.provider ?? CLAUDE_SDK_PROVIDER;
  const requestedModel = (params.model ?? "opus").trim();
  const sdkModel = resolveClaudeSdkModel(requestedModel);
  const messageProvider = params.messageProvider ?? params.messageChannel;

  // Create clawdbot tools using existing factory
  const codingTools = createOpenClawCodingTools({
    sessionKey: params.sessionKey,
    messageProvider,
    agentAccountId: params.agentAccountId,
    messageTo: params.messageTo,
    messageThreadId: params.messageThreadId,
    currentChannelId: params.currentChannelId,
    currentThreadTs: params.currentThreadTs,
    replyToMode: params.replyToMode,
    hasRepliedRef: params.hasRepliedRef,
    agentDir: params.agentDir,
    workspaceDir: params.workspaceDir,
    config: params.config,
    abortSignal: params.abortSignal,
    exec: params.execOverrides,
    modelProvider: provider,
    modelId: requestedModel,
  });

  // Create MCP server with all tools
  const clawdbotMcp = createClawdbotMcpServerWithTools(codingTools, messagingTracker);

  // Build allowed tools list (prefixed with mcp__clawdbot__)
  const allowedTools = codingTools.map((t: AnyAgentTool) => `mcp__clawdbot__${t.name}`);

  const abortController = new AbortController();
  const abortListener = () => abortController.abort();
  if (params.abortSignal) {
    if (params.abortSignal.aborted) {
      abortController.abort();
    } else {
      params.abortSignal.addEventListener("abort", abortListener, { once: true });
    }
  }
  const timeout = setTimeout(() => abortController.abort(), params.timeoutMs);

  let partialBuffer = "";
  let sawPartialText = false;
  let assistantStopReason: string | undefined;
  let resultText: string | undefined;
  let resultError: string | undefined;
  let usageMeta: EmbeddedPiAgentMeta["usage"] | undefined;
  let finalTextChunks: string[] = [];
  let didStartAssistant = false;

  const shouldEmitToolOutput = params.shouldEmitToolOutput?.() ?? true;
  const shouldEmitToolResult = params.shouldEmitToolResult?.() ?? true;

  try {
    const sessionHistoryPrompt = await buildSessionHistoryPrompt({
      sessionFile: params.sessionFile,
    });
    const promptParts: string[] = [];
    const effectiveWorkspace = params.workspaceDir?.trim() || process.cwd();
    const skillsPrompt = params.skillsSnapshot?.prompt?.trim();

    const { defaultAgentId, sessionAgentId } = resolveSessionAgentIds({
      sessionKey: params.sessionKey,
      config: params.config,
    });
    const runtimeChannel = normalizeMessageChannel(params.messageChannel ?? params.messageProvider);
    let runtimeCapabilities = runtimeChannel
      ? (resolveChannelCapabilities({
          cfg: params.config,
          channel: runtimeChannel,
          accountId: params.agentAccountId,
        }) ?? [])
      : undefined;
    if (runtimeChannel === "telegram" && params.config) {
      const inlineButtonsScope = resolveTelegramInlineButtonsScope({
        cfg: params.config,
        accountId: params.agentAccountId ?? undefined,
      });
      if (inlineButtonsScope !== "off") {
        if (!runtimeCapabilities) runtimeCapabilities = [];
        if (
          !runtimeCapabilities.some((cap) => String(cap).trim().toLowerCase() === "inlinebuttons")
        )
          runtimeCapabilities.push("inlineButtons");
      }
    }
    const reactionGuidance =
      runtimeChannel === "telegram" && params.config
        ? (() => {
            const resolved = resolveTelegramReactionLevel({
              cfg: params.config,
              accountId: params.agentAccountId ?? undefined,
            });
            const level = resolved.agentReactionGuidance;
            return level ? { level, channel: "Telegram" } : undefined;
          })()
        : undefined;
    const channelActions = runtimeChannel
      ? listChannelSupportedActions({
          cfg: params.config,
          channel: runtimeChannel,
        })
      : undefined;

    const reasoningTagHint = isReasoningTagProvider(provider);
    const machineName = await getMachineDisplayName();
    const defaultModelRef = resolveDefaultModelForAgent({
      cfg: params.config ?? {},
      agentId: sessionAgentId,
    });
    const defaultModelLabel = `${defaultModelRef.provider}/${defaultModelRef.model}`;
    const { runtimeInfo, userTimezone, userTime, userTimeFormat } = buildSystemPromptParams({
      config: params.config,
      agentId: sessionAgentId,
      runtime: {
        host: machineName,
        os: `${os.type()} ${os.release()}`,
        arch: os.arch(),
        node: process.version,
        model: `${provider}/${sdkModel}`,
        defaultModel: defaultModelLabel,
        channel: runtimeChannel,
        capabilities: runtimeCapabilities,
        channelActions,
      },
    });

    const { contextFiles } = await resolveBootstrapContextForRun({
      workspaceDir: effectiveWorkspace,
      config: params.config,
      sessionKey: params.sessionKey,
      sessionId: params.sessionId,
      agentId: sessionAgentId,
    });
    const docsPath = await resolveOpenClawDocsPath({
      workspaceDir: effectiveWorkspace,
      argv1: process.argv[1],
      cwd: process.cwd(),
      moduleUrl: import.meta.url,
    });
    const promptMode = isSubagentSessionKey(params.sessionKey) ? "minimal" : "full";
    const isDefaultAgent = sessionAgentId === defaultAgentId;

    const appendPrompt = buildEmbeddedSystemPrompt({
      workspaceDir: effectiveWorkspace,
      defaultThinkLevel: params.thinkLevel,
      reasoningLevel: params.reasoningLevel ?? "off",
      extraSystemPrompt: params.extraSystemPrompt,
      ownerNumbers: params.ownerNumbers,
      reasoningTagHint,
      heartbeatPrompt: isDefaultAgent
        ? resolveHeartbeatPrompt(params.config?.agents?.defaults?.heartbeat?.prompt)
        : undefined,
      skillsPrompt,
      docsPath: docsPath ?? undefined,
      reactionGuidance,
      promptMode,
      runtimeInfo,
      sandboxInfo: undefined,
      tools: codingTools,
      modelAliasLines: buildModelAliasLines(params.config),
      userTimezone,
      userTime,
      userTimeFormat,
      contextFiles,
    });
    if (appendPrompt.trim()) promptParts.push(appendPrompt);
    if (sessionHistoryPrompt) promptParts.push(sessionHistoryPrompt);

    const systemPrompt =
      promptParts.length > 0
        ? {
            type: "preset" as const,
            preset: "claude_code" as const,
            append: promptParts.join("\n\n"),
          }
        : undefined;

    let prompt = params.prompt;
    if (params.images?.length) {
      prompt += `\n\n[${params.images.length} image(s) attached]`;
    }

    // Run SDK query with streaming
    for await (const message of query({
      prompt,
      options: {
        cwd: params.workspaceDir ?? process.cwd(),
        model: sdkModel,
        systemPrompt,
        permissionMode: "acceptEdits",
        includePartialMessages: true,
        tools: [],
        mcpServers: {
          clawdbot: clawdbotMcp,
        },
        allowedTools,
        abortController,
        env: { ...process.env }, // Inherits ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN from shell
      },
    })) {
      const msgType = (message as { type?: string }).type;

      if (msgType === "stream_event") {
        const event = (message as { event?: Record<string, unknown> }).event;
        if (event && typeof event.type === "string") {
          if (event.type === "message_start" && !didStartAssistant) {
            didStartAssistant = true;
            await params.onAssistantMessageStart?.();
          }
          if (event.type === "message_delta") {
            const stopReason = (event as { delta?: { stop_reason?: string } }).delta?.stop_reason;
            if (stopReason) {
              assistantStopReason = stopReason;
            }
          }
          if (event.type === "content_block_delta") {
            const delta = (event as { delta?: { type?: string; text?: string; thinking?: string } })
              .delta;
            if (delta?.type === "text_delta" && typeof delta.text === "string") {
              partialBuffer += delta.text;
              sawPartialText = true;
              await params.onPartialReply?.({ text: delta.text });
            }
            if (delta?.type === "thinking_delta" && typeof delta.thinking === "string") {
              await params.onReasoningStream?.({ text: delta.thinking });
            }
          }
          if (event.type === "content_block_start") {
            const block = (
              event as { content_block?: { type?: string; name?: string; id?: string } }
            ).content_block;
            if (
              block &&
              (block.type === "tool_use" ||
                block.type === "mcp_tool_use" ||
                block.type === "server_tool_use")
            ) {
              params.onAgentEvent?.({
                stream: "tool",
                data: {
                  phase: "start",
                  name: block.name,
                  toolCallId: block.id,
                },
              });
            }
          }
        }
      }

      // Handle assistant messages with content blocks
      if (msgType === "assistant") {
        if (!didStartAssistant) {
          didStartAssistant = true;
          await params.onAssistantMessageStart?.();
        }
        const msg = (message as { message?: { content?: unknown; stop_reason?: string } }).message;
        const content = msg?.content;
        if (msg?.stop_reason) {
          assistantStopReason = msg.stop_reason;
        }

        if (typeof content === "string") {
          finalTextChunks = [content];
        } else if (Array.isArray(content)) {
          const chunks: string[] = [];
          for (const block of content) {
            if (
              block &&
              typeof block === "object" &&
              "type" in block &&
              block.type === "text" &&
              "text" in block
            ) {
              chunks.push(block.text as string);
            }
            // Handle thinking/reasoning blocks
            if (
              block &&
              typeof block === "object" &&
              "type" in block &&
              block.type === "thinking" &&
              "thinking" in block
            ) {
              await params.onReasoningStream?.({ text: block.thinking as string });
            }
            if (
              block &&
              typeof block === "object" &&
              "type" in block &&
              (block.type === "tool_use" ||
                block.type === "mcp_tool_use" ||
                block.type === "server_tool_use")
            ) {
              params.onAgentEvent?.({
                stream: "tool",
                data: {
                  phase: "start",
                  name: (block as { name?: string }).name,
                  toolCallId: (block as { id?: string }).id,
                },
              });
            }
          }
          if (chunks.length > 0) {
            finalTextChunks = chunks;
          }
        }
      }

      // Handle final result (success or error)
      if (msgType === "result") {
        const result = message as {
          subtype?: string;
          result?: string;
          errors?: string[];
          usage?: {
            input_tokens?: number;
            output_tokens?: number;
            cache_read_input_tokens?: number;
            cache_creation_input_tokens?: number;
          };
        };
        if (result.subtype === "success") {
          if (result.result) {
            resultText = result.result;
          }
        } else {
          resultError = (result.errors ?? []).join("\n") || "SDK execution error";
        }
        if (result.usage) {
          const input = result.usage.input_tokens;
          const output = result.usage.output_tokens;
          const cacheRead = result.usage.cache_read_input_tokens;
          const cacheWrite = result.usage.cache_creation_input_tokens;
          const total = (input ?? 0) + (output ?? 0) + (cacheRead ?? 0) + (cacheWrite ?? 0);
          usageMeta = {
            input,
            output,
            cacheRead,
            cacheWrite,
            total,
          };
        }
      }

      // Handle tool_result messages (legacy format)
      if (msgType === "tool_result") {
        const toolName = (message as { tool_name?: string }).tool_name ?? "unknown";
        if (params.onToolResult && shouldEmitToolOutput && shouldEmitToolResult) {
          await params.onToolResult({ text: `Tool ${toolName} completed` });
        }
      }

      // Handle user messages containing tool_result (SDK format)
      if (msgType === "user") {
        const userContent = (message as { message?: { content?: unknown[] } }).message?.content;
        if (Array.isArray(userContent)) {
          for (const block of userContent) {
            if (
              block &&
              typeof block === "object" &&
              "type" in block &&
              block.type === "tool_result"
            ) {
              const toolUseId = (block as { tool_use_id?: string }).tool_use_id ?? "unknown";
              if (params.onToolResult && shouldEmitToolOutput && shouldEmitToolResult) {
                await params.onToolResult({ text: `Tool ${toolUseId} completed` });
              }
              params.onAgentEvent?.({
                stream: "tool",
                data: {
                  phase: "result",
                  toolCallId: toolUseId,
                },
              });
            }
          }
        }
      }
    }

    const finalText =
      finalTextChunks.length > 0
        ? finalTextChunks.join("")
        : (resultText ?? (sawPartialText ? partialBuffer : undefined));
    if (finalText) {
      payloads.push({ text: finalText, isError: Boolean(resultError) });
    }

    // Build result metadata
    const agentMeta: EmbeddedPiAgentMeta = {
      sessionId: params.sessionId,
      provider,
      model: requestedModel || sdkModel,
      usage: usageMeta,
    };

    const meta: EmbeddedPiRunMeta = {
      durationMs: Date.now() - startTime,
      agentMeta,
      stopReason: assistantStopReason ?? (resultError ? "error" : "completed"),
      error: resultError
        ? {
            kind: "compaction_failure",
            message: resultError,
          }
        : undefined,
    };

    try {
      await appendSessionTurns({
        sessionFile: params.sessionFile,
        sessionId: params.sessionId,
        cwd: params.workspaceDir ?? process.cwd(),
        userText: params.prompt,
        assistantText: finalText,
        provider,
        model: requestedModel || sdkModel,
        stopReason: assistantStopReason,
        usage: usageMeta,
      });
    } catch {
      // Best-effort session persistence.
    }

    return {
      payloads: payloads.length > 0 ? payloads : undefined,
      meta,
      didSendViaMessagingTool: messagingTracker.didSend,
      messagingToolSentTexts:
        messagingTracker.sentTexts.length > 0 ? messagingTracker.sentTexts : undefined,
      messagingToolSentTargets:
        messagingTracker.sentTargets.length > 0
          ? messagingTracker.sentTargets.map(
              (t): MessagingToolSend => ({
                tool: t.channel,
                provider: t.channel,
                to: t.to,
              }),
            )
          : undefined,
    };
  } catch (error) {
    // Handle abort
    if (abortController.signal.aborted || params.abortSignal?.aborted) {
      return {
        payloads: payloads.length > 0 ? payloads : undefined,
        meta: {
          durationMs: Date.now() - startTime,
          aborted: true,
        },
      };
    }

    // Handle other errors
    const errorMessage = error instanceof Error ? error.message : String(error);
    const isContextOverflow = isContextOverflowError(errorMessage);
    const isCompactionFailure = isCompactionFailureError(errorMessage);
    const isRoleOrdering = /role order|roles must alternate|incorrect role/i.test(errorMessage);

    const meta: EmbeddedPiRunMeta = {
      durationMs: Date.now() - startTime,
      error: {
        kind: isContextOverflow
          ? "context_overflow"
          : isRoleOrdering
            ? "role_ordering"
            : isCompactionFailure
              ? "compaction_failure"
              : "compaction_failure",
        message: errorMessage,
      },
    };

    return {
      payloads:
        payloads.length > 0 ? payloads : [{ text: `Error: ${errorMessage}`, isError: true }],
      meta,
    };
  } finally {
    clearTimeout(timeout);
    if (params.abortSignal) {
      params.abortSignal.removeEventListener("abort", abortListener);
    }
  }
}

function resolveClaudeSdkModel(raw?: string): string {
  let normalized = (raw ?? "").trim().toLowerCase();
  if (!normalized) return "opus";
  if (normalized.includes("/")) {
    normalized = normalized.split("/").pop() ?? normalized;
  }
  if (normalized.startsWith("claude-opus")) return "opus";
  if (normalized.startsWith("claude-sonnet")) return "sonnet";
  if (normalized.startsWith("claude-haiku")) return "haiku";
  const alias = CLAUDE_SDK_MODEL_ALIASES.get(normalized);
  if (alias) return alias;
  if (CLAUDE_SDK_MODELS.has(normalized)) return normalized;
  throw new Error(
    `Unsupported ${CLAUDE_SDK_PROVIDER} model "${raw}". Use "opus", "sonnet", or "haiku".`,
  );
}

// ============================================================================
// Test Helper
// ============================================================================

/**
 * Quick test to verify the adapter works with full signature.
 * Usage: bun vendor/clawdbot/src/agents/claude-sdk-runner/adapter.ts
 */
async function testAdapter() {
  console.log("🧪 Testing Claude SDK Adapter with full signature...\n");

  const testParams: RunEmbeddedPiAgentParams = {
    sessionId: `test-${Date.now()}`,
    sessionFile: "/tmp/test-session.jsonl",
    workspaceDir: process.cwd(),
    prompt: "Say 'Adapter test successful!' in exactly those words.",
    timeoutMs: 60000,
    runId: `run-${Date.now()}`,
    onPartialReply: (payload) => {
      console.log("📝 Partial:", payload.text?.slice(0, 100));
    },
    onToolResult: (payload) => {
      console.log("🔧 Tool result:", payload.text);
    },
    onAgentEvent: (evt) => {
      console.log("📡 Event:", evt.stream, evt.data);
    },
  };

  console.log("--- Test 1: Basic query with callbacks ---");
  const result = await runClaudeSdkAgent(testParams);

  console.log("\n✅ Result:");
  console.log("  - Payloads:", result.payloads?.length ?? 0);
  console.log("  - Duration:", result.meta.durationMs, "ms");
  console.log("  - Error:", result.meta.error?.message ?? "none");

  if (result.payloads?.length) {
    console.log("\n💬 Response:");
    console.log(
      result.payloads
        .map((p) => p.text)
        .join("\n")
        .slice(0, 500),
    );
  }

  const passed = (result.payloads?.length ?? 0) > 0 && !result.meta.error;
  console.log(`\n${passed ? "✅ PASS" : "❌ FAIL"}`);
  process.exitCode = passed ? 0 : 1;
}

// Run test if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  testAdapter().catch(console.error);
}
