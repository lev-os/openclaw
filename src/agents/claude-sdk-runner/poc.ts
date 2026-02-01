/**
 * POC: Claude Agent SDK Adapter
 *
 * This is a minimal proof-of-concept for replacing pi-agent with Claude Agent SDK.
 * Goal: Wrap sessions_send tool and test round-trip message flow.
 *
 * @see clawd-fbtq (BD epic)
 * @see .good/clawdbot-pi-integration-research.md
 */

import { z } from "zod";
import {
  query,
  tool,
  createSdkMcpServer,
  type McpSdkServerConfigWithInstance,
} from "@anthropic-ai/claude-agent-sdk";

// CallToolResult type (from @modelcontextprotocol/sdk/types.js)
type ToolResultContent =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string }
  | { type: "resource"; resource: { uri: string; text?: string; blob?: string } };

type CallToolResult = {
  content: ToolResultContent[];
  isError?: boolean;
};

import { createSessionsSendTool } from "../tools/sessions-send-tool.js";
import type { EmbeddedPiRunResult, EmbeddedPiRunMeta } from "../pi-embedded-runner/types.js";

// ============================================================================
// Zod Schemas (manually matching TypeBox schemas from sessions-send-tool.ts)
// ============================================================================

const SessionsSendSchema = {
  sessionKey: z.string().optional(),
  label: z.string().min(1).max(100).optional(),
  agentId: z.string().min(1).max(64).optional(),
  message: z.string(),
  timeoutSeconds: z.number().min(0).optional(),
};

// ============================================================================
// Tool Wrapper: AgentTool → SdkMcpToolDefinition
// ============================================================================

type AnyAgentTool = {
  name: string;
  description: string;
  parameters: unknown;
  execute: (
    id: string,
    params: Record<string, unknown>,
  ) => Promise<{
    content: Array<{ type: string; text?: string; data?: string; mimeType?: string }>;
    details?: unknown;
  }>;
};

/**
 * Wrap a clawdbot AgentTool as an SDK MCP tool.
 * For POC, we manually specify the Zod schema.
 */
function wrapSessionsSendTool(agentTool: AnyAgentTool, zodSchema: Record<string, z.ZodTypeAny>) {
  return tool(
    agentTool.name,
    agentTool.description,
    zodSchema,
    async (args: Record<string, unknown>): Promise<CallToolResult> => {
      const toolCallId = crypto.randomUUID();
      const result = await agentTool.execute(toolCallId, args);

      // Convert AgentToolResult → CallToolResult
      return {
        content: result.content.map((item) => {
          if (item.type === "text" && item.text) {
            return { type: "text" as const, text: item.text };
          }
          if (item.type === "image" && item.data && item.mimeType) {
            return {
              type: "image" as const,
              data: item.data,
              mimeType: item.mimeType,
            };
          }
          // Fallback: convert to text
          return { type: "text" as const, text: JSON.stringify(item) };
        }),
      };
    },
  );
}

// ============================================================================
// MCP Server Factory
// ============================================================================

export interface ClawdbotMcpServerOptions {
  agentSessionKey?: string;
  agentChannel?: string;
  sandboxed?: boolean;
}

/**
 * Create an in-process MCP server with clawdbot gateway tools.
 * For POC, only includes sessions_send.
 */
export function createClawdbotMcpServer(
  options?: ClawdbotMcpServerOptions,
): McpSdkServerConfigWithInstance {
  // Create the original AgentTool
  const sessionsSendAgentTool = createSessionsSendTool({
    agentSessionKey: options?.agentSessionKey,
    agentChannel: options?.agentChannel as any,
    sandboxed: options?.sandboxed,
  });

  // Wrap as SDK MCP tool
  const sessionsSendMcpTool = wrapSessionsSendTool(
    sessionsSendAgentTool as AnyAgentTool,
    SessionsSendSchema,
  );

  // Bundle into MCP server
  return createSdkMcpServer({
    name: "clawdbot-gateway",
    version: "0.1.0",
    tools: [sessionsSendMcpTool],
  });
}

// ============================================================================
// SDK Runner (main entry point)
// ============================================================================

export interface RunClaudeSdkAgentParams {
  prompt: string;
  systemPrompt?: string;
  sessionKey?: string;
  model?: string;
  workspaceDir?: string;
  onPartialReply?: (msg: unknown) => void;
  onToolResult?: (toolName: string, result: unknown) => void;
}

/**
 * Run an agent turn using Claude Agent SDK instead of pi-agent.
 * Returns EmbeddedPiRunResult for compatibility with existing clawdbot code.
 */
export async function runClaudeSdkAgent(
  params: RunClaudeSdkAgentParams,
): Promise<EmbeddedPiRunResult> {
  const startTime = Date.now();
  const payloads: EmbeddedPiRunResult["payloads"] = [];

  // Create MCP server with clawdbot tools
  const clawdbotMcp = createClawdbotMcpServer({
    agentSessionKey: params.sessionKey,
  });

  try {
    // Run SDK query
    for await (const message of query({
      prompt: params.prompt,
      options: {
        cwd: params.workspaceDir ?? process.cwd(),
        model: params.model ?? "claude-sonnet-4-20250514",
        systemPrompt: params.systemPrompt,
        mcpServers: {
          clawdbot: clawdbotMcp,
        },
        // Allow our custom tools
        allowedTools: ["mcp__clawdbot__sessions_send"],
      },
    })) {
      // Handle different message types
      const msgType = (message as any).type;
      if (msgType === "text") {
        payloads.push({ text: (message as any).content });
        params.onPartialReply?.(message);
      } else if (msgType === "assistant") {
        // SDK uses "assistant" with nested message.content
        const msg = (message as any).message;
        const content = msg?.content;
        if (typeof content === "string") {
          payloads.push({ text: content });
        } else if (Array.isArray(content)) {
          for (const block of content) {
            if (block.type === "text") {
              payloads.push({ text: block.text });
            }
          }
        }
      } else if (msgType === "result") {
        // Final result message
        const resultText = (message as any).result;
        if (resultText && !payloads.some((p) => p.text === resultText)) {
          payloads.push({ text: resultText });
        }
      } else if (msgType === "tool_result") {
        params.onToolResult?.((message as any).tool_name ?? "unknown", message);
      } else if (msgType === "user") {
        // SDK embeds tool_result in user messages
        const userContent = (message as any).message?.content;
        if (Array.isArray(userContent)) {
          for (const block of userContent) {
            if (block.type === "tool_result") {
              params.onToolResult?.(block.tool_use_id ?? "unknown", block);
            }
          }
        }
      }
    }

    const meta: EmbeddedPiRunMeta = {
      durationMs: Date.now() - startTime,
      agentMeta: {
        sessionId: params.sessionKey ?? `sdk-${Date.now()}`,
        provider: "anthropic",
        model: params.model ?? "claude-sonnet-4-20250514",
      },
      stopReason: "completed",
    };

    return { payloads, meta };
  } catch (error) {
    const meta: EmbeddedPiRunMeta = {
      durationMs: Date.now() - startTime,
      error: {
        kind: "compaction_failure", // Generic error bucket
        message: error instanceof Error ? error.message : String(error),
      },
    };

    return { payloads, meta };
  }
}

// ============================================================================
// Test Helper
// ============================================================================

/**
 * Quick test to verify the adapter works.
 * Usage: bun vendor/clawdbot/src/agents/claude-sdk-runner/poc.ts
 */
async function testPoc() {
  console.log("🧪 Testing Claude SDK Adapter POC...\n");

  // Test 1: Basic query (no tool use)
  console.log("--- Test 1: Basic query ---");
  const result1 = await runClaudeSdkAgent({
    prompt: "Say 'POC test successful!' in exactly those words.",
    workspaceDir: process.cwd(),
  });
  console.log("✅ Basic query:", result1.payloads?.length ? "got response" : "no response");

  // Test 2: Tool listing (verify our MCP tool is visible)
  console.log("\n--- Test 2: Tool visibility ---");
  const result2 = await runClaudeSdkAgent({
    prompt: "Do you have access to a tool called sessions_send? Just answer yes or no.",
    workspaceDir: process.cwd(),
  });
  const hasSessionsSend = result2.payloads?.some(
    (p) => p.text?.toLowerCase().includes("yes") || p.text?.toLowerCase().includes("sessions_send"),
  );
  console.log("✅ sessions_send visible:", hasSessionsSend ? "YES" : "NO");

  // Test 3: Actual tool invocation
  console.log("\n--- Test 3: Tool invocation ---");
  let toolWasCalled = false;
  let toolCallDetails: { name: string; args?: unknown; result?: unknown } | null = null;

  const result3 = await runClaudeSdkAgent({
    prompt: `You have access to a sessions_send tool. Please use it RIGHT NOW to send a test message.

Call the tool with these exact parameters:
- message: "🤖 SDK POC test - ${new Date().toISOString()}"
- sessionKey: "main"

After calling the tool, just say "Tool called successfully" or report any error.`,
    workspaceDir: process.cwd(),
    onToolResult: (toolName, result) => {
      console.log(`  📡 Tool called: ${toolName}`);
      toolWasCalled = true;
      toolCallDetails = { name: toolName, result };
    },
  });

  console.log("✅ Tool invocation:", toolWasCalled ? "CALLED" : "NOT CALLED");
  if (toolCallDetails) {
    console.log("  Tool details:", JSON.stringify(toolCallDetails, null, 2).slice(0, 500));
  }

  // Summary
  console.log("\n📦 Summary:");
  console.log("- Test 1 (basic):", result1.payloads?.length ? "PASS" : "FAIL");
  console.log("- Test 2 (tool):", hasSessionsSend ? "PASS" : "FAIL");
  console.log("- Test 3 (invoke):", toolWasCalled ? "PASS" : "FAIL");
  console.log(
    "- Total duration:",
    result1.meta.durationMs + result2.meta.durationMs + result3.meta.durationMs,
    "ms",
  );

  if (result1.payloads?.length) {
    console.log("\n💬 Response 1:");
    console.log(
      result1.payloads
        .map((p) => p.text)
        .join("\n")
        .slice(0, 500),
    );
  }
  if (result2.payloads?.length) {
    console.log("\n💬 Response 2:");
    console.log(
      result2.payloads
        .map((p) => p.text)
        .join("\n")
        .slice(0, 500),
    );
  }
  if (result3.payloads?.length) {
    console.log("\n💬 Response 3:");
    console.log(
      result3.payloads
        .map((p) => p.text)
        .join("\n")
        .slice(0, 500),
    );
  }

  // Return exit code
  const allPassed = result1.payloads?.length && hasSessionsSend && toolWasCalled;
  process.exitCode = allPassed ? 0 : 1;
}

// Run test if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  testPoc().catch(console.error);
}
