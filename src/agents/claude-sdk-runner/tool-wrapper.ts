/**
 * Claude SDK tool wrapper helpers.
 */
import crypto from "node:crypto";

import { z } from "zod";
import {
  tool,
  createSdkMcpServer,
  type McpSdkServerConfigWithInstance,
} from "@anthropic-ai/claude-agent-sdk";

import type { AnyAgentTool } from "../pi-tools.types.js";

// CallToolResult type (SDK doesn't export this directly)
type ToolResultContent =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string }
  | { type: "resource"; resource: { uri: string; text?: string; blob?: string } };

type CallToolResult = {
  content: ToolResultContent[];
  isError?: boolean;
};

export type MessagingToolTracker = {
  didSend: boolean;
  sentTexts: string[];
  sentTargets: Array<{ channel: string; to: string }>;
};

/**
 * Create a fresh messaging tracker for SDK tool runs.
 */
export function createMessagingToolTracker(): MessagingToolTracker {
  return {
    didSend: false,
    sentTexts: [],
    sentTargets: [],
  };
}

/**
 * Convert TypeBox-like schemas to a Zod raw shape (best-effort).
 */
function typeBoxToZod(schema: unknown): Record<string, z.ZodTypeAny> {
  if (!schema || typeof schema !== "object") return {};

  const tbSchema = schema as {
    type?: string;
    properties?: Record<
      string,
      {
        type?: string;
        description?: string;
        enum?: unknown[];
        items?: unknown;
        anyOf?: unknown[];
        oneOf?: unknown[];
      }
    >;
    required?: string[];
  };

  if (tbSchema.type !== "object" || !tbSchema.properties) return {};

  const required = new Set(tbSchema.required ?? []);
  const zodProps: Record<string, z.ZodTypeAny> = {};

  const buildFromProp = (prop: {
    type?: string;
    description?: string;
    enum?: unknown[];
    items?: unknown;
    anyOf?: unknown[];
    oneOf?: unknown[];
  }): z.ZodTypeAny => {
    if (Array.isArray(prop.enum) && prop.enum.length > 0) {
      const literalValues = prop.enum.filter(
        (value): value is string | number | boolean | null =>
          typeof value === "string" ||
          typeof value === "number" ||
          typeof value === "boolean" ||
          value === null,
      );
      if (literalValues.length === 0) return z.unknown();
      const literals = literalValues.map((value) => z.literal(value));
      if (literals.length === 1) return literals[0];
      return z.union(literals);
    }

    if (Array.isArray(prop.anyOf) && prop.anyOf.length > 0) {
      const variants = prop.anyOf.map((item) => buildFromProp(item as typeof prop));
      if (variants.length === 1) return variants[0];
      return z.union(variants);
    }

    if (Array.isArray(prop.oneOf) && prop.oneOf.length > 0) {
      const variants = prop.oneOf.map((item) => buildFromProp(item as typeof prop));
      if (variants.length === 1) return variants[0];
      return z.union(variants);
    }

    switch (prop.type) {
      case "string":
        return z.string();
      case "number":
      case "integer":
        return z.number();
      case "boolean":
        return z.boolean();
      case "array":
        return z.array(z.unknown());
      case "object":
        return z.record(z.string(), z.unknown());
      default:
        return z.unknown();
    }
  };

  for (const [key, prop] of Object.entries(tbSchema.properties)) {
    let zodType = buildFromProp(prop);
    if (prop.description) {
      zodType = zodType.describe(prop.description);
    }
    zodProps[key] = required.has(key) ? zodType : zodType.optional();
  }

  return zodProps;
}

/**
 * Wrap a clawdbot AgentTool as an SDK MCP tool.
 */
export function wrapAgentToolForSdk(
  agentTool: AnyAgentTool,
  tracker: MessagingToolTracker,
): ReturnType<typeof tool> {
  const zodSchema = typeBoxToZod(agentTool.parameters);

  return tool(
    agentTool.name,
    agentTool.description,
    zodSchema,
    async (args: Record<string, unknown>): Promise<CallToolResult> => {
      const toolCallId = crypto.randomUUID();
      const result = await agentTool.execute(toolCallId, args);
      const resultRecord = result as { isError?: boolean };

      // Track messaging tool sends
      if (
        agentTool.name === "sessions_send" ||
        agentTool.name.includes("telegram") ||
        agentTool.name.includes("whatsapp") ||
        agentTool.name.includes("discord") ||
        agentTool.name.includes("slack")
      ) {
        tracker.didSend = true;
        if (typeof args.message === "string") {
          tracker.sentTexts.push(args.message);
        }
        if (typeof args.channel === "string" && typeof args.to === "string") {
          tracker.sentTargets.push({ channel: args.channel, to: args.to });
        }
      }

      // Convert AgentToolResult -> CallToolResult
      const rawContent = Array.isArray(result.content) ? result.content : [];
      const content = rawContent.map((item) => {
        const record = item as unknown as Record<string, unknown>;
        if (record.type === "text" && typeof record.text === "string") {
          return { type: "text" as const, text: record.text };
        }
        if (
          record.type === "image" &&
          typeof record.data === "string" &&
          typeof record.mimeType === "string"
        ) {
          return {
            type: "image" as const,
            data: record.data as string,
            mimeType: record.mimeType as string,
          };
        }
        if (record.type === "resource" && record.resource && typeof record.resource === "object") {
          return {
            type: "resource" as const,
            resource: record.resource as { uri: string; text?: string; blob?: string },
          };
        }
        return { type: "text" as const, text: JSON.stringify(item) };
      });

      const textHasError = content.some(
        (c) => c.type === "text" && c.text?.toLowerCase().includes("error"),
      );

      return {
        content,
        isError: resultRecord.isError ?? textHasError,
      };
    },
  );
}

/**
 * Create an in-process MCP server with all clawdbot coding tools.
 */
export function createClawdbotMcpServerWithTools(
  tools: AnyAgentTool[],
  tracker: MessagingToolTracker,
): McpSdkServerConfigWithInstance {
  const sdkTools = tools.map((toolDef) => wrapAgentToolForSdk(toolDef, tracker));

  return createSdkMcpServer({
    name: "clawdbot-gateway",
    version: "0.1.0",
    tools: sdkTools,
  });
}
