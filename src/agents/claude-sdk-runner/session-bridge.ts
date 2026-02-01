/**
 * Session bridge helpers for Claude SDK runs.
 */

import fs from "node:fs/promises";

import { ensureSessionHeader } from "../pi-embedded-helpers/bootstrap.js";
import { acquireSessionWriteLock } from "../session-write-lock.js";

type SessionMessageEntry = {
  type?: string;
  message?: {
    role?: string;
    content?: unknown;
  };
};

type SessionHistoryOptions = {
  sessionFile: string;
  maxMessages?: number;
  maxChars?: number;
  maxMessageChars?: number;
};

type SessionAppendParams = {
  sessionFile: string;
  sessionId: string;
  cwd: string;
  userText: string;
  assistantText?: string;
  provider?: string;
  model?: string;
  stopReason?: string;
  usage?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
    total?: number;
  };
};

const DEFAULT_HISTORY_LIMIT = 20;
const DEFAULT_HISTORY_MAX_CHARS = 8000;
const DEFAULT_MESSAGE_MAX_CHARS = 2000;

function extractTextFromContent(content: unknown): string | null {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return null;
  const blocks = content
    .map((block) => {
      if (!block || typeof block !== "object") return null;
      const rec = block as { type?: unknown; text?: unknown };
      if (rec.type !== "text") return null;
      return typeof rec.text === "string" ? rec.text : null;
    })
    .filter((text): text is string => Boolean(text));
  if (blocks.length === 0) return null;
  return blocks.join("\n");
}

function trimText(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}...`;
}

/**
 * Build a session history prompt from a JSONL session transcript.
 */
export async function buildSessionHistoryPrompt(
  opts: SessionHistoryOptions,
): Promise<string | null> {
  const maxMessages = opts.maxMessages ?? DEFAULT_HISTORY_LIMIT;
  const maxChars = opts.maxChars ?? DEFAULT_HISTORY_MAX_CHARS;
  const maxMessageChars = opts.maxMessageChars ?? DEFAULT_MESSAGE_MAX_CHARS;

  let raw: string;
  try {
    raw = await fs.readFile(opts.sessionFile, "utf-8");
  } catch {
    return null;
  }

  const entries = raw
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line) as SessionMessageEntry;
      } catch {
        return null;
      }
    })
    .filter((entry): entry is SessionMessageEntry => Boolean(entry))
    .filter((entry) => entry.type === "message")
    .map((entry) => entry.message)
    .filter((message): message is NonNullable<SessionMessageEntry["message"]> => Boolean(message))
    .filter((message) => message.role === "user" || message.role === "assistant");

  if (entries.length === 0) return null;

  const recent = entries.slice(-maxMessages);
  const lines: string[] = [];
  let totalChars = 0;

  for (const message of recent) {
    const text = extractTextFromContent(message.content);
    if (!text) continue;
    const trimmed = trimText(text, maxMessageChars);
    const prefix = message.role === "assistant" ? "Assistant" : "User";
    const line = `${prefix}: ${trimmed}`;
    if (totalChars + line.length > maxChars) break;
    lines.push(line);
    totalChars += line.length;
  }

  if (lines.length === 0) return null;

  return ["Session history (most recent last):", ...lines].join("\n");
}

/**
 * Append the latest user/assistant turns to a JSONL session transcript.
 */
export async function appendSessionTurns(params: SessionAppendParams): Promise<void> {
  if (!params.userText && !params.assistantText) return;
  await ensureSessionHeader({
    sessionFile: params.sessionFile,
    sessionId: params.sessionId,
    cwd: params.cwd,
  });

  const lock = await acquireSessionWriteLock({ sessionFile: params.sessionFile });
  try {
    const now = Date.now();
    const entries: Array<Record<string, unknown>> = [];

    if (params.userText) {
      entries.push({
        type: "message",
        message: {
          role: "user",
          content: [{ type: "text", text: params.userText }],
          timestamp: now,
        },
      });
    }

    if (params.assistantText) {
      const hasUsage =
        params.usage &&
        (params.usage.input ||
          params.usage.output ||
          params.usage.cacheRead ||
          params.usage.cacheWrite);
      const usage = hasUsage
        ? {
            input: params.usage?.input ?? 0,
            output: params.usage?.output ?? 0,
            cacheRead: params.usage?.cacheRead ?? 0,
            cacheWrite: params.usage?.cacheWrite ?? 0,
            totalTokens:
              params.usage?.total ??
              (params.usage?.input ?? 0) +
                (params.usage?.output ?? 0) +
                (params.usage?.cacheRead ?? 0) +
                (params.usage?.cacheWrite ?? 0),
            cost: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              total: 0,
            },
          }
        : undefined;

      entries.push({
        type: "message",
        message: {
          role: "assistant",
          content: [{ type: "text", text: params.assistantText }],
          stopReason: params.stopReason ?? "stop",
          provider: params.provider ?? "anthropic",
          model: params.model ?? "unknown",
          usage,
          timestamp: now,
        },
      });
    }

    const payload = `${entries.map((entry) => JSON.stringify(entry)).join("\n")}\n`;
    await fs.appendFile(params.sessionFile, payload, "utf-8");
  } finally {
    await lock.release();
  }
}
