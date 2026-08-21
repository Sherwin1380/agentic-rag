const COMPLETE_PRIVATE_BLOCK =
  /\\?<\s*(think|tool_call|function)\b[^>]*>[\s\S]*?\\?<\s*\/\s*\1\s*>/gi;
const ENCODED_PRIVATE_BLOCK =
  /&lt;\s*(think|tool_call|function)\b[\s\S]*?&gt;[\s\S]*?&lt;\s*\/\s*\1\s*&gt;/gi;
const UNCLOSED_PRIVATE_BLOCK =
  /(?:\\?<|&lt;)\s*(?:think|tool_call|function)\b[\s\S]*$/i;
const PRIVATE_CLOSING_TAG =
  /(?:\\?<|&lt;)\s*\/\s*(?:think|tool_call|function)\s*(?:>|&gt;)/gi;

const SAFE_EMPTY_RESPONSE =
  "I couldn't produce a supported answer from the retrieved sources. Please try rephrasing the question.";

export function sanitizeAssistantContent(content: string): string {
  let previous = content;

  while (true) {
    const cleaned = previous
      .replace(COMPLETE_PRIVATE_BLOCK, "")
      .replace(ENCODED_PRIVATE_BLOCK, "");
    if (cleaned === previous) {
      return (
        cleaned
          .replace(UNCLOSED_PRIVATE_BLOCK, "")
          .replace(PRIVATE_CLOSING_TAG, "")
          .trim() || SAFE_EMPTY_RESPONSE
      );
    }
    previous = cleaned;
  }
}
