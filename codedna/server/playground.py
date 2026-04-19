"""Browser interface for the CodeDNA local server."""

from __future__ import annotations


def render_playground() -> str:
    """Return a minimal chat interface for the local model."""

    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CodeDNA</title>
  <style>
    :root {
      --bg: #ffffff;
      --panel: #ffffff;
      --line: #e5e7eb;
      --ink: #111827;
      --muted: #6b7280;
      --user: #f3f4f6;
      --assistant: #ffffff;
      --accent: #111827;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
    }
    .page {
      min-height: 100%;
      display: flex;
      flex-direction: column;
    }
    .header {
      text-align: center;
      padding: 28px 20px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.94);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .header-inner {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 16px;
    }
    .title {
      margin: 0;
      font-size: clamp(1.9rem, 4vw, 2.8rem);
      letter-spacing: -0.04em;
      font-weight: 600;
    }
    .subtitle {
      margin: 8px auto 0;
      max-width: 560px;
      color: var(--muted);
      font-size: 0.98rem;
      line-height: 1.5;
    }
    .clear-btn {
      border: 1px solid var(--line);
      background: transparent;
      color: var(--muted);
      border-radius: 999px;
      padding: 6px 14px;
      font: inherit;
      font-size: 0.85rem;
      cursor: pointer;
      transition: color 0.15s, border-color 0.15s;
    }
    .clear-btn:hover {
      color: var(--ink);
      border-color: #9ca3af;
    }
    .messages {
      flex: 1;
      width: 100%;
      max-width: 860px;
      margin: 0 auto;
      padding: 28px 20px 160px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .message {
      width: 100%;
      display: flex;
    }
    .bubble {
      max-width: min(780px, 100%);
      padding: 16px 18px;
      border-radius: 22px;
      border: 1px solid var(--line);
      line-height: 1.6;
      font-size: 0.98rem;
    }
    .bubble p {
      margin: 0 0 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .bubble p:last-child {
      margin-bottom: 0;
    }
    .code-block {
      margin: 12px 0;
      border: 1px solid #d7dce2;
      border-radius: 16px;
      overflow: hidden;
      background: #f3f4f6;
      color: #1f2937;
    }
    .code-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
      background: #e5e7eb;
      border-bottom: 1px solid #d1d5db;
      font-size: 0.82rem;
      color: #4b5563;
    }
    .code-copy {
      border: 1px solid #d1d5db;
      background: #ffffff;
      color: #374151;
      border-radius: 999px;
      padding: 5px 10px;
      font: inherit;
      cursor: pointer;
    }
    .code-copy:disabled {
      opacity: 0.7;
      cursor: default;
    }
    .code-block pre {
      margin: 0;
      padding: 14px 16px 16px;
      overflow-x: auto;
      white-space: pre;
      font: 0.9rem/1.55 "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }
    .message.user {
      justify-content: flex-end;
    }
    .message.user .bubble {
      background: var(--user);
    }
    .message.assistant .bubble {
      background: var(--assistant);
    }
    .message.error .bubble {
      color: #991b1b;
      border-color: #fecaca;
      background: #fef2f2;
    }
    .loading .bubble {
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }
    .spinner {
      width: 16px;
      height: 16px;
      border-radius: 50%;
      border: 2px solid #d1d5db;
      border-top-color: var(--accent);
      animation: spin 0.8s linear infinite;
      flex: 0 0 auto;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .composer-wrap {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      padding: 16px 20px 22px;
      background: linear-gradient(180deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.92) 28%, #ffffff 100%);
    }
    .composer {
      max-width: 860px;
      margin: 0 auto;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 28px;
      display: flex;
      align-items: flex-end;
      gap: 12px;
      padding: 12px 12px 12px 18px;
      box-shadow: 0 10px 30px rgba(17, 24, 39, 0.06);
    }
    .input {
      flex: 1;
      min-height: 26px;
      max-height: 180px;
      border: none;
      resize: none;
      outline: none;
      font: inherit;
      color: var(--ink);
      background: transparent;
      line-height: 1.5;
      padding: 6px 0;
    }
    .input::placeholder {
      color: #9ca3af;
    }
    .send {
      border: none;
      background: var(--accent);
      color: white;
      width: 42px;
      height: 42px;
      border-radius: 50%;
      font-size: 1rem;
      cursor: pointer;
      flex: 0 0 auto;
    }
    .send:disabled {
      cursor: not-allowed;
      opacity: 0.45;
    }
    @media (max-width: 720px) {
      .header {
        padding: 22px 16px 14px;
      }
      .messages {
        padding: 22px 14px 148px;
      }
      .composer-wrap {
        padding: 12px 14px 18px;
      }
      .bubble {
        font-size: 0.95rem;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <header class="header">
      <div class="header-inner">
        <h1 class="title">CodeDNA</h1>
        <button class="clear-btn" id="clearBtn" type="button">Clear chat</button>
      </div>
      <p class="subtitle">A coding assistant tuned on your repository style.</p>
    </header>

    <main id="messages" class="messages">
      <div class="message assistant">
        <div class="bubble">How can I help with your code?</div>
      </div>
    </main>

    <div class="composer-wrap">
      <form id="composer" class="composer">
        <textarea
          id="userPrompt"
          class="input"
          rows="1"
          placeholder="Message CodeDNA"
        ></textarea>
        <button id="sendBtn" class="send" type="submit" aria-label="Send">↑</button>
      </form>
    </div>
  </div>

  <script>
    const messagesEl = document.getElementById("messages");
    const composer = document.getElementById("composer");
    const userPrompt = document.getElementById("userPrompt");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");

    // FIX: maintain full conversation history so the model has context across turns,
    // exactly like the OpenAI / GPT API works.  Every request sends the complete
    // history; the server is stateless but the browser holds the state.
    const SYSTEM_PROMPT = "You are a Python coding assistant. Answer with a short explanation followed by one clean, complete code block. Do not cut the code short — always finish it. No usage examples or extra notes after the code block.";

    let history = []; // array of {role, content} objects
    let loadingNode = null;

    function autoResize() {
      userPrompt.style.height = "auto";
      userPrompt.style.height = Math.min(userPrompt.scrollHeight, 180) + "px";
    }

    function scrollToBottom() {
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }

    function escapeHtml(value) {
      return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function renderBubbleContent(bubble, content) {
      const segments = content.split(/```([\w+-]*)\n([\s\S]*?)```/g);
      if (segments.length === 1) {
        appendTextBlock(bubble, content);
        return;
      }
      for (let index = 0; index < segments.length; index += 1) {
        if (index % 3 === 0) {
          appendTextBlock(bubble, segments[index]);
          continue;
        }
        const language = (segments[index] || "").trim() || "code";
        const code = segments[index + 1] || "";
        bubble.appendChild(createCodeBlock(language, code.replace(/\n$/, "")));
      }
    }

    function appendTextBlock(container, text) {
      const normalized = text.trim();
      if (!normalized) return;
      const paragraphs = normalized.split(/\n{2,}/);
      for (const paragraph of paragraphs) {
        const node = document.createElement("p");
        node.textContent = paragraph.trim();
        container.appendChild(node);
      }
    }

    function createCodeBlock(language, code) {
      const wrapper = document.createElement("div");
      wrapper.className = "code-block";

      const head = document.createElement("div");
      head.className = "code-head";

      const label = document.createElement("span");
      label.textContent = language;

      const button = document.createElement("button");
      button.className = "code-copy";
      button.type = "button";
      button.textContent = "Copy";
      button.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(code);
          button.textContent = "Copied";
          button.disabled = true;
          setTimeout(() => {
            button.textContent = "Copy";
            button.disabled = false;
          }, 1200);
        } catch (error) {
          button.textContent = "Failed";
          setTimeout(() => { button.textContent = "Copy"; }, 1200);
        }
      });

      head.appendChild(label);
      head.appendChild(button);

      const pre = document.createElement("pre");
      const codeNode = document.createElement("code");
      codeNode.innerHTML = escapeHtml(code);
      pre.appendChild(codeNode);

      wrapper.appendChild(head);
      wrapper.appendChild(pre);
      return wrapper;
    }

    function addMessage(role, content, extraClass = "") {
      const wrapper = document.createElement("div");
      wrapper.className = `message ${role} ${extraClass}`.trim();
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      renderBubbleContent(bubble, content);
      wrapper.appendChild(bubble);
      messagesEl.appendChild(wrapper);
      scrollToBottom();
      return wrapper;
    }

    function showLoading() {
      hideLoading();
      loadingNode = document.createElement("div");
      loadingNode.className = "message assistant loading";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      const spinner = document.createElement("span");
      spinner.className = "spinner";
      const text = document.createElement("span");
      text.textContent = "Thinking";
      bubble.appendChild(spinner);
      bubble.appendChild(text);
      loadingNode.appendChild(bubble);
      messagesEl.appendChild(loadingNode);
      scrollToBottom();
    }

    function hideLoading() {
      if (loadingNode) {
        loadingNode.remove();
        loadingNode = null;
      }
    }

    function resetChat() {
      history = [];
      messagesEl.innerHTML = "";
      const welcome = document.createElement("div");
      welcome.className = "message assistant";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = "How can I help with your code?";
      welcome.appendChild(bubble);
      messagesEl.appendChild(welcome);
    }

    async function sendPrompt() {
      const userContent = userPrompt.value.trim();
      if (!userContent) return;

      // Add user turn to UI and history
      addMessage("user", userContent);
      history.push({ role: "user", content: userContent });

      userPrompt.value = "";
      autoResize();
      sendBtn.disabled = true;
      showLoading();

      // Build full message array: system prompt + entire conversation history
      const messages = [
        { role: "system", content: SYSTEM_PROMPT },
        ...history,
      ];

      try {
        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "codedna-local",
            messages,
            // FIX: 300 was too low for any real code response — bumped to 700.
            // Adjust higher if your model/hardware can handle it.
            max_tokens: 700,
            temperature: 0.2,
          }),
        });

        // FIX: always read the body before checking ok so error details are
        // available in the message rather than a generic "Request failed".
        const payload = await response.json();
        if (!response.ok) {
          const detail = payload.detail || payload.message || JSON.stringify(payload);
          throw new Error(`${response.status}: ${detail}`);
        }

        const content = payload.choices?.[0]?.message?.content || "(empty response)";
        hideLoading();
        addMessage("assistant", content);

        // Store assistant reply in history so future turns have full context
        history.push({ role: "assistant", content });
      } catch (error) {
        hideLoading();
        addMessage("assistant", error.message, "error");
        // Remove the user message from history on failure so the broken
        // exchange doesn't corrupt future turns
        history.pop();
      } finally {
        sendBtn.disabled = false;
        userPrompt.focus();
      }
    }

    composer.addEventListener("submit", async (event) => {
      event.preventDefault();
      await sendPrompt();
    });

    clearBtn.addEventListener("click", () => {
      resetChat();
    });

    userPrompt.addEventListener("input", autoResize);
    userPrompt.addEventListener("keydown", async (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        await sendPrompt();
      }
    });

    autoResize();
  </script>
</body>
</html>
"""