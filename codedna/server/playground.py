"""Browser interface for the CodeDNA local server."""

from __future__ import annotations


def render_playground() -> str:
    """Return a minimal chat interface for the local model."""

    return """<!DOCTYPE html>
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
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.6;
      font-size: 0.98rem;
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
      <h1 class="title">CodeDNA</h1>
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
    const messages = document.getElementById("messages");
    const composer = document.getElementById("composer");
    const userPrompt = document.getElementById("userPrompt");
    const sendBtn = document.getElementById("sendBtn");
    const systemPrompt = "You are a coding assistant that writes Python in the developer's style.";
    let loadingNode = null;

    function autoResize() {
      userPrompt.style.height = "auto";
      userPrompt.style.height = Math.min(userPrompt.scrollHeight, 180) + "px";
    }

    function scrollToBottom() {
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }

    function addMessage(role, content, extraClass = "") {
      const wrapper = document.createElement("div");
      wrapper.className = `message ${role} ${extraClass}`.trim();
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = content;
      wrapper.appendChild(bubble);
      messages.appendChild(wrapper);
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
      messages.appendChild(loadingNode);
      scrollToBottom();
    }

    function hideLoading() {
      if (loadingNode) {
        loadingNode.remove();
        loadingNode = null;
      }
    }

    async function sendPrompt() {
      const userContent = userPrompt.value.trim();
      if (!userContent) return;

      addMessage("user", userContent);
      userPrompt.value = "";
      autoResize();
      sendBtn.disabled = true;
      showLoading();

      try {
        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "codedna-local",
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: userContent }
            ],
            max_tokens: 240,
            temperature: 0.2
          })
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Request failed");
        }

        const content = payload.choices?.[0]?.message?.content || "(empty response)";
        hideLoading();
        addMessage("assistant", content);
      } catch (error) {
        hideLoading();
        addMessage("assistant", error.message, "error");
      } finally {
        sendBtn.disabled = false;
        userPrompt.focus();
      }
    }

    composer.addEventListener("submit", async (event) => {
      event.preventDefault();
      await sendPrompt();
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
