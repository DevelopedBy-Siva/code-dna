"""Simple browser playground for the CodeDNA local server."""

from __future__ import annotations


def render_playground() -> str:
    """Return a lightweight HTML playground for chatting with the local model."""

    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CodeDNA Playground</title>
  <style>
    :root {
      --bg: #f3efe7;
      --panel: #fffaf3;
      --line: #d8ccba;
      --ink: #1f1c18;
      --muted: #766c60;
      --accent: #195c4f;
      --accent-2: #d29c2e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(210, 156, 46, 0.18), transparent 26%),
        linear-gradient(180deg, #f8f2e8 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }
    .shell {
      max-width: 1100px;
      margin: 0 auto;
      min-height: 100vh;
      padding: 32px 20px 40px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 20px;
      margin-bottom: 22px;
    }
    .title {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }
    .subtitle {
      margin: 10px 0 0;
      max-width: 580px;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.5;
    }
    .status {
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
      min-width: 220px;
    }
    .status strong { display: block; margin-bottom: 6px; }
    .grid {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 20px;
    }
    .panel {
      background: color-mix(in srgb, var(--panel) 88%, white);
      border: 1px solid var(--line);
      box-shadow: 0 14px 30px rgba(46, 35, 21, 0.08);
    }
    .controls { padding: 18px; }
    .controls h2, .chat h2 {
      margin: 0 0 14px;
      font-size: 0.95rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    label {
      display: block;
      margin: 12px 0 6px;
      font-size: 0.9rem;
      color: var(--muted);
    }
    textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.8);
      color: var(--ink);
      padding: 12px;
      font: inherit;
      resize: vertical;
    }
    textarea { min-height: 150px; }
    .button-row {
      display: flex;
      gap: 10px;
      margin-top: 16px;
    }
    button {
      border: none;
      padding: 12px 16px;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button.primary {
      background: var(--accent);
      color: white;
      flex: 1;
    }
    button.secondary {
      background: var(--accent-2);
      color: #2b1f0a;
    }
    .chat {
      display: flex;
      flex-direction: column;
      min-height: 70vh;
    }
    .chat-head {
      padding: 18px 18px 0;
    }
    .messages {
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      overflow: auto;
      flex: 1;
    }
    .message {
      padding: 14px 16px;
      border-left: 4px solid var(--line);
      background: rgba(255,255,255,0.74);
      white-space: pre-wrap;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }
    .message.user { border-left-color: var(--accent-2); }
    .message.assistant { border-left-color: var(--accent); }
    .message.error { border-left-color: #a33b2f; color: #7f281f; }
    .msg-label {
      display: block;
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .composer {
      border-top: 1px solid var(--line);
      padding: 18px;
      background: rgba(255,255,255,0.55);
    }
    .hint {
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.88rem;
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
      .header { flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div>
        <h1 class="title">CodeDNA<br/>Playground</h1>
        <p class="subtitle">A minimal browser UI for your locally served fine-tuned coding assistant. This page talks directly to the same <code>/v1/chat/completions</code> endpoint that editors and tools use.</p>
      </div>
      <div class="status panel">
        <strong>Server Status</strong>
        <div id="healthText">Checking health...</div>
      </div>
    </div>

    <div class="grid">
      <aside class="panel controls">
        <h2>Session Controls</h2>
        <label for="systemPrompt">System Prompt</label>
        <textarea id="systemPrompt">You are a coding assistant that writes Python in the developer's style.</textarea>

        <label for="temperature">Temperature</label>
        <input id="temperature" type="number" min="0" max="2" step="0.1" value="0.2" />

        <label for="maxTokens">Max Tokens</label>
        <input id="maxTokens" type="number" min="1" max="4096" step="1" value="200" />

        <div class="button-row">
          <button class="primary" id="sendBtn">Send</button>
          <button class="secondary" id="clearBtn">Clear</button>
        </div>
        <p class="hint">Tip: keep prompts concrete and implementation-oriented for the strongest responses.</p>
      </aside>

      <section class="panel chat">
        <div class="chat-head">
          <h2>Conversation</h2>
        </div>
        <div id="messages" class="messages">
          <div class="message assistant">
            <span class="msg-label">Assistant</span>
            Server is ready. Ask for a function, refactor, or debugging helper.
          </div>
        </div>
        <div class="composer">
          <label for="userPrompt">Prompt</label>
          <textarea id="userPrompt">Write a Python retry helper with exponential backoff.</textarea>
        </div>
      </section>
    </div>
  </div>

  <script>
    const healthText = document.getElementById("healthText");
    const messages = document.getElementById("messages");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");
    const systemPrompt = document.getElementById("systemPrompt");
    const userPrompt = document.getElementById("userPrompt");
    const temperature = document.getElementById("temperature");
    const maxTokens = document.getElementById("maxTokens");

    async function refreshHealth() {
      try {
        const response = await fetch("/health");
        const payload = await response.json();
        healthText.textContent = payload.model_loaded
          ? "Loaded and ready on /v1/chat/completions"
          : `Not loaded: ${payload.load_error || "unknown error"}`;
      } catch (error) {
        healthText.textContent = `Health check failed: ${error.message}`;
      }
    }

    function addMessage(role, content, extraClass = "") {
      const wrapper = document.createElement("div");
      wrapper.className = `message ${role} ${extraClass}`.trim();
      const label = document.createElement("span");
      label.className = "msg-label";
      label.textContent = role === "user" ? "User" : role === "assistant" ? "Assistant" : "System";
      wrapper.appendChild(label);
      wrapper.appendChild(document.createTextNode(content));
      messages.appendChild(wrapper);
      messages.scrollTop = messages.scrollHeight;
    }

    async function sendPrompt() {
      const userContent = userPrompt.value.trim();
      if (!userContent) return;

      sendBtn.disabled = true;
      addMessage("user", userContent);

      try {
        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "codedna-local",
            messages: [
              { role: "system", content: systemPrompt.value.trim() },
              { role: "user", content: userContent }
            ],
            max_tokens: Number(maxTokens.value || 200),
            temperature: Number(temperature.value || 0.2)
          })
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Request failed");
        }

        const content = payload.choices?.[0]?.message?.content || "(empty response)";
        addMessage("assistant", content);
      } catch (error) {
        addMessage("assistant", error.message, "error");
      } finally {
        sendBtn.disabled = false;
      }
    }

    sendBtn.addEventListener("click", sendPrompt);
    clearBtn.addEventListener("click", () => {
      messages.innerHTML = "";
      addMessage("assistant", "Conversation cleared. Ready for the next prompt.");
    });
    userPrompt.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        sendPrompt();
      }
    });

    refreshHealth();
  </script>
</body>
</html>
"""
