const sessionListEl = document.getElementById("session-list");
const chatWindowEl = document.getElementById("chat-window");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat");

const state = {
  sessions: [],
  activeSessionId: null,
  messages: [],
};

function createSessionId() {
  if (window.crypto && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function persistActiveSession(id) {
  localStorage.setItem("active_session_id", id);
}

function loadActiveSession() {
  return localStorage.getItem("active_session_id");
}

async function fetchSessions() {
  const response = await fetch("/api/sessions");
  if (!response.ok) {
    return [];
  }
  return response.json();
}

async function fetchSessionMessages(sessionId) {
  const response = await fetch(`/api/sessions/${sessionId}`);
  if (!response.ok) {
    return [];
  }
  const payload = await response.json();
  return payload.messages || [];
}

function renderSessions() {
  sessionListEl.innerHTML = "";
  if (!state.sessions.length) {
    const empty = document.createElement("div");
    empty.className = "session-item";
    empty.textContent = "No sessions yet";
    sessionListEl.appendChild(empty);
    if (state.activeSessionId) {
      const active = document.createElement("div");
      active.className = "session-item active";
      active.textContent = "New Chat";
      sessionListEl.appendChild(active);
    }
    return;
  }

  const seen = new Set();
  state.sessions.forEach((session) => {
    seen.add(session.session_id);
    const item = document.createElement("div");
    item.className = "session-item";
    if (session.session_id === state.activeSessionId) {
      item.classList.add("active");
    }
    item.textContent = session.title || session.session_id;
    item.addEventListener("click", () => setActiveSession(session.session_id));
    sessionListEl.appendChild(item);
  });

  if (state.activeSessionId && !seen.has(state.activeSessionId)) {
    const active = document.createElement("div");
    active.className = "session-item active";
    active.textContent = "New Chat";
    sessionListEl.appendChild(active);
  }
}

function renderMessages() {
  chatWindowEl.innerHTML = "";
  state.messages.forEach((message) => renderMessage(message));
  chatWindowEl.scrollTop = chatWindowEl.scrollHeight;
}

function renderMessage(message) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${message.role}`;

  if (message.role === "assistant" && message.thinking_logs?.length) {
    const logPanel = document.createElement("div");
    logPanel.className = "log-panel";
    const title = document.createElement("h4");
    title.textContent = "Thinking Logs";
    logPanel.appendChild(title);
    const list = document.createElement("ul");
    message.thinking_logs.forEach((entry) => {
      const item = document.createElement("li");
      if (typeof entry === "string") {
        item.textContent = entry;
      } else {
        item.textContent = `${entry.event}: ${entry.detail}`;
      }
      list.appendChild(item);
    });
    logPanel.appendChild(list);
    wrapper.appendChild(logPanel);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (message.role === "assistant") {
    bubble.classList.add("answer");
    renderMarkdown(bubble, message.content || "");
  } else {
    bubble.textContent = message.content;
  }

  wrapper.appendChild(bubble);
  chatWindowEl.appendChild(wrapper);
  chatWindowEl.scrollTop = chatWindowEl.scrollHeight;
}

function renderMarkdown(container, text) {
  const content = String(text || "").replace(/\r/g, "");
  if (!content.trim()) {
    return;
  }

  const lines = content.split("\n");
  let listEl = null;

  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) {
      listEl = null;
      return;
    }

    const listMatch = line.match(/^[-*]\s+(.*)/);
    if (listMatch) {
      if (!listEl) {
        listEl = document.createElement("ul");
        container.appendChild(listEl);
      }
      const li = document.createElement("li");
      appendInline(li, listMatch[1]);
      listEl.appendChild(li);
      return;
    }

    listEl = null;
    const p = document.createElement("p");
    appendInline(p, line);
    container.appendChild(p);
  });
}

function appendInline(container, text) {
  let index = 0;
  while (index < text.length) {
    const imageToken = parseImageToken(text, index);
    if (imageToken) {
      const img = document.createElement("img");
      img.src = imageToken.url;
      img.alt = imageToken.alt || "Product image";
      container.appendChild(img);
      index += imageToken.length;
      continue;
    }

    if (text.startsWith("**", index)) {
      const end = text.indexOf("**", index + 2);
      if (end !== -1) {
        const strong = document.createElement("strong");
        strong.textContent = text.slice(index + 2, end);
        container.appendChild(strong);
        index = end + 2;
        continue;
      }
    }

    if (text.startsWith("*", index)) {
      const end = text.indexOf("*", index + 1);
      if (end !== -1) {
        const em = document.createElement("em");
        em.textContent = text.slice(index + 1, end);
        container.appendChild(em);
        index = end + 1;
        continue;
      }
    }

    const nextSpecial = findNextSpecial(text, index + 1);
    const chunk = text.slice(index, nextSpecial);
    appendText(container, chunk);
    index = nextSpecial;
  }
}

function parseImageToken(text, startIndex) {
  if (!text.startsWith("![", startIndex)) {
    return null;
  }
  const altEnd = text.indexOf("]", startIndex + 2);
  if (altEnd === -1 || text[altEnd + 1] !== "(") {
    return null;
  }
  const urlEnd = text.indexOf(")", altEnd + 2);
  if (urlEnd === -1) {
    return null;
  }
  const alt = text.slice(startIndex + 2, altEnd).trim();
  const url = text.slice(altEnd + 2, urlEnd).trim();
  if (!url) {
    return null;
  }
  return {
    alt,
    url,
    length: urlEnd - startIndex + 1,
  };
}

function findNextSpecial(text, startIndex) {
  const candidates = [
    text.indexOf("![", startIndex),
    text.indexOf("**", startIndex),
    text.indexOf("*", startIndex),
  ].filter((value) => value !== -1);

  if (!candidates.length) {
    return text.length;
  }
  return Math.min(...candidates);
}

function appendText(container, text) {
  if (!text) {
    return;
  }
  container.appendChild(document.createTextNode(text));
}

async function setActiveSession(sessionId) {
  state.activeSessionId = sessionId;
  persistActiveSession(sessionId);
  state.messages = await fetchSessionMessages(sessionId);
  renderSessions();
  renderMessages();
}

async function refreshSessions() {
  state.sessions = await fetchSessions();
  renderSessions();
}

function addLocalMessage(message) {
  state.messages.push(message);
  renderMessage(message);
}

async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text) {
    return;
  }
  messageInput.value = "";

  addLocalMessage({ role: "user", content: text });

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.activeSessionId,
      message: text,
    }),
  });

  if (!response.ok) {
    addLocalMessage({
      role: "assistant",
      content: "Da co loi khi goi API. Anh/Chi thu lai giup Em nhe.",
    });
    return;
  }

  const payload = await response.json();
  state.activeSessionId = payload.session_id;
  persistActiveSession(payload.session_id);

  addLocalMessage({
    role: "assistant",
    content: payload.answer_text,
    thinking_logs: payload.thinking_logs,
  });

  await refreshSessions();
}

function startNewChat() {
  const newId = createSessionId();
  state.activeSessionId = newId;
  persistActiveSession(newId);
  state.messages = [];
  renderMessages();
  refreshSessions();
}

sendBtn.addEventListener("click", sendMessage);
newChatBtn.addEventListener("click", startNewChat);
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

async function bootstrap() {
  await refreshSessions();
  const saved = loadActiveSession();
  if (saved) {
    await setActiveSession(saved);
  } else {
    startNewChat();
  }
}

bootstrap();
