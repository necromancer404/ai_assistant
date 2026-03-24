const $ = (id) => document.getElementById(id);

const state = {
  messages: [], // { role: "user"|"assistant", content: string, sources?: Array }
};

function render() {
  const container = $("messages");
  container.innerHTML = "";
  for (const m of state.messages) {
    const wrap = document.createElement("div");
    const bubble = document.createElement("div");
    bubble.className = "bubble " + (m.role === "user" ? "user" : "bot");
    bubble.textContent = m.content;
    wrap.appendChild(bubble);

    if (m.role === "assistant" && Array.isArray(m.sources) && m.sources.length > 0) {
      const meta = document.createElement("div");
      meta.className = "meta";

      const toggle = document.createElement("button");
      toggle.className = "btn secondary";
      toggle.style.height = "32px";
      toggle.style.padding = "0 10px";
      toggle.textContent = "Show sources";

      const src = document.createElement("div");
      src.className = "sources";
      src.textContent = m.sources.map(s => `- ${s.source} (chunk ${s.chunk_index})`).join("\n");

      toggle.addEventListener("click", () => {
        const show = !src.classList.contains("show");
        src.classList.toggle("show", show);
        toggle.textContent = show ? "Hide sources" : "Show sources";
      });

      meta.appendChild(toggle);
      meta.appendChild(src);
      wrap.appendChild(meta);
    }

    container.appendChild(wrap);
  }
  container.scrollTop = container.scrollHeight;
}

function setLoading(loading) {
  $("ask").disabled = loading;
  $("q").disabled = loading;
}

async function ask() {
  const qEl = $("q");
  const api = $("api").value.trim();
  const topk = parseInt(($("topk").value || "3"), 10);
  const text = qEl.value.trim();
  if (!text) return;

  // Push user message
  state.messages.push({ role: "user", content: text });
  render();
  qEl.value = "";
  autoResize(qEl);

  // Placeholder assistant "typing" bubble
  const loaderIndex = state.messages.push({ role: "assistant", content: "Thinking…" }) - 1;
  render();

  setLoading(true);
  try {
    const res = await fetch(api, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, top_k: topk }),
    });
    if (!res.ok) {
      state.messages[loaderIndex] = { role: "assistant", content: `Error: ${res.status} ${res.statusText}` };
      render();
      return;
    }
    const data = await res.json();
    const answer = data.answer || "(no answer)";
    const sources = Array.isArray(data.sources) ? data.sources : [];
    state.messages[loaderIndex] = { role: "assistant", content: answer, sources };
    render();
  } catch (err) {
    state.messages[loaderIndex] = { role: "assistant", content: "Request failed. Is the API running at the configured URL?" };
    render();
  } finally {
    setLoading(false);
  }
}

function clearChat() {
  state.messages = [];
  render();
  $("q").focus();
}

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 160) + "px";
}

$("ask").addEventListener("click", ask);
$("clear").addEventListener("click", clearChat);

$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});

$("q").addEventListener("input", (e) => {
  autoResize(e.target);
});

// Initial greeting
state.messages.push({
  role: "assistant",
  content: "Hi! I’m the VIT Chennai AI Assistant. Ask about campus, academics, placements, and more from your local knowledge base.",
});
render();