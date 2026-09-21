// Plain JavaScript on purpose. No framework and no CDN: the office machine
// has no internet, so a <script src="https://..."> tag is a page that
// silently does not work there, and vendoring a library would cost more code
// than the two interactions this app actually needs.
"use strict";

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

async function api(url, opts = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" }, ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

const trustBadge = (t) => {
  const cls = t >= 0.8 ? "b-ok" : t >= 0.6 ? "b-warn" : "b-bad";
  const b = el("span", "badge " + cls, (t ?? 0).toFixed(2));
  return b;
};

/* ---------------------------------------------------------- drafting page */
function initDraft() {
  const go = $("#go");
  if (!go) return;

  go.addEventListener("click", async () => {
    const request = $("#request").value.trim();
    if (request.length < 10) {
      $("#status").textContent = "कृपया एक वाक्य में बताइए कि किस बारे में पत्र चाहिए।";
      return;
    }
    go.disabled = true;
    // At the measured ~9 Hindi words/second a 400-word letter is about 45
    // seconds. Saying so beats a spinner that looks like a hang.
    $("#status").textContent = "तैयार किया जा रहा है… (लगभग ४५ सेकंड)";
    try {
      const d = await api("/api/draft", {
        method: "POST",
        body: JSON.stringify({
          request,
          department: $("#department").value,
          letter_type: $("#letter_type").value,
          subject: $("#subject").value,
          letter_number: $("#letter_number").value,
        }),
      });
      renderDraft(d);
      $("#status").textContent = d.queued
        ? "(एक और प्रारूप चल रहा था, इसलिए प्रतीक्षा करनी पड़ी)" : "";
    } catch (e) {
      $("#status").textContent = "त्रुटि: " + e.message;
    } finally {
      go.disabled = false;
    }
  });

  document.querySelectorAll("[data-export]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const fmt = btn.dataset.export;
      btn.disabled = true;
      try {
        const res = await fetch("/api/export/" + fmt, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: $("#letter").value }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
        const blob = await res.blob();
        const a = el("a");
        a.href = URL.createObjectURL(blob);
        a.download = (res.headers.get("content-disposition") || "")
          .split("filename=")[1]?.replace(/"/g, "") || "letter." + fmt;
        a.click();
        URL.revokeObjectURL(a.href);
      } catch (e) {
        $("#status").textContent = "निर्यात विफल: " + e.message;
      } finally {
        btn.disabled = false;
      }
    });
  });
}

function renderDraft(d) {
  $("#result").hidden = false;
  $("#evidence").hidden = false;
  $("#letter").value = d.text;
  $("#timing").textContent =
    `${d.timing.seconds}s · ${d.timing.prompt_tokens}+${d.timing.output_tokens} टोकन · ` +
    `${d.department || "?"} / ${d.letter_type || "?"}`;

  // The verification surface. Rendering the letter and hiding this would
  // undo the whole Phase 5 design, so it is built first and cannot be
  // collapsed away.
  const banner = $("#banner");
  banner.textContent = "";
  const box = el("div", "banner " + (d.needs_review ? "bad" : "ok"));
  box.appendChild(el("strong", null, d.needs_review
    ? "इस प्रारूप की जाँच आवश्यक है"
    : "कोई स्वतः पकड़ी गई समस्या नहीं — फिर भी पढ़कर ही भेजें"));
  if (d.invented_numbers.length) {
    const p = el("p");
    p.appendChild(el("strong", null, "मॉडल ने ये अंक स्वयं बनाए हैं: "));
    p.appendChild(document.createTextNode(d.invented_numbers.join(", ")));
    box.appendChild(p);
    box.appendChild(el("div", "en",
      "These appear nowhere in your request or in the source letters. Check every one."));
  }
  if (d.warnings.length) {
    const ul = el("ul");
    d.warnings.forEach((w) => ul.appendChild(el("li", null, w)));
    box.appendChild(ul);
  }
  banner.appendChild(box);

  const list = $("#sources");
  list.textContent = "";
  if (!d.sources.length) {
    list.appendChild(el("li", null,
      "कोई पुराना पत्र नहीं मिला — यह प्रारूप संग्रह पर आधारित नहीं है।"));
  }
  d.sources.forEach((s) => {
    const li = el("li");
    const a = el("a", null, "पत्र " + s.id);
    a.href = "#";
    a.addEventListener("click", async (ev) => {
      ev.preventDefault();
      const full = await api("/api/letter/" + s.id);
      let pre = li.querySelector("pre");
      if (pre) { pre.remove(); return; }
      pre = el("pre", "log", full.text);
      li.appendChild(pre);
    });
    li.appendChild(a);
    li.appendChild(document.createTextNode(" — " + s.why));
    list.appendChild(li);
  });

  const notes = $("#notes");
  notes.textContent = "";
  if (d.removed.length) {
    notes.appendChild(el("p", "meta",
      "मॉडल के उत्तर से हटाया गया: " + d.removed.join(", ")));
  }
  notes.appendChild(el("p", "meta",
    `भाषा जाँच: ${d.quality.score} (${d.quality.verdict})`));
}

/* ------------------------------------------------------------- admin page */
function initAdmin() {
  if (!$("#stats")) return;
  refreshStats();
  search();

  $("#upload").addEventListener("click", async () => {
    const input = $("#files");
    if (!input.files.length) return;
    const fd = new FormData();
    for (const f of input.files) fd.append("files", f);
    $("#upload").disabled = true;
    $("#uploadlog").hidden = false;
    $("#uploadlog").textContent = "पढ़ा जा रहा है…";
    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const r = await res.json();
      const lines = [
        `जोड़े गए: ${r.inserted}   पहले से मौजूद: ${r.duplicates}`,
        ...r.files.map((f) => `  ${f.file}: ${f.letters} पत्र`),
        ...r.skipped.map((s) => `  छोड़ा गया ${s.file}: ${s.reason}`),
      ];
      $("#uploadlog").textContent = lines.join("\n");
      refreshStats();
      search();
    } catch (e) {
      $("#uploadlog").textContent = "त्रुटि: " + e.message;
    } finally {
      $("#upload").disabled = false;
    }
  });

  $("#search").addEventListener("click", search);
  $("#q").addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });
  $("#save").addEventListener("click", saveLetter);
  $("#delete").addEventListener("click", deleteLetter);
  $("#sk").addEventListener("change", loadSkeleton);
  $("#sksave").addEventListener("click", saveSkeleton);
}

async function refreshStats() {
  const s = await api("/api/stats");
  const box = $("#stats");
  box.textContent = "";
  const add = (value, label) => {
    const d = el("div", "stat");
    d.appendChild(el("b", null, String(value)));
    d.appendChild(el("span", null, label));
    box.appendChild(d);
  };
  add(s.letters, "कुल पत्र");
  add((s.mean_trust ?? 0).toFixed(2), "औसत विश्वसनीयता");
  Object.entries(s.by_verdict || {}).forEach(([k, v]) => add(v, k));
  Object.entries(s.by_form || {}).forEach(([k, v]) => add(v, k));
}

async function search() {
  const params = new URLSearchParams({
    q: $("#q").value, verdict_filter: $("#verdict").value, form: $("#form").value,
  });
  const rows = await api("/api/letters?" + params);
  const body = $("#letters").querySelector("tbody");
  body.textContent = "";
  rows.forEach((r) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, String(r.id)));
    tr.appendChild(el("td", null, r.subject || r.preview));
    tr.appendChild(el("td", null, r.form || "?"));
    const td = el("td");
    td.appendChild(trustBadge(r.trust));
    tr.appendChild(td);
    let missing = [];
    try { missing = JSON.parse(r.missing || "[]"); } catch (_) {}
    tr.appendChild(el("td", "meta", missing.join(", ")));
    const act = el("td");
    const b = el("button", null, "सुधारें");
    b.addEventListener("click", () => openLetter(r.id));
    act.appendChild(b);
    tr.appendChild(act);
    body.appendChild(tr);
  });
}

let editing = null;
async function openLetter(id) {
  const r = await api("/api/letter/" + id);
  editing = id;
  $("#editor").hidden = false;
  $("#edit-id").textContent = "#" + id;
  $("#edit-meta").textContent =
    `${r.source_file} · ${r.form || "?"} · विश्वसनीयता ${(r.trust ?? 0).toFixed(2)} (${r.verdict})`;
  $("#edit-text").value = r.text;
  $("#editstatus").textContent = "";
  $("#editor").scrollIntoView({ behavior: "smooth" });
}

async function saveLetter() {
  if (editing === null) return;
  $("#editstatus").textContent = "सुरक्षित किया जा रहा है…";
  try {
    // Re-scored on save, not just stored: the reason to correct a letter is
    // usually that its conversion was wrong, which means its trust score was
    // computed from the wrong text.
    const r = await api("/api/letters/" + editing, {
      method: "POST", body: JSON.stringify({ text: $("#edit-text").value }),
    });
    $("#editstatus").textContent =
      `सुरक्षित। नई विश्वसनीयता ${r.trust.toFixed(2)} (${r.verdict}), प्रकार ${r.form || "?"}`;
    refreshStats();
    search();
  } catch (e) {
    $("#editstatus").textContent = "त्रुटि: " + e.message;
  }
}

async function deleteLetter() {
  if (editing === null) return;
  if (!confirm("पत्र " + editing + " को स्थायी रूप से हटाएँ?")) return;
  await api("/api/letters/" + editing, { method: "DELETE" });
  $("#editor").hidden = true;
  editing = null;
  refreshStats();
  search();
}

async function loadSkeleton() {
  const name = $("#sk").value;
  const show = Boolean(name);
  $("#sktext").hidden = !show;
  $("#sksave").hidden = !show;
  $("#skstatus").textContent = "";
  if (!show) return;
  $("#sktext").value = (await api("/api/skeleton/" + encodeURIComponent(name))).text;
}

async function saveSkeleton() {
  const name = $("#sk").value;
  if (!name) return;
  try {
    const r = await api("/api/skeleton/" + encodeURIComponent(name), {
      method: "POST", body: JSON.stringify({ text: $("#sktext").value }),
    });
    $("#skstatus").textContent = `सुरक्षित (${r.lines} पंक्तियाँ)। अगले प्रारूप से यही चलेगा।`;
  } catch (e) {
    $("#skstatus").textContent = "त्रुटि: " + e.message;
  }
}

initDraft();
initAdmin();
