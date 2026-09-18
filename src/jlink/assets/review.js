"use strict";
(() => {
  const doc = JSON.parse(document.getElementById("review-data").textContent);
  const source = doc.source;
  const storageKey = `jlink-review-v1:${doc.run_id}`;
  const $ = id => document.getElementById(id);
  const canonical = value => JSON.stringify(value, function (_key, val) {
    return val && typeof val === "object" && !Array.isArray(val)
      ? Object.fromEntries(Object.keys(val).sort().map(k => [k, val[k]])) : val;
  });
  const idKey = id => canonical(id);
  const pairKey = row => canonical([row.left_id, row.right_id]);
  const scores = source.scores.rows;
  const pairs = new Set(scores.map(pairKey));
  const original = new Set(source.links.rows.map(pairKey));
  const records = Object.fromEntries(["left", "right"].map(side => [side,
    new Map(source.records[side].map(r => [idKey(r.id), r]))]));
  const adjacency = {left: new Map(), right: new Map()};
  scores.forEach(row => ["left", "right"].forEach(side => {
    const key = idKey(row[`${side}_id`]);
    if (!adjacency[side].has(key)) adjacency[side].set(key, []);
    adjacency[side].get(key).push(row);
  }));
  let history = [], latest = new Map(), selected = null, queueLimit = 100, candidateLimit = 50;
  const el = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (cls) node.className = cls;
    return node;
  };
  const status = (text, error = false) => {
    $("status").textContent = text;
    $("status").className = error ? "error" : "";
  };
  const badge = (text, cls = "") => el("span", text, `badge ${cls}`);
  const fmt = p => p === null ? "Unjudged" : Number(p).toFixed(3);
  const validID = id => id && ["str", "int", "float"].includes(id.type) && typeof id.value === "string";

  function validateHistory(events) {
    if (!Array.isArray(events)) throw Error("History must be a list.");
    const byID = new Map(), children = new Map();
    const fields = ["event_id", "previous_event", "left_id", "right_id", "decision", "reviewer", "timestamp", "note"].sort();
    for (const e of events) {
      if (!e || canonical(Object.keys(e).sort()) !== canonical(fields)) throw Error("Malformed review event.");
      if (!validID(e.left_id) || !validID(e.right_id) || !pairs.has(pairKey(e))) throw Error("Unknown candidate in review history.");
      if (!["accept", "reject", "unsure"].includes(e.decision)) throw Error("Unknown decision.");
      if (typeof e.reviewer !== "string" || !e.reviewer.trim()) throw Error("Each event needs a reviewer identity.");
      if (typeof e.note !== "string" || typeof e.event_id !== "string" || !e.event_id ||
          (e.previous_event !== null && typeof e.previous_event !== "string")) throw Error("Malformed event identity or note.");
      if (typeof e.timestamp !== "string" || !/(Z|[+-]\d\d:\d\d)$/.test(e.timestamp) || !Number.isFinite(Date.parse(e.timestamp)))
        throw Error("Each timestamp needs a timezone.");
      if (byID.has(e.event_id)) throw Error("Duplicate event ID.");
      byID.set(e.event_id, e);
      const branch = canonical([pairKey(e), e.previous_event]);
      if (children.has(branch)) throw Error("Conflicting amendments to the same pair. Import the shared history, then amend explicitly.");
      children.set(branch, e.event_id);
    }
    const ordered = [], current = new Map();
    const roots = events.filter(e => e.previous_event === null).sort((a, b) =>
      a.timestamp.localeCompare(b.timestamp) || a.event_id.localeCompare(b.event_id));
    for (let event of roots) {
      const key = pairKey(event);
      while (event) {
        ordered.push(event); current.set(key, event);
        event = byID.get(children.get(canonical([key, event.event_id])));
      }
    }
    if (ordered.length !== events.length) throw Error("History has a missing predecessor, wrong pair predecessor or cycle.");
    return {ordered, current};
  }

  function mergeEvents(events) {
    validateHistory(events);
    const union = new Map(history.map(e => [e.event_id, e]));
    for (const e of events) {
      if (union.has(e.event_id) && canonical(union.get(e.event_id)) !== canonical(e))
        throw Error("Import changes an existing event.");
      union.set(e.event_id, e);
    }
    const checked = validateHistory([...union.values()]);
    history = checked.ordered; latest = checked.current;
  }

  function persist() {
    try {
      const existing = localStorage.getItem(storageKey);
      if (existing) {
        const stored = JSON.parse(existing);
        if (stored.run_id !== doc.run_id) throw Error("Stored history has a different source run.");
        mergeEvents(stored.history);
      }
      localStorage.setItem(storageKey, JSON.stringify({run_id: doc.run_id, history}));
      return true;
    } catch (error) {
      status(`Decision recorded in this tab. Browser storage could not be updated: ${error.message}. Export JSON now.`, true);
      return false;
    }
  }

  function conflicts() {
    const mode = source.settings.how;
    const sides = mode === "one-to-one" ? ["left", "right"] : mode === "many-to-one" ? ["left"] : mode === "one-to-many" ? ["right"] : [];
    const found = [];
    for (const side of sides) {
      const used = new Set();
      for (const event of latest.values()) if (event.decision === "accept") {
        const id = event[`${side}_id`], key = idKey(id);
        if (used.has(key)) found.push(`${side} ID ${id.value}`);
        used.add(key);
      }
    }
    $("conflicts").hidden = !found.length;
    $("conflicts").textContent = `Conflicting accepts under ${mode}: ${found.join(", ")}. Application will stop. Amend a decision to reject/unsure, or explicitly change cardinality when applying.`;
  }

  function facts(record, side) {
    const rows = adjacency[side].get(idKey(record.id)) || [];
    const ranked = rows.filter(r => r.p !== null).sort((a, b) => b.p - a.p);
    const close = ranked.length >= 2 && ranked[0].p - ranked[1].p <= source.close_margin;
    const unmatched = !rows.some(r => original.has(pairKey(r)));
    const unjudged = rows.some(r => r.p === null);
    const reviewed = rows.some(r => latest.has(pairKey(r)));
    const reasons = [];
    if (close) reasons.push(`Close competitors (gap ${(ranked[0].p - ranked[1].p).toFixed(3)})`);
    if (unmatched) reasons.push("Originally unmatched");
    if (!rows.length) reasons.push("No candidate");
    if (unjudged) reasons.push("Unjudged / error candidates");
    if (rows.length && rows.every(r => latest.get(pairKey(r))?.decision === "reject")) reasons.push("All candidates manually rejected");
    if (!reasons.length) reasons.push("Original link available");
    return {rows, close, unmatched, unjudged, reviewed, reasons, priority: close ? 0 : unmatched ? 1 : unjudged ? 2 : 3};
  }

  function renderQueue() {
    const side = $("side").value, filter = $("filter").value, query = $("search").value.toLocaleLowerCase();
    const all = source.records[side].map(record => ({record, info: facts(record, side)}));
    const shown = all.filter(({record, info}) =>
      (filter === "all" || (filter === "no-candidate" ? !info.rows.length : info[filter])) &&
      (!query || [record.id.value, ...Object.values(record.fields)].join(" ").toLocaleLowerCase().includes(query)))
      .sort((a, b) => a.info.priority - b.info.priority);
    if (!shown.some(x => idKey(x.record.id) === selected)) selected = shown.length ? idKey(shown[0].record.id) : null;
    $("queue-count").textContent = `${shown.length} of ${all.length} records · ${Math.min(shown.length, queueLimit)} shown`;
    $("queue").replaceChildren();
    for (const {record, info} of shown.slice(0, queueLimit)) {
      const button = el("button", undefined, "record-button" + (selected === idKey(record.id) ? " active" : ""));
      button.setAttribute("aria-pressed", String(selected === idKey(record.id)));
      button.append(el("strong", record.id.value), el("small", Object.values(record.fields).filter(v => v !== null).slice(0, 3).join(" · ")),
        el("small", info.reasons.join(" · ")));
      button.addEventListener("click", () => { selected = idKey(record.id); candidateLimit = 50; render(); });
      $("queue").append(button);
    }
    if (shown.length > queueLimit) {
      const more = el("button", "Show 100 more records", "secondary load-more");
      more.onclick = () => { queueLimit += 100; renderQueue(); };
      $("queue").append(more);
    }
  }

  function comparison(a, b) {
    const grid = el("div", undefined, "comparison");
    for (const text of ["FIELD", "LEFT RECORD", "RIGHT RECORD"]) grid.append(el("div", text, "table-head"));
    const matched = source.settings.on || [];
    const columns = [], usedA = new Set(), usedB = new Set();
    for (const [lc, rc] of matched) { columns.push([lc, rc]); usedA.add(lc); usedB.add(rc); }
    for (const col of Object.keys(a?.fields || {})) if (!usedA.has(col)) {
      columns.push([col, Object.hasOwn(b?.fields || {}, col) && !usedB.has(col) ? col : null]);
      usedB.add(col);
    }
    for (const col of Object.keys(b?.fields || {})) if (!usedB.has(col)) columns.push([null, col]);
    for (const [lc, rc] of columns) {
      grid.append(el("div", lc && rc && lc !== rc ? `${lc} / ${rc}` : lc || rc, "field"));
      for (const [record, field] of [[a, lc], [b, rc]]) {
        const value = field && record ? record.fields[field] : null;
        grid.append(el("div", value === null || value === undefined ? "—" : value));
      }
    }
    return grid;
  }

  function renderDetail() {
    const detail = $("detail"); detail.replaceChildren();
    const side = $("side").value, record = records[side].get(selected);
    if (!record) { detail.append(el("p", "No records match this filter.", "empty")); return; }
    const info = facts(record, side), head = el("div", undefined, "detail-head"), title = el("div");
    title.append(el("p", `${side.toUpperCase()} RECORD`, "eyebrow"), el("h2", record.id.value));
    info.reasons.forEach(reason => title.append(badge(reason, "warn")));
    head.append(title); detail.append(head);
    if (!info.rows.length) {
      const empty = el("div", undefined, "empty");
      empty.append(el("h3", "No candidate was proposed"), el("p", "There is no pair to accept or reject. Review blocking or add candidates in a separate run. No API calls are made here."), comparison(side === "left" ? record : null, side === "right" ? record : null));
      detail.append(empty); return;
    }
    detail.append(el("p", `${info.rows.length} candidates · Original selection is fixed; manual decisions are pending offline application.`, "muted"));
    const rows = [...info.rows].sort((a, b) => (b.p ?? -1) - (a.p ?? -1));
    for (const row of rows.slice(0, candidateLimit)) {
      const key = pairKey(row), event = latest.get(key), card = el("article", undefined, "candidate");
      card.dataset.pair = key;
      const heading = el("div", undefined, "candidate-head"), labels = el("div"), score = el("div", undefined, "score");
      labels.append(el("h3", `${row.left_id.value} ↔ ${row.right_id.value}`));
      labels.append(badge(original.has(key) ? "Original link" : "Not originally linked", original.has(key) ? "selected" : ""));
      if (event) labels.append(badge(event.decision === "accept" ? "Accepted constraint" : event.decision === "reject" ? "Manually rejected" : "Unsure · unconstrained", event.decision === "reject" ? "reject" : "warn"));
      if (row.p === null) labels.append(badge(row.source === "error" ? "Judgment error" : "Unjudged candidate", "warn"));
      else if (row.p < source.settings.threshold) labels.append(badge("Below original threshold", "warn"));
      else if (!original.has(key)) labels.append(badge("Not selected by original assignment / margin", "warn"));
      score.append(el("strong", fmt(row.p)), el("small", "original probability")); heading.append(labels, score); card.append(heading);
      card.append(el("p", `Source: ${row.source || "unknown"} · Similarity: ${fmt(row.sim)} · Block: ${row.block || "—"}`, "muted"));
      if (row.error) card.append(el("p", `Judgment error: ${row.error}`, "muted"));
      const otherSide = side === "left" ? "right" : "left";
      const otherRows = adjacency[otherSide].get(idKey(row[`${otherSide}_id`])) || [];
      const alternatives = otherRows.filter(r => pairKey(r) !== key);
      if (alternatives.length) {
        const competing = el("details"); competing.append(el("summary", `${alternatives.length} competing candidates for ${otherSide} ID ${row[`${otherSide}_id`].value}`));
        for (const alt of alternatives.slice(0, 50)) competing.append(el("p", `${alt.left_id.value} ↔ ${alt.right_id.value} · p ${fmt(alt.p)}${original.has(pairKey(alt)) ? " · original link" : ""}`));
        if (alternatives.length > 50) competing.append(el("p", "Switch record table to inspect all competitors."));
        card.append(competing);
      }
      card.append(comparison(records.left.get(idKey(row.left_id)), records.right.get(idKey(row.right_id))));
      const noteLabel = el("label", "Reason / amendment note"), note = el("textarea");
      note.placeholder = "Evidence for this decision (optional)"; noteLabel.append(note); card.append(noteLabel);
      const actions = el("div", undefined, "actions");
      for (const decision of ["accept", "reject", "unsure"]) {
        const button = el("button", decision[0].toUpperCase() + decision.slice(1), decision + (event?.decision === decision ? " chosen" : ""));
        button.setAttribute("aria-label", `${decision} ${row.left_id.value} to ${row.right_id.value}`);
        button.addEventListener("click", () => {
          const reviewer = $("reviewer").value.trim();
          if (!reviewer) { status("Enter a reviewer identity before recording a decision.", true); $("reviewer").focus(); return; }
          try {
            const previous = latest.get(key);
            const next = {event_id: crypto.randomUUID(), previous_event: previous?.event_id ?? null,
              left_id: row.left_id, right_id: row.right_id, decision, reviewer, timestamp: new Date().toISOString(), note: note.value};
            mergeEvents([...history, next]);
            if (persist()) status(`Recorded ${decision} for ${row.left_id.value} ↔ ${row.right_id.value}. Export JSON to keep this decision.`);
            render();
          } catch (error) { status(error.message, true); }
        });
        actions.append(button);
      }
      card.append(actions);
      const events = history.filter(e => pairKey(e) === key);
      if (events.length) {
        const log = el("details"); log.append(el("summary", `Decision history (${events.length})`));
        for (const e of events) log.append(el("p", `${e.decision} · ${e.reviewer} · ${e.timestamp}\n${e.note}\nEvent ${e.event_id}${e.previous_event ? ` amends ${e.previous_event}` : ""}`, "history-item"));
        card.append(log);
      }
      detail.append(card);
    }
    if (rows.length > candidateLimit) {
      const more = el("button", "Show 50 more candidates", "secondary");
      more.onclick = () => { candidateLimit += 50; renderDetail(); }; detail.append(more);
    }
  }

  function render() {
    $("n-reviewed").textContent = latest.size; $("n-events").textContent = history.length;
    conflicts(); renderQueue(); renderDetail();
  }
  $("n-records").textContent = source.records.left.length + source.records.right.length;
  $("n-links").textContent = original.size;
  $("rule").textContent = source.settings.question || source.settings.definition || "Review whether each candidate pair refers to the same entity.";
  $("settings").textContent = `Original settings: ${source.settings.how} · threshold ${source.settings.threshold} · minimum margin ${source.settings.min_margin ?? "off"} · close competitor gap ≤ ${source.close_margin}`;
  $("run-id").textContent = `Source run snapshot: ${doc.run_id}`;
  ["side", "filter", "search"].forEach(id => $(id).addEventListener("input", () => { queueLimit = 100; candidateLimit = 50; render(); }));
  $("export").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify({...doc, history}, null, 2) + "\n"], {type: "application/json"});
    const url = URL.createObjectURL(blob), a = el("a"); a.href = url; a.download = "review.json";
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    status(`Exported ${history.length} history events. Apply review.json offline to recompute links.`);
  });
  $("import").addEventListener("change", async event => {
    try {
      const file = event.target.files[0]; if (!file) return;
      const incoming = JSON.parse(await file.text());
      if (incoming.format !== doc.format || incoming.version !== doc.version || incoming.run_id !== doc.run_id || canonical(incoming.source) !== canonical(source))
        throw Error("Cannot import a different or altered source run.");
      mergeEvents(incoming.history);
      if (persist()) status(`Imported and merged ${incoming.history.length} events. Local history now has ${history.length} events.`);
      render();
    } catch (error) { status(`Import refused: ${error.message}`, true); }
    event.target.value = "";
  });
  try {
    mergeEvents(doc.history);
    const stored = localStorage.getItem(storageKey);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (parsed.run_id !== doc.run_id) throw Error("Stored history has a different source run.");
      mergeEvents(parsed.history); status(`Restored ${history.length} events for this source run. Export JSON for a durable copy.`);
    }
  } catch (error) { status(`Could not restore browser history: ${error.message}. Existing artifact history is retained.`, true); }
  render();
})();
