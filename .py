"""SmartBasket - Budget Shopping Optimizer (0/1 Knapsack with Dynamic Programming).

ONE-FILE VERSION (backend + frontend together).
Run:   python budget_shopping_optimizer.py
It opens http://localhost:8000 in your browser. Press Ctrl+C in the terminal to stop.

Parts of this file:
  1. Optimization engine  - 0/1 Knapsack with Dynamic Programming (+ Must Have locking)
  2. Python backend       - standard library http.server, POST /optimize
  3. Frontend             - HTML + CSS + JavaScript (stored as text below)
"""
import json
import threading
import webbrowser
from functools import reduce
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import gcd


# ============ 1. OPTIMIZATION ENGINE (0/1 Knapsack + Dynamic Programming) ============
def knapsack(sub, cap):
    """Return indices (into `sub`) of the subset with max total gain whose total cost <= cap."""
    if not sub or cap <= 0:
        return [], 0
    g = reduce(gcd, (it["cost"] for it in sub))      # shrink the table; answer is unchanged
    c = cap // g
    cost = [it["cost"] // g for it in sub]
    val = [it["gain"] for it in sub]
    n = len(sub)
    # dp[i][b] = best total value using the first i products with budget b
    dp = [[0] * (c + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for b in range(c + 1):
            dp[i][b] = dp[i - 1][b]                          # exclude product i
            if cost[i - 1] <= b:                              # include product i (if affordable)
                with_item = dp[i - 1][b - cost[i - 1]] + val[i - 1]
                if with_item > dp[i][b]:
                    dp[i][b] = with_item
    chosen, b = [], c                                         # trace back the chosen products
    for i in range(n, 0, -1):
        if dp[i][b] != dp[i - 1][b]:
            chosen.append(i - 1)
            b -= cost[i - 1]
    chosen.reverse()
    return chosen, n * (c + 1)


def solve(items, budget, lock=True):
    """items: dicts with id, name, price, value, priority, qty. Cost = price*qty, gain = value*qty.
    If lock is True, 'Must Have' products are bought first and the DP optimizes the rest."""
    for it in items:
        it["cost"], it["gain"] = it["price"] * it["qty"], it["value"] * it["qty"]
    musts = [i for i, it in enumerate(items) if lock and it["priority"] == "Must Have"]
    must_cost = sum(items[i]["cost"] for i in musts)
    warn = ""
    if must_cost > budget:
        warn = "Your Must Have items alone cost more than the budget, so they could not all be locked in."
        musts, must_cost = [], 0
    rest = [i for i in range(len(items)) if i not in musts]
    picked, cells = knapsack([items[i] for i in rest], budget - must_cost)
    chosen = set(musts) | {rest[k] for k in picked}
    selected = [items[i] for i in range(len(items)) if i in chosen]
    rejected = []
    for i, it in enumerate(items):
        if i not in chosen:
            rejected.append(dict(it, reason=f"Costs ₹{it['cost']:,} for {it['gain']:g} value - other items gave more total value within ₹{budget:,}."))
    total_cost = sum(it["cost"] for it in selected)
    return {"budget": budget, "selected": selected, "rejected": rejected, "total_cost": total_cost,
            "remaining": budget - total_cost, "total_value": sum(it["gain"] for it in selected),
            "warn": warn, "dp_cells": cells}


# ============ 2. PYTHON BACKEND ============
PRIORITIES = ("Must Have", "Important", "Optional")
MAX_ITEMS = 100
MAX_TABLE = 20_000_000          # limit on DP table cells so one request cannot hog memory


def parse_request(data):
    """Validate the JSON sent by the browser. Returns (items, budget, lock) or raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError("Invalid request.")
    budget = data.get("budget")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ValueError("Budget must be a whole number (0 or more).")
    raw = data.get("items")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Add at least one product.")
    if len(raw) > MAX_ITEMS:
        raise ValueError(f"At most {MAX_ITEMS} products are allowed.")
    items = []
    for r in raw:
        if not isinstance(r, dict):
            raise ValueError("Invalid product.")
        name = str(r.get("name", "")).strip()[:60]
        price, value, pr, qty = r.get("price"), r.get("value"), r.get("priority"), r.get("qty", 1)
        if not name:
            raise ValueError("Every product needs a name.")
        if not isinstance(price, int) or isinstance(price, bool) or price <= 0:
            raise ValueError(f"Price of '{name}' must be a whole number above 0.")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 10:
            raise ValueError(f"Value of '{name}' must be between 0 and 10.")
        if not isinstance(qty, int) or isinstance(qty, bool) or not 1 <= qty <= 99:
            raise ValueError(f"Quantity of '{name}' must be 1 to 99.")
        if pr not in PRIORITIES:
            raise ValueError(f"Priority of '{name}' is invalid.")
        items.append({"id": str(r.get("id", name))[:40], "name": name, "price": price,
                      "value": value, "priority": pr, "qty": qty})
    g = reduce(gcd, (i["price"] * i["qty"] for i in items))
    if len(items) * (budget // g + 1) > MAX_TABLE:
        raise ValueError("Budget is too large for this many products.")
    return items, budget, data.get("lock") is not False


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        page = {"/": (HTML, "text/html"), "/index.html": (HTML, "text/html"),
                "/style.css": (CSS, "text/css"), "/script.js": (JS, "application/javascript")}.get(path)
        if page is None:
            return self._send(404, json.dumps({"error": "Not found"}))
        self._send(200, page[0], page[1])

    def do_POST(self):
        if self.path != "/optimize":
            return self._send(404, json.dumps({"error": "Not found"}))
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 200_000:
                raise ValueError("Request too large.")
            items, budget, lock = parse_request(json.loads(self.rfile.read(length) or b"{}"))
            self._send(200, json.dumps(solve(items, budget, lock)))
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "Invalid JSON."}))
        except ValueError as e:
            self._send(400, json.dumps({"error": str(e)}))


# ============ 3. FRONTEND (HTML + CSS + JavaScript) ============
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SmartBasket - Shop smarter. Stay within budget.</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header><div class="hd"><div class="logo">🛒 SmartBasket</div><nav id="nav"></nav></div></header>
  <main id="app"></main>
  <div id="toast"></div>
  <script src="script.js"></script>
</body>
</html>
"""

CSS = r""":root{--blue:#2563eb;--navy:#0f2a5c;--g:#10b981;--bg:#f3f7fd;--line:#e2eaf6;--mut:#64748b;--red:#ef4444;--amb:#f59e0b}
*{box-sizing:border-box}
body{margin:0;font-family:Inter,"Segoe UI",system-ui,sans-serif;background:var(--bg);color:#1e293b;padding-bottom:84px}
header{background:linear-gradient(120deg,var(--navy),var(--blue));color:#fff;position:sticky;top:0;z-index:5;box-shadow:0 4px 18px rgba(15,42,92,.25)}
.hd{max-width:1080px;margin:auto;padding:12px 16px;display:flex;align-items:center;justify-content:space-between}
.logo{font-weight:800;font-size:1.3rem;letter-spacing:-.3px}
nav{display:flex;gap:4px}
nav button{background:none;border:0;color:#dbe7ff;padding:8px 14px;border-radius:999px;font:600 .9rem inherit;cursor:pointer;transition:.2s}
nav button.on{background:#fff;color:var(--navy)}
main{max-width:1080px;margin:auto;padding:18px 16px}
.view{animation:up .4s ease}
@keyframes up{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
h1{font-size:clamp(1.7rem,5vw,2.5rem);margin:0 0 8px;color:var(--navy);letter-spacing:-.5px}
h2{margin:26px 0 12px;color:var(--navy);font-size:1.25rem}
h4{margin:0 0 10px;color:var(--navy)}
.muted{color:var(--mut)}.err{color:var(--red);font-weight:600;min-height:1.2em;margin:6px 0}
.hero{display:grid;grid-template-columns:1.3fr 1fr;gap:20px;align-items:center}
.card,.bcard,.alt,.empty{background:#fff;border-radius:20px;box-shadow:0 8px 28px rgba(15,42,92,.08);padding:20px;border:1px solid var(--line)}
.bcard{display:flex;gap:18px;align-items:center;flex-wrap:wrap;justify-content:center}
.leg{flex:1;min-width:170px}.leg div{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px dashed var(--line)}
.bar{height:12px;background:#e5edf9;border-radius:99px;overflow:hidden;margin-top:12px;width:100%}
.bar i{display:block;height:100%;border-radius:99px;transition:width .8s}
label{display:block;font-size:.8rem;font-weight:700;color:var(--mut);margin-bottom:4px}
input,select{width:100%;padding:11px 12px;border:1.5px solid #cfdcf3;border-radius:12px;font:1rem inherit;background:#fff}
input:focus,select:focus{outline:2px solid var(--blue);border-color:transparent}
.inl{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}.inl input{flex:1;min-width:120px}
.btn{padding:11px 18px;border-radius:12px;border:1.5px solid var(--blue);background:#fff;color:var(--blue);font:700 .95rem inherit;cursor:pointer;transition:.2s}
.btn:hover{transform:translateY(-1px);box-shadow:0 6px 16px rgba(37,99,235,.2)}
.btn.pri{background:var(--blue);color:#fff}.btn.ghost{background:#fff}
.btn.cta{background:linear-gradient(120deg,var(--blue),var(--g));color:#fff;border:0;padding:14px 26px;font-size:1.05rem}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:20px}
.stat{background:#fff;border-radius:18px;padding:16px;border:1px solid var(--line);box-shadow:0 6px 18px rgba(15,42,92,.06)}
.stat span{font-size:.8rem;color:var(--mut);font-weight:600}.stat b{display:block;font-size:1.5rem;color:var(--navy);margin-top:4px}
.stat.neg b{color:var(--red)}.stat.pos b{color:var(--g)}
.tools{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}.tools input{flex:2;min-width:180px}.tools select{flex:1;min-width:150px}
.chips{display:flex;gap:8px;overflow-x:auto;padding-bottom:6px;margin-bottom:12px}
.chip{white-space:nowrap;padding:8px 14px;border-radius:99px;border:1.5px solid var(--line);background:#fff;font:600 .85rem inherit;cursor:pointer}
.chip.on{background:var(--navy);color:#fff;border-color:var(--navy)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}
.pc{background:#fff;border-radius:20px;padding:16px;border:2px solid transparent;box-shadow:0 6px 20px rgba(15,42,92,.07);display:grid;gap:10px;transition:.25s}
.pc:hover{transform:translateY(-3px);box-shadow:0 12px 28px rgba(15,42,92,.13)}.pc.in{border-color:var(--g)}
.pc>.ic{width:64px;height:64px;font-size:2rem}
.ic{display:grid;place-items:center;background:linear-gradient(135deg,#e6f0ff,#e3f8ef);border-radius:18px;flex:none}.ic.sm{width:46px;height:46px;font-size:1.5rem;border-radius:14px}
.pi small{display:block;color:var(--mut)}.pm{display:flex;justify-content:space-between;align-items:center}
.price{font-size:1.3rem;font-weight:800;color:var(--navy)}.val{background:#fff7e0;color:#a16207;padding:3px 10px;border-radius:99px;font-weight:700;font-size:.85rem}
.pr{display:flex;gap:4px;align-items:center}
.pb{border:1.5px solid var(--line);background:#fff;border-radius:99px;padding:3px 7px;cursor:pointer;opacity:.5}.pb.on{opacity:1}
.pb.on.must{border-color:var(--red)}.pb.on.imp{border-color:var(--amb)}.pb.on.opt{border-color:var(--g)}
.plabel{font-size:.78rem;font-weight:700;padding:3px 10px;border-radius:99px;margin-left:4px;display:inline-block}
.plabel.must{background:#fee2e2;color:#b91c1c}.plabel.imp{background:#fef3c7;color:#92400e}.plabel.opt{background:#d1fae5;color:#065f46}
.pa{display:flex;gap:10px;align-items:center;justify-content:space-between}
.qty{display:flex;align-items:center;gap:8px;background:var(--bg);border-radius:99px;padding:3px}
.qty button{width:28px;height:28px;border-radius:50%;border:0;background:#fff;font-weight:800;cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,.12)}
.qty span{min-width:20px;text-align:center;font-weight:700}
.row{display:flex;align-items:center;gap:12px;background:#fff;border-radius:16px;padding:12px;margin-bottom:10px;border:1px solid var(--line);flex-wrap:wrap}
.row .grow{flex:1;min-width:130px}.row small{color:var(--mut);display:block;margin-top:2px}.row.no{opacity:.85;background:#fafcff}
.x{border:0;background:none;color:var(--red);font-size:1.1rem;cursor:pointer}
.banner{padding:14px 18px;border-radius:16px;font-weight:700;margin:14px 0}
.banner.ok{background:#d1fae5;color:#065f46}.banner.bad{background:#fee2e2;color:#991b1b}.banner.warn{background:#fef3c7;color:#92400e}
.alt{margin-top:14px;border-color:#fcd34d;background:#fffbeb}
.empty{text-align:center;padding:40px 20px}.empty .big{font-size:3rem}
.acts{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
details.card{margin-bottom:14px}summary{cursor:pointer;font-weight:700;color:var(--blue)}
.form{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px;align-items:end}
#toast{position:fixed;left:50%;bottom:92px;transform:translate(-50%,30px);background:var(--navy);color:#fff;padding:12px 20px;border-radius:14px;opacity:0;pointer-events:none;transition:.3s;z-index:9;font-weight:600}
#toast.show{opacity:1;transform:translate(-50%,0)}#toast.err{background:var(--red)}
@media(max-width:760px){
 .hero{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}
 nav{position:fixed;bottom:0;left:0;right:0;background:#fff;justify-content:space-around;padding:8px;box-shadow:0 -4px 20px rgba(0,0,0,.12)}
 nav button{color:var(--mut);padding:8px 10px;font-size:.8rem}nav button.on{background:#e8f0ff;color:var(--blue)}}
"""

JS = r"""// SmartBasket frontend: catalog, basket, budget tracking, and calls POST /optimize (0/1 Knapsack DP).
const $ = (id) => document.getElementById(id);
const rs = (n) => (n < 0 ? "-" : "") + "₹" + Math.abs(Number(n)).toLocaleString("en-IN");
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const CATS = { "Groceries": "🌾", "Fruits & Vegetables": "🥕", "Personal Care": "🧴", "Snacks": "🍿", "Household": "🧽" };
const PRI = { "Must Have": ["🔴", "must"], "Important": ["🟡", "imp"], "Optional": ["🟢", "opt"] };
const D = (id, name, icon, cat, price, value, pr) => ({ id, name, icon, cat, price, value, pr });
const DEF = [
  D("rice", "Rice", "🍚", "Groceries", 300, 8, "Must Have"), D("milk", "Milk", "🥛", "Groceries", 100, 5, "Must Have"),
  D("fruits", "Fruits", "🍎", "Fruits & Vegetables", 250, 7, "Important"), D("snacks", "Snacks", "🍿", "Snacks", 200, 4, "Optional"),
  D("oil", "Oil", "🫒", "Groceries", 500, 9, "Important"),
  D("flour", "Wheat Flour", "🌾", "Groceries", 180, 6, "Important"), D("dal", "Lentils (Dal)", "🫘", "Groceries", 140, 6, "Important"),
  D("sunoil", "Sunflower Oil 500ml", "🌻", "Groceries", 160, 6, "Optional"),
  D("banana", "Bananas", "🍌", "Fruits & Vegetables", 80, 5, "Optional"), D("tomato", "Tomatoes", "🍅", "Fruits & Vegetables", 50, 4, "Important"),
  D("onion", "Onions", "🧅", "Fruits & Vegetables", 45, 4, "Important"),
  D("soap", "Soap", "🧼", "Personal Care", 40, 3, "Optional"), D("shampoo", "Shampoo", "🧴", "Personal Care", 180, 5, "Optional"),
  D("paste", "Toothpaste", "🪥", "Personal Care", 90, 6, "Important"),
  D("biscuit", "Biscuits", "🍪", "Snacks", 60, 3, "Optional"), D("chips", "Chips", "🍟", "Snacks", 50, 3, "Optional"),
  D("detergent", "Detergent", "🧺", "Household", 220, 6, "Important"), D("dish", "Dishwash Liquid", "🍽️", "Household", 90, 4, "Optional"),
  D("tissue", "Tissue Rolls", "🧻", "Household", 70, 3, "Optional"),
];
let S = {};
try { S = JSON.parse(localStorage.getItem("smartbasket")) || {}; } catch (e) {}
S = Object.assign({ budget: 1000, products: DEF.map((p) => ({ ...p })), cart: {}, q: {}, res: null, view: "home", cat: "All", sort: "name", search: "", lock: true, last: null }, S);
S.search = "";
const save = () => { try { localStorage.setItem("smartbasket", JSON.stringify(S)); } catch (e) {} };
const P = (id) => S.products.find((p) => p.id === id);
const qtyOf = (id) => S.cart[id] || S.q[id] || 1;

function totals() {
  let spent = 0, value = 0, n = 0;
  for (const id in S.cart) { const p = P(id); if (!p) continue; spent += p.price * S.cart[id]; value += p.value * S.cart[id]; n++; }
  return { spent, value, n, left: S.budget - spent };
}
function toast(msg, kind) {
  const t = $("toast"); t.textContent = msg; t.className = "show " + (kind || "");
  clearTimeout(toast.t); toast.t = setTimeout(() => (t.className = ""), 2600);
}

function donut(spent, budget) {
  const pct = budget > 0 ? spent / budget : 0, c = 2 * Math.PI * 52, f = Math.min(pct, 1) * c;
  const col = pct > 1 ? "#ef4444" : pct > 0.85 ? "#f59e0b" : "#10b981";
  return `<svg viewBox="0 0 130 130" width="150" height="150"><circle cx="65" cy="65" r="52" fill="none" stroke="#e5edf9" stroke-width="13"/>
  <circle cx="65" cy="65" r="52" fill="none" stroke="${col}" stroke-width="13" stroke-linecap="round" stroke-dasharray="${f} ${c}" transform="rotate(-90 65 65)"/>
  <text x="65" y="64" text-anchor="middle" font-size="22" font-weight="700" fill="#0f2a5c">${Math.round(pct * 100)}%</text>
  <text x="65" y="80" text-anchor="middle" font-size="9" fill="#64748b">of budget used</text></svg>`;
}
function budgetCard(spent, budget) {
  const pct = budget > 0 ? Math.min(spent / budget, 1) * 100 : 0, left = budget - spent;
  const col = left < 0 ? "#ef4444" : pct > 85 ? "#f59e0b" : "#10b981";
  return `${donut(spent, budget)}<div class="leg"><div><span>Budget</span><b>${rs(budget)}</b></div><div><span>Spent</span><b>${rs(spent)}</b></div>
  <div><span>Remaining</span><b style="color:${col}">${rs(left)}</b></div></div><div class="bar"><i style="width:${pct}%;background:${col}"></i></div>`;
}
const stat = (label, val, cls) => `<div class="stat ${cls || ""}"><span>${label}</span><b>${val}</b></div>`;

function card(p) {
  const inC = p.id in S.cart, q = qtyOf(p.id);
  return `<div class="pc ${inC ? "in" : ""}"><div class="ic">${p.icon}</div>
  <div class="pi"><b>${esc(p.name)}</b><small>${esc(p.cat)}</small></div>
  <div class="pm"><span class="price">${rs(p.price)}</span><span class="val">★ Value ${p.value}</span></div>
  <div class="pr">${Object.keys(PRI).map((k) => `<button class="pb ${p.pr === k ? "on " + PRI[k][1] : ""}" title="${k}" data-a="pri" data-id="${p.id}" data-v="${k}">${PRI[k][0]}</button>`).join("")}<span class="plabel ${PRI[p.pr][1]}">${p.pr}</span></div>
  <div class="pa"><div class="qty"><button data-a="q-" data-id="${p.id}">−</button><span>${q}</span><button data-a="q+" data-id="${p.id}">+</button></div>
  <button class="btn ${inC ? "ghost" : "pri"}" data-a="${inC ? "rm" : "add"}" data-id="${p.id}">${inC ? "Remove" : "Add to basket"}</button></div></div>`;
}
function alts(p) {
  return S.products.filter((x) => x.cat === p.cat && x.id !== p.id && !(x.id in S.cart) && x.price < p.price)
    .sort((a, b) => b.value / b.price - a.value / a.price).slice(0, 3);
}
function altBlock(p) {
  const a = alts(p);
  return `<div class="alt"><h4>💡 ${esc(p.name)}: This item exceeds your budget. Try these affordable alternatives.</h4>` +
    (a.length ? `<div class="grid">${a.map(card).join("")}</div>` : `<p class="muted">No cheaper alternatives in ${esc(p.cat)} yet. Add one in Products.</p>`) + `</div>`;
}
const empty = (icon, title, text, btn) => `<div class="empty"><div class="big">${icon}</div><h2>${title}</h2><p class="muted">${text}</p>${btn || ""}</div>`;

function vHome() {
  const t = totals();
  return `<section class="hero"><div><h1>Shop smarter. Stay within budget.</h1>
  <p class="muted">Pick what you need and set your budget. SmartBasket finds the highest-value basket using 0/1 Knapsack dynamic programming.</p>
  <label>Your budget (₹)</label><div class="inl"><input id="bud" type="number" min="1" step="1" value="${S.budget}"><button class="btn pri" data-a="setbud">Edit budget</button></div>
  <div class="inl"><button class="btn cta" data-a="opt">✨ Optimize My Basket</button><button class="btn ghost" data-a="sample">Load sample (₹1,000)</button></div></div>
  <div class="bcard">${budgetCard(t.spent, S.budget)}</div></section>
  <div class="stats">${stat("💳 Current spending", rs(t.spent))}${stat("🟢 Remaining budget", rs(t.left), t.left < 0 ? "neg" : "pos")}
  ${stat("🛍️ Products selected", t.n)}${stat("⭐ Total value score", t.value)}</div>
  ${t.left < 0 ? `<div class="banner bad">⚠️ You are ${rs(-t.left)} over budget. Tap "Optimize My Basket" to fix it.</div>` : ""}`;
}
function list() {
  const f = { name: (a, b) => a.name.localeCompare(b.name), pa: (a, b) => a.price - b.price, pd: (a, b) => b.price - a.price,
    vd: (a, b) => b.value - a.value, bv: (a, b) => b.value / b.price - a.value / a.price };
  return S.products.filter((p) => (S.cat === "All" || p.cat === S.cat) && p.name.toLowerCase().includes(S.search.toLowerCase())).sort(f[S.sort]);
}
function gridHtml() {
  const l = list();
  return l.length ? l.map(card).join("") : empty("🔍", "No products found", "Try a different search or category.");
}
function vProducts() {
  const so = { name: "Name (A-Z)", pa: "Price: low to high", pd: "Price: high to low", vd: "Value: high to low", bv: "Best value per ₹" };
  return `<h2 style="margin-top:0">Products</h2>
  <div class="tools"><input id="q" placeholder="🔍 Search products" value="${esc(S.search)}">
  <select id="sort">${Object.keys(so).map((k) => `<option value="${k}" ${S.sort === k ? "selected" : ""}>${so[k]}</option>`).join("")}</select></div>
  <div class="chips">${["All", ...Object.keys(CATS)].map((c) => `<button class="chip ${S.cat === c ? "on" : ""}" data-a="cat" data-v="${c}">${CATS[c] || "🛒"} ${c}</button>`).join("")}</div>
  <details class="card"><summary>➕ Add your own product</summary><div class="form">
  <div><label>Name</label><input id="cn" maxlength="60"></div>
  <div><label>Category</label><select id="cc">${Object.keys(CATS).map((c) => `<option>${c}</option>`).join("")}</select></div>
  <div><label>Price (₹)</label><input id="cp" type="number" min="1" step="1"></div>
  <div><label>Value (0-10)</label><input id="cv" type="number" min="0" max="10" step="1"></div>
  <div><label>Priority</label><select id="cr">${Object.keys(PRI).map((k) => `<option>${k}</option>`).join("")}</select></div>
  <button class="btn pri" data-a="addp">Add product</button></div><p id="cerr" class="err"></p></details>
  <div class="grid" id="grid">${gridHtml()}</div>`;
}
function vBasket() {
  const t = totals(), ids = Object.keys(S.cart).filter((id) => P(id));
  if (!ids.length) return empty("🧺", "Your basket is empty", "Add products you need, then let SmartBasket optimize them.", `<button class="btn pri" data-a="nav" data-v="products">Browse products</button>`);
  const order = { "Must Have": 0, "Important": 1, "Optional": 2 };
  ids.sort((a, b) => order[P(a).pr] - order[P(b).pr]);
  const rows = ids.map((id) => { const p = P(id), q = S.cart[id]; return `<div class="row"><div class="ic sm">${p.icon}</div>
    <div class="grow"><b>${esc(p.name)}</b><small>${rs(p.price)} each · Value ${p.value}</small><span class="plabel ${PRI[p.pr][1]}" style="margin:4px 0 0">${PRI[p.pr][0]} ${p.pr}</span></div>
    <div class="qty"><button data-a="q-" data-id="${id}">−</button><span>${q}</span><button data-a="q+" data-id="${id}">+</button></div>
    <b>${rs(p.price * q)}</b><button class="x" title="Remove" data-a="rm" data-id="${id}">✕</button></div>`; }).join("");
  const over = t.left < 0, target = P(S.last) && S.last in S.cart ? P(S.last) : P(ids.reduce((a, b) => (P(a).price * S.cart[a] >= P(b).price * S.cart[b] ? a : b)));
  return `<h2 style="margin-top:0">My Basket</h2><div class="bcard" style="margin-bottom:14px">${budgetCard(t.spent, S.budget)}</div>
  <div class="banner ${over ? "bad" : "ok"}">${over ? `⚠️ Over budget by ${rs(-t.left)}. Remove something or optimize.` : `✅ Within budget. ${rs(t.left)} left.`}</div>
  ${rows}<div class="acts"><button class="btn cta" data-a="opt">✨ Optimize My Basket</button><button class="btn" data-a="savelist">💾 Save shopping list</button><button class="btn" data-a="reset">Reset basket</button></div>
  ${over ? altBlock(target) : ""}`;
}
function resRow(it, no) {
  const p = P(it.id) || { icon: "🛒", pr: it.priority };
  return `<div class="row ${no ? "no" : ""}"><div class="ic sm">${p.icon}</div><div class="grow"><b>${esc(it.name)}${it.qty > 1 ? " ×" + it.qty : ""}</b>
  <small>${no ? esc(it.reason) : "Value " + it.gain}</small><span class="plabel ${PRI[it.priority][1]}" style="margin:4px 0 0">${PRI[it.priority][0]} ${it.priority}</span></div><b>${rs(it.cost)}</b></div>`;
}
function vOpt() {
  const r = S.res;
  if (!r) return empty("🧠", "Nothing optimized yet", "Add products to your basket, then run the optimizer.", `<button class="btn cta" data-a="opt">✨ Optimize My Basket</button>`);
  const order = { "Must Have": 0, "Important": 1, "Optional": 2 }, srt = (a) => [...a].sort((x, y) => order[x.priority] - order[y.priority]);
  const good = r.selected.length > 0;
  return `<h2 style="margin-top:0">Your Smart Basket</h2>
  <div class="banner ${good ? "ok" : "bad"}">${good ? "Great! Your basket is within budget 🎉" : "Nothing fits in this budget. Try increasing it."}</div>
  ${r.warn ? `<div class="banner warn">⚠️ ${esc(r.warn)}</div>` : ""}
  <div class="hero"><div class="bcard">${budgetCard(r.total_cost, r.budget)}</div>
  <div class="stats" style="margin:0;grid-template-columns:1fr 1fr">${stat("Total cost", rs(r.total_cost))}${stat("Remaining", rs(r.remaining), "pos")}${stat("Total value", r.total_value)}${stat("Not selected", r.rejected.length)}</div></div>
  <label style="margin-top:14px"><input type="checkbox" id="lock" style="width:auto" ${S.lock ? "checked" : ""}> Always keep 🔴 Must Have items (re-run to apply)</label>
  <div class="acts"><button class="btn pri" data-a="apply">Apply to my basket</button><button class="btn" data-a="opt">Re-run optimizer</button><button class="btn" data-a="savelist">💾 Save list</button></div>
  <h2>✅ Selected products</h2>${srt(r.selected).map((i) => resRow(i)).join("") || `<p class="muted">None.</p>`}
  <h2>✗ Products not selected</h2>${srt(r.rejected).map((i) => resRow(i, true)).join("") || `<p class="muted">Everything fit!</p>`}
  ${srt(r.rejected).slice(0, 2).map((i) => P(i.id) ? altBlock(P(i.id)) : "").join("")}
  <p class="muted">Solved with 0/1 Knapsack DP over ${r.dp_cells.toLocaleString("en-IN")} table cells.</p>`;
}

function draw() {
  const views = { home: vHome, products: vProducts, basket: vBasket, opt: vOpt };
  const names = [["home", "🏠 Home"], ["products", "🛍️ Products"], ["basket", "🧺 My Basket (" + totals().n + ")"], ["opt", "🧠 Optimization"]];
  $("nav").innerHTML = names.map(([k, n]) => `<button class="${S.view === k ? "on" : ""}" data-a="nav" data-v="${k}">${n}</button>`).join("");
  $("app").innerHTML = `<div class="view">${views[S.view]()}</div>`;
  save();
}
async function optimize() {
  const ids = Object.keys(S.cart).filter((id) => P(id));
  if (!Number.isInteger(S.budget) || S.budget <= 0) return toast("Set a valid budget first.", "err");
  if (!ids.length) { toast("Your basket is empty. Add some products first.", "err"); S.view = "products"; return draw(); }
  const items = ids.map((id) => { const p = P(id); return { id, name: p.name, price: p.price, value: p.value, priority: p.pr, qty: S.cart[id] }; });
  try {
    const res = await fetch("/optimize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ budget: S.budget, lock: S.lock, items }) });
    const d = await res.json();
    if (!res.ok) return toast(d.error || "Something went wrong.", "err");
    S.res = d; S.view = "opt"; draw(); toast("Basket optimized ✨"); window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (e) { toast("Cannot reach the server. Is the Python app running?", "err"); }
}
function saveList() {
  const t = totals(); if (!t.n) return toast("Your basket is empty.", "err");
  const L = ["SmartBasket shopping list", "Budget: " + rs(S.budget), ""];
  Object.keys(S.cart).forEach((id) => { const p = P(id), q = S.cart[id]; if (p) L.push(`- ${p.name} x${q}  ${rs(p.price * q)}  [${p.pr}]`); });
  L.push("", "Total: " + rs(t.spent), "Remaining: " + rs(t.left), "Value score: " + t.value);
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([L.join("\n")], { type: "text/plain" }));
  a.download = "smartbasket-list.txt"; a.click(); toast("Shopping list saved 💾");
}
function addCustom() {
  const name = $("cn").value.trim(), price = Number($("cp").value), value = Number($("cv").value), err = (m) => ($("cerr").textContent = m);
  if (!name) return err("Enter a product name.");
  if (!$("cp").value || !Number.isInteger(price) || price <= 0) return err("Price must be a whole number above 0.");
  if ($("cv").value === "" || !(value >= 0 && value <= 10)) return err("Value must be between 0 and 10.");
  const cat = $("cc").value;
  S.products.push(D("c" + Date.now(), name.slice(0, 60), CATS[cat], cat, price, value, $("cr").value));
  toast("Product added ✅"); draw();
}

document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-a]"); if (!b) return;
  const a = b.dataset.a, id = b.dataset.id;
  if (a === "nav") { S.view = b.dataset.v; draw(); window.scrollTo(0, 0); }
  else if (a === "cat") { S.cat = b.dataset.v; draw(); }
  else if (a === "pri") { P(id).pr = b.dataset.v; draw(); }
  else if (a === "q+" || a === "q-") { const q = Math.min(99, Math.max(1, qtyOf(id) + (a === "q+" ? 1 : -1))); S.q[id] = q; if (id in S.cart) S.cart[id] = q; draw(); }
  else if (a === "add") {
    S.cart[id] = qtyOf(id); S.last = id; const t = totals(); draw();
    t.left < 0 ? toast("Over budget by " + rs(-t.left) + ". See alternatives in My Basket.", "err") : toast(P(id).name + " added 🛒");
  }
  else if (a === "rm") { delete S.cart[id]; draw(); }
  else if (a === "setbud") {
    const v = Number($("bud").value);
    if ($("bud").value === "" || !Number.isInteger(v) || v <= 0 || v > 10000000) return toast("Budget must be a whole number from 1 to 1,00,00,000.", "err");
    S.budget = v; draw(); toast("Budget updated to " + rs(v));
  }
  else if (a === "opt") optimize();
  else if (a === "sample") {
    ["rice", "milk", "fruits", "snacks", "oil"].forEach((k) => { const d = DEF.find((x) => x.id === k), p = P(k); if (p) Object.assign(p, d); S.cart[k] = 1; S.q[k] = 1; });
    S.budget = 1000; S.res = null; draw(); toast("Sample loaded. Try Optimize!");
  }
  else if (a === "reset") { S.cart = {}; S.q = {}; S.res = null; S.last = null; draw(); toast("Basket reset"); }
  else if (a === "savelist") saveList();
  else if (a === "apply") { S.cart = {}; S.res.selected.forEach((i) => (S.cart[i.id] = i.qty)); S.view = "basket"; draw(); toast("Optimized basket applied ✅"); }
  else if (a === "addp") addCustom();
});
document.addEventListener("input", (e) => { if (e.target.id === "q") { S.search = e.target.value; $("grid").innerHTML = gridHtml(); } });
document.addEventListener("change", (e) => {
  if (e.target.id === "sort") { S.sort = e.target.value; $("grid").innerHTML = gridHtml(); save(); }
  if (e.target.id === "lock") { S.lock = e.target.checked; save(); }
});
draw();
"""


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("SmartBasket running at http://localhost:8000  (Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:8000")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")