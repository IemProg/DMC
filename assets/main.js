/* DMC project page */
(function () {
  "use strict";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function (id) { return document.getElementById(id); };
  var lerp = function (a, b, t) { return a + (b - a) * t; };
  var clamp = function (x) { return Math.max(0, Math.min(1, x)); };
  var ease = function (t) { t = clamp(t); return t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  var fmt = function (v, d) { return (v < 0 ? "−" : "") + Math.abs(v).toFixed(d); };
  var signed = function (v) { return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2); };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  /* ---------- math ---------- */
  function renderMath() {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(document.body, {
      delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
      trust: function (c) { return c.command === "\\htmlClass"; },
      strict: false, throwOnError: false
    });
  }
  window.addEventListener("load", renderMath);

  /* ---------- hero animation: prompt space (left) + accuracy plane (right) ---------- */
  (function hero() {
    var L = $("hL"), R = $("hR"); if (!L || !R) return;
    var NS = "http://www.w3.org/2000/svg";
    var set = function (e, a) { for (var k in a) e.setAttribute(k, a[k]); };
    var P = function (a, b, t) { return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) }; };
    // left: goals and the line joining them
    var G = { x: 72, y: 262 }, S = { x: 408, y: 82 };
    var onLine = function (s) { return P(G, S, 0.14 + 0.72 * s); };      // single prompt at position s (0 = zero-shot side)
    var Gp = P(G, S, 0.10), Sp = P(G, S, 0.90);                            // DMC endpoints
    var ux = S.x - G.x, uy = S.y - G.y, ul = Math.hypot(ux, uy); ux /= ul; uy /= ul;
    var px = -uy, py = ux;
    // right: accuracy plane (schematic)
    var X = function (u) { return 60 + u * 400; }, Y = function (v) { return 300 - v * 280; };
    var front = function (s) { var th = s * Math.PI / 2; return { x: X(0.10 + 0.66 * Math.sin(th)), y: Y(0.10 + 0.66 * Math.cos(th)) }; };
    var dmc = function (a) { var th = Math.pow(a, 0.5) * Math.PI / 2; return { x: X(0.14 + 0.76 * Math.sin(th)), y: Y(0.14 + 0.76 * Math.cos(th)) }; };
    var path = function (f, a0, a1, n) { var d = ""; n = n || 60; for (var k = 0; k <= n; k++) { var q = f(lerp(a0, a1, k / n)); d += (k ? "L" : "M") + q.x.toFixed(1) + " " + q.y.toFixed(1); } return d; };
    // iso-HM contours: v = h u / (2u - h)
    (function iso() {
      var g = $("hIso");
      [0.35, 0.5, 0.65, 0.8].forEach(function (h) {
        var d = "", first = true;
        for (var k = 0; k <= 80; k++) {
          var u = h / 2 + 0.0015 + (1.2 - h / 2) * Math.pow(k / 80, 1.8), v = h * u / (2 * u - h);
          if (v > 1.3) continue;
          d += (first ? "M" : "L") + X(u).toFixed(1) + " " + Y(v).toFixed(1); first = false;
        }
        var p = document.createElementNS(NS, "path"); p.setAttribute("d", d); p.setAttribute("class", "iso"); g.appendChild(p);
      });
    })();
    var $$ = function (ids) { var o = {}; ids.split(" ").forEach(function (k) { o[k] = $(k); }); return o; };
    var e = $$("rOneL hRail hCorr hArrows hArS hArG hTxS hTxG hLam hGen hSpec hOne hTrav hOneL hGenL hSpecL hStarL hStarLt hFront hDmc hFrontT hDmcT hShift hShiftT rGen rSpec rOne rTrav hStarR");
    set(e.hRail, { x1: G.x, y1: G.y, x2: S.x, y2: S.y });
    var stepsLi = document.querySelectorAll(".steps li"), stepText = $("stepText");
    var TEXT = [
      "A single prompt receives both gradients and settles where they cancel: one point on the accuracy plane.",
      "Changing λ only moves the prompt along the line between the two goals. Every λ lands on the same trade-off curve.",
      "DMC gives each objective its own prompt. Each one moves toward its own goal.",
      "A corridor joins them. Its classifiers trace a curve beyond the single-prompt one; we deploy α\u00a0=\u00a00.20."
    ];
    // timeline (ms)
    var T = [0, 2600, 7400, 9300, 13200], END = 13200;
    var sOf = function (t) {                    // position of the single prompt during the lambda sweep
      if (t < T[1]) return 0.5;
      var k = clamp((t - T[1]) / (T[2] - T[1] - 300));
      var keys = [0.5, 0.06, 0.94, 0.5], seg = Math.min(2, Math.floor(k * 3)), f = ease(k * 3 - seg);
      return lerp(keys[seg], keys[seg + 1], f);
    };
    var sMin = 0.5, sMax = 0.5, cur = -1;
    function stage(t) { return t < T[1] ? 0 : t < T[2] ? 1 : t < T[3] ? 2 : 3; }
    function frame(t) {
      var st = stage(t);
      if (st !== cur) {
        cur = st;
        for (var k = 0; k < stepsLi.length; k++) stepsLi[k].classList.toggle("on", k === st);
        stepText.textContent = TEXT[st];
      }
      // ---- single prompt ----
      var s = sOf(t);
      if (t >= T[1] && t < T[2]) { sMin = Math.min(sMin, s); sMax = Math.max(sMax, s); }
      if (t < T[1]) { sMin = sMax = 0.5; }
      if (t >= T[2]) { sMin = 0.06; sMax = 0.94; }
      var w = t < T[1] ? Math.sin(t * 0.0105) * 0.6 + Math.sin(t * 0.0231) * 0.4 : 0;
      var c = onLine(s); c = { x: c.x + ux * 4 * w, y: c.y + uy * 4 * w };
      var split = ease((t - T[2]) / 1500);
      var oneA = t < T[2] ? 1 : 0;
      var arrA = 1 - clamp((t - T[2]) / 400);
      set(e.hArS, { x1: c.x + ux * 13, y1: c.y + uy * 13, x2: c.x + ux * (70 + 7 * w), y2: c.y + uy * (70 + 7 * w) });
      set(e.hArG, { x1: c.x - ux * 13, y1: c.y - uy * 13, x2: c.x - ux * (70 - 7 * w), y2: c.y - uy * (70 - 7 * w) });
      set(e.hTxS, { x: c.x + ux * 48 - px * 22, y: c.y + uy * 48 - py * 22 });
      set(e.hTxG, { x: c.x - ux * 48 + px * 22, y: c.y - uy * 48 + py * 22 + 12 });
      e.hArrows.setAttribute("opacity", arrA);
      set(e.hOne, { cx: c.x, cy: c.y, opacity: oneA });
      set(e.hOneL, { x: c.x - px * 22, y: c.y - py * 22 + 7, opacity: oneA });
      var lamA = t >= T[1] && t < T[2] ? clamp((t - T[1]) / 300) * (1 - clamp((t - T[2] + 300) / 300)) : 0;
      set(e.hLam, { x: 462, y: 322, opacity: lamA });
      e.hLam.textContent = s < 0.4 ? "large λ: closer to zero-shot" : s > 0.6 ? "small λ: closer to base optimum" : "λ";
      // ---- DMC prompts ----
      var g = P(c, Gp, split), sp = P(c, Sp, split);
      var dA = t >= T[2] ? 1 : 0;
      set(e.hGen, { cx: g.x, cy: g.y, opacity: dA }); set(e.hSpec, { cx: sp.x, cy: sp.y, opacity: dA });
      var labA = clamp((t - T[2] - 900) / 500);
      set(e.hGenL, { x: g.x - px * 24, y: g.y - py * 24 + 7, opacity: labA });
      set(e.hSpecL, { x: sp.x + px * 26, y: sp.y + py * 26 + 7, opacity: labA });
      // corridor draw + traveler (stage 3), then star settles at alpha = 0.2
      var draw = clamp((t - T[3]) / 700);
      var ce = P(Gp, Sp, draw);
      set(e.hCorr, { x1: Gp.x, y1: Gp.y, x2: ce.x, y2: ce.y, opacity: t >= T[3] ? 1 : 0 });
      var trav = clamp((t - T[3] - 500) / 1900);            // alpha 0 -> 1
      var starK = ease((t - T[3] - 2500) / 900);             // alpha 1 -> 0.2
      var alpha = t < T[3] + 2500 ? ease(trav) : lerp(1, 0.2, starK);
      var travOn = t >= T[3] + 500 && t < T[3] + 2500;
      var q = P(Gp, Sp, alpha);
      set(e.hTrav, { cx: q.x, cy: q.y, opacity: travOn ? 1 : 0 });
      var starOn = t >= T[3] + 2500 ? 1 : 0;
      e.hStarL.setAttribute("transform", "translate(" + q.x + "," + q.y + ")"); e.hStarL.setAttribute("opacity", starOn);
      set(e.hStarLt, { x: q.x + px * 30, y: q.y + py * 30 + 10, opacity: clamp((t - T[3] - 3200) / 300) });
      // ---- right panel ----
      e.hFront.setAttribute("d", path(front, sMin, sMax));
      e.hFront.setAttribute("opacity", t >= T[1] ? 1 : 0);
      var fl = front(0.06); set(e.hFrontT, { x: 72, y: fl.y + 46, opacity: clamp((t - T[2] + 600) / 500) });
      var f = front(s); set(e.rOne, { cx: f.x, cy: f.y });
      set(e.rOneL, { x: f.x - 16, y: f.y - 12, opacity: t < T[3] + 3300 ? 1 : 0 });
      var d0 = dmc(0), d1 = dmc(1);
      var r0 = P(f, d0, split), r1 = P(f, d1, split);
      set(e.rGen, { cx: r0.x, cy: r0.y, opacity: dA }); set(e.rSpec, { cx: r1.x, cy: r1.y, opacity: dA });
      var dmcTo = t < T[3] + 2500 ? ease(trav) : 1;
      e.hDmc.setAttribute("d", dmcTo > 0.002 ? path(dmc, 0, dmcTo, 80) : "");
      set(e.hDmcT, { x: dmc(0).x + 14, y: dmc(0).y - 12, opacity: clamp((t - T[3] - 1800) / 500) });
      var rq = dmc(alpha);
      set(e.rTrav, { cx: rq.x, cy: rq.y, opacity: travOn ? 1 : 0 });
      e.hStarR.setAttribute("transform", "translate(" + rq.x + "," + rq.y + ")"); e.hStarR.setAttribute("opacity", starOn);
      var shA = clamp((t - T[3] - 3300) / 500), fm = front(0.5), sr = dmc(0.2);
      var sh = P(fm, sr, 0.82);
      set(e.hShift, { x1: fm.x + 7, y1: fm.y - 7, x2: lerp(fm.x + 7, sh.x, shA), y2: lerp(fm.y - 7, sh.y, shA), opacity: shA > 0 ? 1 : 0 });
      set(e.hShiftT, { x: fm.x - 12, y: fm.y + 24, opacity: clamp((t - T[3] - 3700) / 400) });
    }
    var t0 = null, raf = null, off = 0;
    function loop(now) {
      if (t0 === null) t0 = now - off;
      var t = now - t0; frame(t);
      if (t < END + 3000) raf = requestAnimationFrame(loop); else raf = null;
    }
    function playFrom(k) {
      if (raf) cancelAnimationFrame(raf);
      cur = -1;
      if (reduce) { frame(END + 3000); return; }
      off = T[k]; if (k >= 2) { sMin = 0.06; sMax = 0.94; } t0 = null; raf = requestAnimationFrame(loop);
    }
    frame(0);
    var started = false;
    onVisible($("teaser"), function () { if (!started) { started = true; setTimeout(function () { playFrom(0); }, reduce ? 0 : 250); } }, 0.35);
    $("replay").addEventListener("click", function () { playFrom(0); });
    [L, R].forEach(function (s) { s.addEventListener("click", function () { playFrom(0); }); });
    stepsLi.forEach(function (li) { li.querySelector("button").addEventListener("click", function () { playFrom(+li.dataset.step); }); });
    if (reduce) $("replay").hidden = true;
  })();

  /* ---------- trade-off bars ---------- */
  (function trade() {
    var rows = document.querySelectorAll("#tradeChart .brow"); if (!rows.length) return;
    rows.forEach(function (r) {
      var from = +r.dataset.from;
      r.querySelector(".fill").style.width = from + "%";
      r.querySelector(".ghost").style.left = from + "%";
    });
    function run() {
      rows.forEach(function (r, i) {
        var from = +r.dataset.from, to = +r.dataset.to, val = r.querySelector(".bval");
        setTimeout(function () {
          r.classList.add("is-to");
          r.querySelector(".fill").style.width = to + "%";
          var t0 = null, dur = reduce ? 0 : 1100;
          (function tick(now) {
            if (t0 === null) t0 = now;
            var k = dur ? ease((now - t0) / dur) : 1;
            val.textContent = lerp(from, to, k).toFixed(1);
            if (k < 1) requestAnimationFrame(tick);
            else { var b = el("b", null, r.dataset.d); val.appendChild(b); }
          })(performance.now());
        }, reduce ? 0 : 250 + i * 160);
      });
    }
    onVisible($("tradeChart"), run, 0.6);
  })();

  /* ---------- gamma bars ---------- */
  (function gamma() {
    var host = $("gbars"); if (!host) return;
    var data = [["Flowers102", -0.955, -0.923], ["Food101", -0.776, -0.528], ["Caltech101", -0.716, -0.289], ["OxfordPets", -0.211, -0.580]];
    data.forEach(function (d) {
      var row = el("div", "grow");
      row.appendChild(el("span", "blab", d[0].replace(/\d+$/, "")));
      var pair = el("div", "gpair");
      [[d[1], "", "CoOp"], [d[2], "kg", "KgCoOp"]].forEach(function (b) {
        var bar = el("div", "gbar " + b[1]);
        var i = el("i"); i.style.width = (-b[0] * 100) + "%"; bar.appendChild(i);
        bar.setAttribute("data-tip", d[0] + ", " + b[2] + ": γ = " + fmt(b[0], 3));
        pair.appendChild(bar);
      });
      row.appendChild(pair);
      var vals = el("span", "bval"); vals.appendChild(el("span", null, fmt(d[1], 2))); vals.appendChild(el("span", null, fmt(d[2], 2)));
      row.appendChild(vals);
      host.appendChild(row);
    });
  })();

  /* ---------- heatmap: DMC minus MergeTune HM ---------- */
  (function heat() {
    var wide = $("heatWide"), tall = $("heatTall"); if (!wide) return;
    var ds = ["ImageNet", "Caltech101", "OxfordPets", "StanfordCars", "Flowers102", "Food101", "FGVCAircraft", "SUN397", "DTD", "EuroSAT", "UCF101"];
    var ab = ["IN", "Cal", "Pets", "Cars", "Flo", "Food", "Air", "SUN", "DTD", "Euro", "UCF"];
    // HM from the paper's main table (3-seed means at alpha = 0.20)
    var mt = {
      CoOp:   [72.89, 96.27, 96.13, 73.38, 83.92, 91.05, 33.33, 77.85, 64.26, 73.45, 78.42, 76.45],
      KgCoOp: [72.82, 95.94, 96.28, 74.90, 83.39, 91.08, 34.10, 78.03, 62.66, 73.44, 78.08, 76.43]
    };
    var dmc = {
      CoOp:   [72.90, 96.41, 96.04, 75.21, 84.47, 91.13, 35.23, 78.75, 63.53, 75.75, 80.07, 77.23],
      KgCoOp: [73.09, 96.11, 96.23, 75.38, 83.67, 91.19, 33.17, 78.85, 64.34, 73.38, 78.90, 76.76]
    };
    function cell(base, j, isAvg) {
      var v = Math.round((dmc[base][j] - mt[base][j]) * 100) / 100;
      var c = el("div", "cell" + (isAvg ? " avg" : ""), signed(v));
      var pct = Math.min(88, Math.abs(v) / 2.3 * 80 + 8);
      var hue = v >= 0 ? "var(--corr)" : "var(--neg)";
      c.style.background = "color-mix(in oklab, " + hue + " " + pct.toFixed(0) + "%, var(--bg))";
      if (pct > 58) c.classList.add("dark");
      var name = isAvg ? "Average over 11 datasets" : ds[j];
      c.setAttribute("data-tip", base + ", " + name + ": MergeTune " + mt[base][j].toFixed(2) + " → DMC " + dmc[base][j].toFixed(2));
      c.setAttribute("aria-label", c.getAttribute("data-tip"));
      return c;
    }
    // wide layout: rows = base methods
    wide.appendChild(el("div", "hh", ""));
    ab.forEach(function (a) { wide.appendChild(el("div", "hh", a)); });
    wide.appendChild(el("div", "", ""));
    wide.appendChild(el("div", "hh", "Avg"));
    ["CoOp", "KgCoOp"].forEach(function (b) {
      wide.appendChild(el("div", "rl", b));
      for (var j = 0; j < 11; j++) wide.appendChild(cell(b, j, false));
      wide.appendChild(el("div", "", ""));
      wide.appendChild(cell(b, 11, true));
    });
    // tall layout: rows = datasets
    tall.appendChild(el("div", "hh", ""));
    tall.appendChild(el("div", "hh", "CoOp"));
    tall.appendChild(el("div", "hh", "KgCoOp"));
    for (var j = 0; j < 12; j++) {
      tall.appendChild(el("div", "rl", j < 11 ? ds[j] : "Average"));
      tall.appendChild(cell("CoOp", j, j === 11));
      tall.appendChild(cell("KgCoOp", j, j === 11));
    }
  })();

  /* ---------- parameter dot plot ---------- */
  (function dots() {
    var host = $("dots"); if (!host) return;
    var lo = 65, hi = 100;
    var pos = function (v) { return ((v - lo) / (hi - lo) * 100) + "%"; };
    var data = [
      ["Flowers102", 80.51, 77.84, 79.40, 84.47],
      ["Caltech101", 94.42, 92.19, 95.34, 96.41],
      ["StanfordCars", 71.98, 68.35, 70.21, 75.21],
      ["EuroSAT", 71.18, 69.95, 72.29, 75.75]
    ];
    var names = ["1 prompt, 2K", "1 prompt, 4K", "1 prompt, 8K", "DMC, 2 × 2K"];
    data.forEach(function (d) {
      var row = el("div", "drow");
      row.appendChild(el("span", "blab", d[0]));
      var tr = el("div", "dtrack");
      for (var k = 1; k <= 4; k++) {
        var dot = el("i", "dot d" + k);
        dot.style.left = pos(d[k]);
        dot.setAttribute("data-tip", d[0] + ", " + names[k - 1] + ": " + d[k].toFixed(2) + " HM");
        tr.appendChild(dot);
      }
      var lab = el("span", "dlab", d[4].toFixed(1)); lab.style.left = pos(d[4]); tr.appendChild(lab);
      row.appendChild(tr);
      host.appendChild(row);
    });
    var ax = document.querySelector(".daxis");
    [65, 75, 85, 95].forEach(function (v) { var s = el("span", null, v); s.style.left = pos(v); ax.appendChild(s); });
  })();

  /* ---------- cos bars ---------- */
  document.querySelectorAll("#cosChart .crow").forEach(function (r) {
    r.querySelector(".fill").style.width = (+r.dataset.v * 100) + "%";
  });

  /* ---------- alpha widget ---------- */
  (function widget() {
    var slider = $("alpha"); if (!slider) return;
    var A = { x: 80, y: 132 }, B = { x: 520, y: 62 };
    var at = function (a) { return { x: lerp(A.x, B.x, a), y: lerp(A.y, B.y, a) }; };
    var NS = "http://www.w3.org/2000/svg", ticks = $("wTicks");
    for (var i = 1; i < 10; i++) {
      var p = at(i / 10), c = document.createElementNS(NS, "circle");
      c.setAttribute("cx", p.x); c.setAttribute("cy", p.y); c.setAttribute("r", 2); c.setAttribute("class", "tick");
      ticks.appendChild(c);
    }
    var s02 = at(0.2); $("wStar").setAttribute("transform", "translate(" + s02.x + "," + s02.y + ")");
    var star = document.querySelector("#wStar .star");
    var pt = $("wPt"), out = $("alphaOut"), wg = $("wg"), ws = $("ws"), txt = $("wText");
    function describe(a) {
      if (a === 0) return "The generalization endpoint alone: text features held close to zero-shot CLIP.";
      if (a === 1) return "The specialization endpoint alone, trained by cross-entropy and the Visual Anchor. Best on base classes, furthest from zero-shot CLIP.";
      if (Math.abs(a - 0.2) < 1e-9) return "The deployed classifier. Mostly the generalization endpoint, with some base-class sharpness from the specialization endpoint. We use this value on every dataset.";
      if (a < 0.35) return "Mostly the generalization endpoint, with some base-class sharpness from the specialization endpoint.";
      if (a <= 0.65) return "An even mix. Because α is sampled at every training step, the corridor keeps this classifier accurate on base classes too.";
      return "Mostly the specialization endpoint: sharper on base classes, further from zero-shot CLIP, and weaker on novel ones.";
    }
    function update() {
      var a = Math.round(parseFloat(slider.value) * 100) / 100, p = at(a);
      pt.setAttribute("transform", "translate(" + p.x + "," + p.y + ")");
      out.textContent = a.toFixed(2);
      wg.textContent = (1 - a).toFixed(2); ws.textContent = a.toFixed(2);
      slider.setAttribute("aria-valuetext", "alpha " + a.toFixed(2));
      txt.textContent = describe(a);
      var on = Math.abs(a - 0.2) < 1e-9;
      star.classList.toggle("is-ghost", !on);
      pt.style.opacity = on ? 0 : 1;
    }
    slider.addEventListener("input", update);
    // click/drag on the segment
    var svg = $("wSvg"), drag = false;
    function fromEvent(e) {
      var r = svg.getBoundingClientRect(), sx = 600 / r.width;
      var x = (e.clientX - r.left) * sx, y = (e.clientY - r.top) * sx;
      var dx = B.x - A.x, dy = B.y - A.y;
      var a = clamp(((x - A.x) * dx + (y - A.y) * dy) / (dx * dx + dy * dy));
      slider.value = a.toFixed(2); update();
    }
    svg.addEventListener("pointerdown", function (e) { drag = true; svg.setPointerCapture(e.pointerId); fromEvent(e); });
    svg.addEventListener("pointermove", function (e) { if (drag) fromEvent(e); });
    svg.addEventListener("pointerup", function () { drag = false; });
    svg.style.touchAction = "pan-y";
    update();
  })();

  /* ---------- tooltips ---------- */
  var tip = $("tip");
  function showTip(t, x, y) {
    tip.textContent = t.getAttribute("data-tip"); tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(Math.max(8, x - w / 2), window.innerWidth - w - 8) + "px";
    var top = y - h - 12; tip.style.top = (top < 8 ? y + 18 : top) + "px";
  }
  document.addEventListener("pointermove", function (e) {
    var t = e.target.closest && e.target.closest("[data-tip]");
    if (t) showTip(t, e.clientX, e.clientY); else tip.hidden = true;
  });
  document.addEventListener("scroll", function () { tip.hidden = true; }, { passive: true });

  /* ---------- copy bibtex ---------- */
  var copy = $("copyBib");
  if (copy) copy.addEventListener("click", function () {
    var text = $("bibtex").textContent;
    var done = function () { copy.textContent = "Copied"; setTimeout(function () { copy.textContent = "Copy"; }, 1600); };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
    else { var r = document.createRange(); r.selectNodeContents($("bibtex")); var s = getSelection(); s.removeAllRanges(); s.addRange(r); }
  });

  /* ---------- visibility helpers ---------- */
  function onVisible(node, fn, thr) {
    if (!node) return;
    if (!("IntersectionObserver" in window)) { fn(); return; }
    var io = new IntersectionObserver(function (en) {
      en.forEach(function (x) { if (x.isIntersecting) { io.disconnect(); fn(); } });
    }, { threshold: thr || 0.2 });
    io.observe(node);
  }
  if (!reduce && "IntersectionObserver" in window) {
    var items = document.querySelectorAll("main section > figure, main section > .aside");
    var io = new IntersectionObserver(function (en) {
      en.forEach(function (x) { if (x.isIntersecting) { x.target.classList.add("in"); io.unobserve(x.target); } });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    items.forEach(function (n) {
      var r = n.getBoundingClientRect();
      if (r.top < window.innerHeight) return;
      n.classList.add("reveal"); io.observe(n);
    });
  }
})();
