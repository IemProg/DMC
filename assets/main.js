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

  /* ---------- hero animation ---------- */
  (function hero() {
    var svg = $("heroSvg"); if (!svg) return;
    var G = { x: 92, y: 190 }, S = { x: 508, y: 88 };
    var M = { x: 300, y: 139 };
    var Gp = { x: 150, y: 176 }, Sp = { x: 450, y: 102 };
    var ux = S.x - G.x, uy = S.y - G.y, ul = Math.hypot(ux, uy); ux /= ul; uy /= ul;
    var px = -uy, py = ux;
    var one = $("hOne"), oneL = $("hOneL"), gen = $("hGen"), spec = $("hSpec"), genL = $("hGenL"), specL = $("hSpecL");
    var arS = $("hArS"), arG = $("hArG"), txS = $("hTxS"), txG = $("hTxG"), arrows = $("hArrows");
    var corr = $("hCorr"), band = $("hBand"), line = $("hLine"), star = $("hStar");
    var steps = document.querySelectorAll(".steps li");
    var T1 = 2600, T2 = 4100, T3 = 5500;
    var set = function (e, a) { for (var k in a) e.setAttribute(k, a[k]); };
    var t0 = null, raf = null;

    function step(i) { for (var k = 0; k < steps.length; k++) steps[k].classList.toggle("on", k === i); }

    function frame(t) {
      // phase A: one prompt, jittering at the compromise
      var jig = t < T1 + 400 ? 1 : 0;
      var w = Math.sin(t * 0.0105) * 0.6 + Math.sin(t * 0.0231) * 0.4;
      var P = { x: M.x + ux * 4 * w + px * 1.6 * Math.sin(t * 0.017), y: M.y + uy * 4 * w + py * 1.6 * Math.sin(t * 0.017) };
      var split = ease((t - T1) / (T2 - T1));
      var fadeA = 1 - clamp((t - T1) / 450);
      var LS = 104 + 9 * w, LG = 104 - 9 * w;
      set(arS, { x1: P.x + ux * 13, y1: P.y + uy * 13, x2: P.x + ux * LS, y2: P.y + uy * LS });
      set(arG, { x1: P.x - ux * 13, y1: P.y - uy * 13, x2: P.x - ux * LG, y2: P.y - uy * LG });
      set(txS, { x: M.x + ux * 70 - px * 22, y: M.y + uy * 70 - py * 22 });
      set(txG, { x: M.x - ux * 70 + px * 22, y: M.y - uy * 70 + py * 22 + 12 });
      arrows.setAttribute("opacity", fadeA);
      // points
      var g = { x: lerp(P.x, Gp.x, split), y: lerp(P.y, Gp.y, split) };
      var s = { x: lerp(P.x, Sp.x, split), y: lerp(P.y, Sp.y, split) };
      var splitting = t >= T1;
      set(one, { cx: P.x, cy: P.y, opacity: splitting ? 0 : 1 });
      set(oneL, { x: P.x, y: P.y - 20, opacity: splitting ? 0 : 1 });
      set(gen, { cx: g.x, cy: g.y, opacity: splitting ? 1 : 0 });
      set(spec, { cx: s.x, cy: s.y, opacity: splitting ? 1 : 0 });
      var lab = clamp((t - T2 + 500) / 500);
      set(genL, { x: g.x, y: g.y - 20, opacity: lab });
      set(specL, { x: s.x, y: s.y + 34, opacity: lab });
      set(band, { x1: g.x, y1: g.y, x2: s.x, y2: s.y });
      set(line, { x1: g.x, y1: g.y, x2: s.x, y2: s.y });
      corr.setAttribute("opacity", clamp((t - T1 - 250) / 900));
      // star slides from c_gen to alpha = 0.2
      var st = ease((t - T2 - 150) / (T3 - T2 - 150));
      var a = 0.2 * st, sx = lerp(g.x, s.x, a), sy = lerp(g.y, s.y, a);
      star.setAttribute("transform", "translate(" + sx + "," + sy + ")");
      star.setAttribute("opacity", clamp((t - T2 - 150) / 300));
      step(t < T1 ? 0 : t < T2 ? 1 : 2);
      return jig;
    }
    function loop(now) {
      if (t0 === null) t0 = now;
      var t = now - t0;
      frame(t);
      if (t < T3 + 50) raf = requestAnimationFrame(loop); else raf = null;
    }
    function play() {
      if (reduce) { frame(T3 + 100); for (var k = 0; k < steps.length; k++) steps[k].classList.add("on"); return; }
      if (raf) cancelAnimationFrame(raf);
      t0 = null; raf = requestAnimationFrame(loop);
    }
    frame(0);
    if (reduce) { play(); } else { setTimeout(play, 350); }
    $("replay").addEventListener("click", play);
    svg.addEventListener("click", play);
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
