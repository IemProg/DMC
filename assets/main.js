/* DMC project page: theme toggle, math, alpha widget, tooltips, nav, copy. */
(function () {
  "use strict";
  var root = document.documentElement;

  /* ---------- theme ---------- */
  var btn = document.getElementById("themeBtn");
  function isDark() {
    var t = root.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  if (btn) {
    btn.addEventListener("click", function () {
      var next = isDark() ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("dmc-theme", next); } catch (e) {}
    });
  }

  /* ---------- math ---------- */
  function renderMath() {
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
        trust: function (ctx) { return ctx.command === "\\htmlClass"; },
        strict: false,
        throwOnError: false
      });
    }
  }
  if (document.readyState === "complete") renderMath();
  else window.addEventListener("load", renderMath);

  /* ---------- paper placeholder ---------- */
  var paper = document.getElementById("paperBtn");
  if (paper) paper.addEventListener("click", function (e) { if (paper.getAttribute("aria-disabled") === "true") e.preventDefault(); });

  /* ---------- alpha widget ---------- */
  var slider = document.getElementById("alpha");
  if (slider) {
    var A = { x: 150, y: 208 }, B = { x: 650, y: 84 };
    var NS = "http://www.w3.org/2000/svg";
    var pt = document.getElementById("pt");
    var lab = document.getElementById("ptLabel");
    var ticks = document.getElementById("ticks");
    var star = document.getElementById("starMark");
    var at = function (a) { return { x: A.x + a * (B.x - A.x), y: A.y + a * (B.y - A.y) }; };

    for (var i = 1; i < 10; i++) {
      var p = at(i / 10), c = document.createElementNS(NS, "circle");
      c.setAttribute("cx", p.x); c.setAttribute("cy", p.y); c.setAttribute("r", 2.2);
      c.setAttribute("fill", "var(--surface)"); c.setAttribute("opacity", ".9");
      ticks.appendChild(c);
    }
    // five-point star at alpha = 0.20
    (function () {
      var s = at(0.2), pts = [], R = 16, r = 7;
      for (var k = 0; k < 10; k++) {
        var ang = -Math.PI / 2 + k * Math.PI / 5, rad = k % 2 ? r : R;
        pts.push((s.x + rad * Math.cos(ang)).toFixed(1) + "," + (s.y + rad * Math.sin(ang)).toFixed(1));
      }
      var poly = document.createElementNS(NS, "polygon");
      poly.setAttribute("points", pts.join(" "));
      poly.setAttribute("fill", "var(--surface)"); poly.setAttribute("stroke", "var(--corr)"); poly.setAttribute("stroke-width", "2.5");
      star.appendChild(poly);
      var t = document.createElementNS(NS, "text");
      t.setAttribute("x", s.x - 18); t.setAttribute("y", s.y + 38); t.setAttribute("text-anchor", "start");
      t.setAttribute("fill", "var(--corr)"); t.setAttribute("style", "font:700 12.5px var(--font)");
      t.textContent = "deployed: α = 0.20";
      star.appendChild(t);
    })();

    var sub = function (s) { return s; };
    var texts = [
      [0, 0, "The generalization prompt alone",
        "Only c_gen, trained solely by the cosine anchor to stay close to zero-shot CLIP's text features. This is the end of the corridor that best preserves CLIP's knowledge of unseen classes."],
      [0.01, 0.35, "Mostly generalization, with a dose of specialization",
        "The classifier stays close to zero-shot geometry but borrows some base-class sharpness from c_spec. DMC deploys α = 0.20 on every dataset: one merged text-feature matrix, the same inference cost as a single prompt."],
      [0.35, 0.65, "An even mix",
        "Halfway between the two endpoints. Because α is sampled uniformly at every training step, the corridor loss keeps this classifier accurate on base classes too, instead of letting it fall into a high-loss barrier."],
      [0.65, 0.99, "Mostly specialization",
        "Sharper on the base classes, further from zero-shot geometry. Moving this way trades novel-class accuracy for base-class accuracy."],
      [1, 1, "The specialization prompt alone",
        "Only c_spec, trained by cross-entropy on base classes (plus the Visual Anchor), free of any pull toward zero-shot features. The most base-specialized end of the corridor."]
    ];
    var elV = document.getElementById("alphaVal"), elF = document.getElementById("formula");
    var mG = document.getElementById("mixG"), mS = document.getElementById("mixS");
    var mGl = document.getElementById("mixGl"), mSl = document.getElementById("mixSl");
    var eT = document.getElementById("explainTitle"), eX = document.getElementById("explainText");

    function update() {
      var a = Math.round(parseFloat(slider.value) * 100) / 100;
      var p = at(a);
      pt.setAttribute("transform", "translate(" + p.x + "," + p.y + ")");
      lab.setAttribute("x", p.x); lab.setAttribute("y", p.y - 22);
      lab.textContent = "f̃(" + a.toFixed(2) + ")";
      elV.textContent = a.toFixed(2);
      slider.setAttribute("aria-valuetext", "alpha " + a.toFixed(2));
      var g = (1 - a), s = a;
      elF.textContent = "f̃(α) = ℓ2( " + g.toFixed(2) + "·f_gen + " + s.toFixed(2) + "·f_spec )";
      mG.style.width = (g * 100) + "%"; mS.style.width = (s * 100) + "%";
      mGl.textContent = Math.round(g * 100) + "% generalization";
      mSl.textContent = Math.round(s * 100) + "% specialization";
      var pick = texts[2];
      for (var j = 0; j < texts.length; j++) if (a >= texts[j][0] - 1e-9 && a <= texts[j][1] + 1e-9) { pick = texts[j]; }
      var title = pick[2];
      if (Math.abs(a - 0.2) < 1e-9) title = "The deployed classifier (α = 0.20)";
      eT.textContent = title;
      eX.textContent = sub(pick[3]);
    }
    slider.addEventListener("input", update);
    Array.prototype.forEach.call(document.querySelectorAll("#widget .chip[data-a]"), function (b) {
      b.addEventListener("click", function () { slider.value = b.getAttribute("data-a"); update(); slider.focus(); });
    });
    update();
  }

  /* ---------- tooltips ---------- */
  var tip = document.getElementById("tip");
  function show(el, x, y) {
    tip.textContent = el.getAttribute("data-tip");
    tip.classList.add("on");
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(Math.max(8, x + 12), window.innerWidth - w - 8);
    var top = y - h - 12; if (top < 8) top = y + 16;
    tip.style.left = left + "px"; tip.style.top = top + "px";
  }
  function hide() { tip.classList.remove("on"); }
  document.addEventListener("mousemove", function (e) {
    var el = e.target.closest ? e.target.closest("[data-tip]") : null;
    if (el) show(el, e.clientX, e.clientY); else hide();
  });
  document.addEventListener("focusin", function (e) {
    var el = e.target.closest ? e.target.closest("[data-tip]") : null;
    if (el) { var r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top); } else hide();
  });
  document.addEventListener("scroll", hide, { passive: true });
  // SVG <title> duplicates the tooltip for screen readers; hide native tooltip on hover-capable devices
  if (window.matchMedia && window.matchMedia("(hover: hover)").matches) {
    Array.prototype.forEach.call(document.querySelectorAll("svg [data-tip] > title"), function (t) {
      t.parentNode.setAttribute("aria-label", t.textContent); t.remove();
    });
  }

  /* ---------- active nav ---------- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".navlinks a"));
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          links.forEach(function (l) { l.classList.toggle("active", l.getAttribute("href") === "#" + en.target.id); });
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("section.q").forEach(function (s) { io.observe(s); });
  }

  /* ---------- copy bibtex ---------- */
  var copy = document.getElementById("copyBib");
  if (copy) copy.addEventListener("click", function () {
    var txt = document.getElementById("bibText").textContent;
    var done = function () { copy.textContent = "Copied"; setTimeout(function () { copy.textContent = "Copy"; }, 1600); };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(txt).then(done, function () {});
    else {
      var r = document.createRange(); r.selectNodeContents(document.getElementById("bibText"));
      var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
      try { document.execCommand("copy"); done(); } catch (e) {}
    }
  });
})();
