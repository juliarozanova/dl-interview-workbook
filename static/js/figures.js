// Figure interactions (replay buttons etc.). Owned by the diagrams work; see content/figures/.
// Animated figures (<svg class="fx-anim">) play once when scrolled into view and get a
// "Replay" button. With prefers-reduced-motion they stay on their static final frame.
(function () {
  "use strict";
  var mq = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  var figs = [];

  function reduced() { return !!(mq && mq.matches); }

  function play(fig) {
    fig.classList.remove("is-playing", "fx-on");
    void fig.getBoundingClientRect(); // restart the CSS animations
    fig.classList.add("fx-on", "is-playing");
  }

  function apply(f) {
    if (reduced()) {
      f.fig.classList.remove("fx-on", "is-playing");
      f.btn.hidden = true;
    } else {
      f.btn.hidden = false;
      if (!f.played) f.fig.classList.add("fx-on"); // armed: paused on the first frame
    }
  }

  function init() {
    var observer = "IntersectionObserver" in window ? new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var f = e.target.__fx;
        observer.unobserve(e.target);
        if (f && !f.played && !reduced()) { f.played = true; play(f.fig); }
      });
    }, { threshold: 0.45 }) : null;

    document.querySelectorAll("figure.fig").forEach(function (fig) {
      if (!fig.querySelector("svg.fx-anim")) return;
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "fig-replay";
      btn.innerHTML = '<span aria-hidden="true">↻</span> Replay';
      btn.setAttribute("aria-label", "Replay the animation");
      fig.querySelector(".fig-art").appendChild(btn);
      var f = { fig: fig, btn: btn, played: false };
      fig.__fx = f;
      figs.push(f);
      btn.addEventListener("click", function () { f.played = true; play(fig); });
      apply(f);
      if (observer) observer.observe(fig);
      else if (!reduced()) { f.played = true; play(fig); }
    });

    if (mq) {
      var onChange = function () { figs.forEach(apply); };
      if (mq.addEventListener) mq.addEventListener("change", onChange);
      else if (mq.addListener) mq.addListener(onChange);
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
