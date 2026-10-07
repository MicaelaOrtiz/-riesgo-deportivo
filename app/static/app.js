const form = document.getElementById("riskForm");
const result = document.getElementById("result");

const RISK_COLORS = {
  Bajo: "var(--risk-bajo)",
  Medio: "var(--risk-medio)",
  Alto: "var(--risk-alto)",
};

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  const data = {};
  for (const [k, v] of fd.entries()) data[k] = Number(v);

  result.classList.remove("hidden");
  result.style.setProperty("--accent", "var(--risk-medio)");
  document.getElementById("risk").textContent = "Analizando…";
  document.getElementById("probability").textContent = "";
  document.getElementById("message").textContent = "";
  document.getElementById("bars").innerHTML = "";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const out = await response.json();
    if (!response.ok || out.error) throw new Error(out.error || "Error de API");

    const accent = RISK_COLORS[out.risk] || "var(--risk-medio)";
    result.style.setProperty("--accent", accent);

    document.getElementById("risk").textContent = out.risk;
    document.getElementById("probability").textContent =
      `${Math.round(out.probability * 100)}% de confianza`;
    document.getElementById("message").textContent = out.message;

    const bars = document.getElementById("bars");
    for (const [name, prob] of Object.entries(out.probabilities)) {
      const color = RISK_COLORS[name] || "var(--risk-medio)";
      bars.innerHTML += `
        <div>
          <div class="bar-top"><span>${name}</span><span>${Math.round(prob * 100)}%</span></div>
          <div class="track"><div class="fill" style="width:${prob * 100}%;background:${color}"></div></div>
        </div>`;
    }
  } catch (err) {
    result.style.setProperty("--accent", "var(--risk-alto)");
    document.getElementById("risk").textContent = "Error";
    document.getElementById("message").textContent = err.message;
  }
});
