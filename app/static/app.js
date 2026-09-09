const form = document.getElementById("riskForm");
const result = document.getElementById("result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  const data = {};
  for (const [k, v] of fd.entries()) data[k] = Number(v);

  result.classList.remove("hidden");
  document.getElementById("risk").textContent = "Analizando...";
  document.getElementById("probability").textContent = "";
  document.getElementById("message").textContent = "";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    });
    const out = await response.json();
    if (!response.ok || out.error) throw new Error(out.error || "Error de API");

    document.getElementById("risk").textContent = out.risk;
    document.getElementById("probability").textContent =
      `${Math.round(out.probability * 100)}%`;
    document.getElementById("message").textContent = out.message;

    const bars = document.getElementById("bars");
    bars.innerHTML = "";
    for (const [name, prob] of Object.entries(out.probabilities)) {
      bars.innerHTML += `
        <div class="bar">
          <div class="bar-top"><span>${name}</span><span>${Math.round(prob*100)}%</span></div>
          <div class="track"><div class="fill" style="width:${prob*100}%"></div></div>
        </div>`;
    }
  } catch (err) {
    document.getElementById("risk").textContent = "Error";
    document.getElementById("message").textContent = err.message;
  }
});
