const form = document.getElementById("riskForm");
const submitBtn = document.getElementById("submitBtn");
const result = document.getElementById("result");
const riskEl = document.getElementById("risk");
const confEl = document.getElementById("confidence");
const alertBox = document.getElementById("alert");
const messageEl = document.getElementById("message");

const pct = (p) => Math.round(p * 100);

function showBars(probabilities) {
  document.querySelectorAll(".bar-row").forEach((row) => {
    const level = row.dataset.level;
    const value = probabilities ? pct(probabilities[level] ?? 0) : 0;
    row.querySelector("b").textContent = `${value}%`;
    row.querySelector(".fill").style.width = `${value}%`;
  });
}

function showError(text) {
  result.dataset.level = "none";
  riskEl.textContent = "Error";
  confEl.textContent = "No se pudo calcular la predicción.";
  showBars(null);
  messageEl.textContent = text;
  alertBox.hidden = false;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const data = {};
  for (const [key, value] of new FormData(form).entries()) data[key] = Number(value);

  submitBtn.disabled = true;
  submitBtn.textContent = "Calculando...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const out = await response.json();
    if (!response.ok || out.error) {
      throw new Error(out.error || "La API respondió con un error. ¿Entrenaste el modelo con python train.py?");
    }

    result.dataset.level = out.risk;
    riskEl.textContent = out.risk.toUpperCase();
    confEl.textContent = `${pct(out.probability)}% de confianza del modelo`;
    showBars(out.probabilities);
    messageEl.textContent = out.message;
    alertBox.hidden = false;
  } catch (err) {
    showError(err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predecir riesgo";
  }
});
