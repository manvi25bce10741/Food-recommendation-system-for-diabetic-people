// 6. RENDER
// ---------------------------------------------------------------
const dtypeLabels = { type1:"Type 1 diabetes", type2:"Type 2 diabetes", gestational:"gestational diabetes", prediabetes:"prediabetes" };
const mealLabels = { breakfast:"breakfast", lunch:"lunch", dinner:"dinner" };

function renderResults(profile, results){
  const grid = document.getElementById("cardsGrid");
  grid.innerHTML = "";

  document.getElementById("resultsHeading").textContent =
    `Your top ${mealLabels[profile.meal]} picks`;
  document.getElementById("resultsCount").textContent =
    `${results.length} of ${FOODS.length} dishes matched · ranked for ${dtypeLabels[profile.dtype]}`;

  if (results.length === 0){
    grid.innerHTML = `<p class="empty-note">No dishes matched that exact combination yet — try switching the meal or dietary preference.</p>`;
  }

  results.forEach(({food, match}) => {
    const gi = giLabel(food.gi);
    const isVeg = food.diet === "veg";

    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `
      <div class="card-top">
        <div class="card-top-row">
          <div>
            <div class="card-title">${food.name}</div>
            <div class="card-serving">${food.serving}</div>
          </div>
          <div class="diet-mark ${isVeg ? "" : "nonveg"}" title="${isVeg ? "Vegetarian" : "Non-vegetarian"}"><span class="dot"></span></div>
        </div>
        <div class="match-row">
          <div class="match-label"><span>Match for you</span><span>${match}%</span></div>
          <div class="match-bar"><div class="match-fill" style="width:${match}%"></div></div>
        </div>
      </div>

      <div class="nutri-head">Nutrition per serving</div>
      <div class="nutri-body">
        <div class="nrow total"><span>Calories</span><span class="val">${food.n.cal} kcal</span></div>
        <div class="nrow"><span>Protein</span><span class="val">${food.n.protein} g</span></div>
        <div class="nrow"><span>Total fat</span><span class="val">${food.n.fat} g</span></div>
        <div class="nrow"><span>Carbohydrates</span><span class="val">${food.n.carbs} g</span></div>
        <div class="nrow sub"><span>— Fibre</span><span class="val">${food.n.fiber} g</span></div>
        <div class="nrow sub"><span>— Sugar</span><span class="val">${food.n.sugar} g</span></div>
        <div class="nrow"><span>Zinc</span><span class="val">${food.n.zinc} mg</span></div>
        <div class="nrow"><span>Magnesium</span><span class="val">${food.n.magnesium} mg</span></div>
        <div class="nrow"><span>Calcium</span><span class="val">${food.n.calcium} mg</span></div>
        <div class="nrow"><span>Iron</span><span class="val">${food.n.iron} mg</span></div>
        <div class="nrow"><span>Sodium</span><span class="val">${food.n.sodium} mg</span></div>
      </div>
      <span class="gi-pill">${gi.text} · GI ${food.gi}</span>
      <span class="meal-tag">${food.meal.map(m => mealLabels[m]).join(" / ")}</span>
      <p class="why"><b>Why it fits:</b> ${food.benefit}</p>
    `;
    grid.appendChild(card);
  });
}

// ---------------------------------------------------------------
// 7. FORM HANDLING
// ---------------------------------------------------------------
const form = document.getElementById("recForm");
const formMsg = document.getElementById("formMsg");

form.addEventListener("submit", function(e){
  e.preventDefault();

  const age = parseInt(document.getElementById("age").value, 10);
  if (isNaN(age) || age < 10 || age > 100){
    formMsg.classList.add("show");
    return;
  }
  formMsg.classList.remove("show");

  const profile = {
    age,
    gender: document.getElementById("gender").value,
    diet: document.getElementById("diet").value,
    meal: document.getElementById("meal").value,
    dtype: document.getElementById("dtype").value,
  };

  const results = recommend(profile);
  const resultsSection = document.getElementById("results");
  resultsSection.hidden = false;
  renderResults(profile, results);
  resultsSection.scrollIntoView({ behavior:"smooth", block:"start" });
});

// Run once on load with the default values so the page never feels empty
window.addEventListener("DOMContentLoaded", () => {
  form.requestSubmit();
});
