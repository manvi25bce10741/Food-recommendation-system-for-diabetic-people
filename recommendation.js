// 2. AGE GROUPING
// ---------------------------------------------------------------
function ageGroup(age){
  if (age <= 30) return "young_adult";
  if (age <= 55) return "adult";
  return "senior";
}

// ---------------------------------------------------------------
// 3. GI BUCKETS (for the label shown on each card)
// ---------------------------------------------------------------
function giLabel(gi){
  if (gi <= 35) return { text:"Low GI", cls:"low" };
  if (gi <= 55) return { text:"Medium GI", cls:"med" };
  return { text:"Higher GI", cls:"high" };
}

// ---------------------------------------------------------------
// 4. SCORING FUNCTION
// Combines: diabetes-type fit, glycemic index, fibre content,
// age-group match, and gender-based micronutrient relevance.
// This is a transparent, rule-weighted scoring model — a simple
// content-based recommender rather than a black-box ML model.
// ---------------------------------------------------------------
function scoreFood(food, profile){
  let score = 0;

  // a) Diabetes-type fit — the strongest single factor (0-10 -> 0-40)
  score += food.diabetesFit[profile.dtype] * 4;

  // b) Glycemic index — lower is better (contributes up to 20)
  score += (100 - food.gi) * 0.2;

  // c) Fibre — slows glucose absorption (contributes up to ~18)
  score += food.n.fiber * 1.5;

  // d) Age-group match (contributes up to 8)
  score += food.ageBest.includes(ageGroup(profile.age)) ? 8 : 2;

  // e) Gender-based micronutrient relevance (contributes up to ~8)
  if (profile.gender === "female"){
    score += Math.min(food.n.iron * 1.2, 6) + Math.min(food.n.calcium / 40, 3);
  } else if (profile.gender === "male"){
    score += Math.min(food.n.zinc * 2.2, 6) + Math.min(food.n.magnesium / 20, 3);
  } else {
    score += Math.min(food.n.zinc * 1.1, 3) + Math.min(food.n.iron * 0.6, 3);
  }

  // f) Mild penalties for sugar and sodium
  score -= food.n.sugar * 1.1;
  score -= (food.n.sodium / 100) * 0.6;

  return score;
}

// ---------------------------------------------------------------
// 5. FILTER + RANK
// ---------------------------------------------------------------
function recommend(profile){
  const pool = FOODS.filter(f => f.diet === profile.diet && f.meal.includes(profile.meal));
  const scored = pool.map(f => ({ food:f, score:scoreFood(f, profile) }));
  scored.sort((a,b) => b.score - a.score);
  const top = scored.slice(0, 3);

  // Normalise to a friendly 0-100% "match" for display, relative
  // to the best score in this particular result set.
  const maxScore = top.length ? top[0].score : 1;
  top.forEach(t => {
    t.match = Math.max(55, Math.min(99, Math.round(60 + (t.score / maxScore) * 39)));
  });
  return top;
}

// ---------------------------------------------------------------
