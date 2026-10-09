// ---------------------------------------------------------------
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
// 3b. 5-YEAR AGE BRACKETS
// Two people land in the same bracket (and therefore get the same
// age-driven scoring) only if they're within the same 5-year band —
// e.g. 30 and 34 share a bracket, but 30 and 36 do not.
// ---------------------------------------------------------------
function ageBracket(age){
  return Math.floor((age - 10) / 5) * 5 + 10; // 10-14, 15-19, 20-24, ...
}
function ageBracketLabel(age){
  const lo = ageBracket(age);
  return `Ages ${lo}\u2013${lo + 4}`;
}

// Metabolic energy-need multiplier per 5-year bracket — a simplified,
// non-clinical curve: needs rise through the 20s-30s, then taper
// gradually from midlife onward.
function ageCalorieMultiplier(age){
  const b = ageBracket(age);
  if (b < 20) return 0.95;
  if (b < 45) return 1.08;
  if (b < 60) return 1.00;
  if (b < 75) return 0.92;
  return 0.85;
}

// ---------------------------------------------------------------
// 3c. MEAL "HEAVINESS" TARGETS
// Breakfast is the heaviest meal of the day, dinner the lightest,
// lunch in between.
// ---------------------------------------------------------------
const MEAL_BASE_CAL = { breakfast: 320, lunch: 300, dinner: 190 };
const mealWeightLabel = { breakfast: "hearty", lunch: "balanced", dinner: "light" };

function calorieTarget(profile){
  const genderMult = profile.gender === "male" ? 1.12
                    : profile.gender === "female" ? 0.90
                    : 1.0;
  return MEAL_BASE_CAL[profile.meal] * genderMult * ageCalorieMultiplier(profile.age);
}

// ---------------------------------------------------------------
// 4. RULE-BASED SCORING FUNCTION
// Used to create the training labels for the ML models below.
// ---------------------------------------------------------------
function scoreFood(food, profile){
  let score = 0;

  // a) Diabetes-type fit (0-10 -> 0-40)
  score += food.diabetesFit[profile.dtype] * 4;

  // b) Glycemic index — lower is better (up to 20)
  score += (100 - food.gi) * 0.2;

  // c) Fibre — slows glucose absorption
  score += food.n.fiber * 1.5;

  // d) Life-stage match (up to 8)
  score += food.ageBest.includes(ageGroup(profile.age)) ? 8 : 2;

  // e) Gender-based micronutrient relevance (up to ~8)
  if (profile.gender === "female"){
    score += Math.min(food.n.iron * 1.2, 6) + Math.min(food.n.calcium / 40, 3);
  } else if (profile.gender === "male"){
    score += Math.min(food.n.zinc * 2.2, 6) + Math.min(food.n.magnesium / 20, 3);
  } else {
    score += Math.min(food.n.zinc * 1.1, 3) + Math.min(food.n.iron * 0.6, 3);
  }

  // f) Meal-heaviness fit (up to 14)
  const target = calorieTarget(profile);
  score += Math.max(0, 14 - Math.abs(food.n.cal - target) / 12);

  // g) Mild penalties for sugar and sodium
  score -= food.n.sugar * 1.1;
  score -= (food.n.sodium / 100) * 0.6;

  return score;
}

// =================================================================
// 5. MACHINE LEARNING PIPELINE
//    dataset -> preprocessing -> feature extraction (PCA) ->
//    proposed algorithm (KNN / SVM / Logistic Regression) -> output
// =================================================================

// --- 5a. Feature construction: one numeric row per dish ---
function engineeredFeatures(food, profile){
  const mealFit = -Math.abs(food.n.cal - calorieTarget(profile)) / 10;
  return [ food.n.cal, food.n.protein, food.n.fat, food.n.carbs,
           food.n.fiber, food.n.sugar, food.n.sodium, food.gi,
           food.diabetesFit[profile.dtype], mealFit ];
}
function idealFeatures(profile){
  // A hypothetical "perfect" dish for this profile
  return [ calorieTarget(profile), 20, 8, 28, 9, 2, 250, 25, 10, 0 ];
}

// --- 5b. Preprocessing: z-score standardization ---
function standardize(X){
  const d = X[0].length, means=[], stds=[];
  for (let j=0; j<d; j++){
    const col = X.map(r=>r[j]);
    const m = col.reduce((a,b)=>a+b,0)/col.length;
    const v = col.reduce((a,b)=>a+(b-m)*(b-m),0)/col.length;
    means.push(m); stds.push(Math.sqrt(v) || 1e-6);
  }
  return { Z: X.map(r=>r.map((v,j)=>(v-means[j])/stds[j])), means, stds };
}

// --- 5c. Feature extraction: PCA via Jacobi eigen-decomposition ---
function covariance(Z){
  const n=Z.length, d=Z[0].length;
  const C = Array.from({length:d}, ()=>Array(d).fill(0));
  for (let i=0;i<d;i++) for (let j=0;j<d;j++){
    let s=0; for (let k=0;k<n;k++) s += Z[k][i]*Z[k][j];
    C[i][j] = s/(n-1);
  }
  return C;
}
function jacobiEigen(Ain, sweeps=60){
  const n = Ain.length;
  const A = Ain.map(r=>r.slice());
  const V = Array.from({length:n}, (_,i)=>Array.from({length:n}, (_,j)=> i===j?1:0));
  for (let s=0; s<sweeps; s++){
    let off=0;
    for (let i=0;i<n;i++) for (let j=i+1;j<n;j++) off += A[i][j]*A[i][j];
    if (off < 1e-10) break;
    for (let p=0;p<n;p++) for (let q=p+1;q<n;q++){
      if (Math.abs(A[p][q]) < 1e-12) continue;
      const theta = (A[q][q]-A[p][p])/(2*A[p][q]);
      const t = (theta>=0?1:-1)/(Math.abs(theta)+Math.sqrt(theta*theta+1));
      const c = 1/Math.sqrt(t*t+1), sn = t*c;
      for (let k=0;k<n;k++){ const akp=A[k][p], akq=A[k][q]; A[k][p]=c*akp-sn*akq; A[k][q]=sn*akp+c*akq; }
      for (let k=0;k<n;k++){ const apk=A[p][k], aqk=A[q][k]; A[p][k]=c*apk-sn*aqk; A[q][k]=sn*apk+c*aqk; }
      for (let k=0;k<n;k++){ const vkp=V[k][p], vkq=V[k][q]; V[k][p]=c*vkp-sn*vkq; V[k][q]=sn*vkp+c*vkq; }
    }
  }
  const eigenvalues = A.map((row,i)=>row[i]);
  const eigenvectors = [];
  for (let j=0;j<n;j++) eigenvectors.push(V.map(row=>row[j]));
  return { eigenvalues, eigenvectors };
}
function pca(X, k){
  const { Z, means, stds } = standardize(X);
  const { eigenvalues, eigenvectors } = jacobiEigen(covariance(Z));
  const order = eigenvalues.map((v,i)=>i).sort((a,b)=>eigenvalues[b]-eigenvalues[a]);
  const total = eigenvalues.reduce((a,b)=>a+b,0) || 1;
  const top = order.slice(0,k);
  const explainedVar = top.reduce((a,i)=>a+eigenvalues[i],0)/total;
  const components = top.map(i=>eigenvectors[i]);
  const project = (zRow)=> components.map(c=>zRow.reduce((s,v,i)=>s+v*c[i],0));
  return { Z: Z.map(project), means, stds, project, explainedVar };
}

// --- 5d. The three algorithms ---
function trainLogReg(Z, y, lr=0.2, iters=400){
  const n=Z.length, d=Z[0].length;
  let w=Array(d).fill(0), b=0;
  for (let it=0; it<iters; it++){
    const gw=Array(d).fill(0); let gb=0;
    for (let i=0;i<n;i++){
      const z = Z[i].reduce((s,v,j)=>s+v*w[j],0)+b;
      const p = 1/(1+Math.exp(-z));
      const err = p - y[i];
      for (let j=0;j<d;j++) gw[j]+=err*Z[i][j];
      gb += err;
    }
    for (let j=0;j<d;j++) w[j]-=lr*gw[j]/n;
    b -= lr*gb/n;
  }
  return { score:(z)=>1/(1+Math.exp(-(z.reduce((s,v,j)=>s+v*w[j],0)+b))) };
}
function trainSVM(Z, y01, lr=0.02, iters=400, lambda=0.02){
  const y = y01.map(v=>v===1?1:-1);
  const n=Z.length, d=Z[0].length;
  let w=Array(d).fill(0), b=0;
  for (let it=0; it<iters; it++){
    for (let i=0;i<n;i++){
      const f = Z[i].reduce((s,v,j)=>s+v*w[j],0)+b;
      if (y[i]*f < 1){
        for (let j=0;j<d;j++) w[j] += lr*(y[i]*Z[i][j] - 2*lambda*w[j]);
        b += lr*y[i];
      } else {
        for (let j=0;j<d;j++) w[j] += lr*(-2*lambda*w[j]);
      }
    }
  }
  return { score:(z)=>z.reduce((s,v,j)=>s+v*w[j],0)+b };
}
function knnScore(Z, idealZ, k){
  const dists = Z.map((z,i)=>({ i, d:Math.sqrt(z.reduce((s,v,j)=>s+(v-idealZ[j])**2,0)) }));
  dists.sort((a,b)=>a.d-b.d);
  const maxD = dists[dists.length-1].d || 1;
  const scores = new Array(Z.length);
  dists.forEach(({i,d}) => { scores[i] = 1 - d/maxD; });
  return { scores, neighbors: dists.slice(0,k) };
}

// --- 5e. Full pipeline for one profile + chosen algorithm ---
function runPipeline(profile, algorithm){
  const X = FOODS.map(f => engineeredFeatures(f, profile));
  const ruleScores = FOODS.map(f => scoreFood(f, profile));
  const med = [...ruleScores].sort((a,b)=>a-b)[Math.floor(ruleScores.length/2)];
  const labels = ruleScores.map(s => s >= med ? 1 : 0);

  const k = 3;
  const { Z, means, stds, project, explainedVar } = pca(X, k);
  const idealZ = project(idealFeatures(profile).map((v,j)=>(v-means[j])/stds[j]));

  let rawScores, algoInfo;
  if (algorithm === "knn"){
    const kn = 7;
    const { scores, neighbors } = knnScore(Z, idealZ, kn);
    rawScores = scores;
    const positive = neighbors.filter(nb => labels[nb.i] === 1).length;
    algoInfo = { name:"K-Nearest Neighbours", detail:`Ranked by distance (in PCA space) to an ideal target dish for this profile — ${positive} of the ${kn} nearest dishes were also rule-flagged as strong fits.` };
  } else if (algorithm === "svm"){
    const model = trainSVM(Z, labels);
    rawScores = Z.map(z => model.score(z));
    const acc = rawScores.filter((s,i)=> (s>0?1:0)===labels[i]).length / rawScores.length;
    algoInfo = { name:"Linear SVM", detail:`Max-margin hyperplane trained over the ${k} principal components — ${Math.round(acc*100)}% agreement with the training labels.` };
  } else {
    const model = trainLogReg(Z, labels);
    rawScores = Z.map(z => model.score(z));
    const acc = rawScores.filter((s,i)=> (s>=0.5?1:0)===labels[i]).length / rawScores.length;
    algoInfo = { name:"Logistic Regression", detail:`Probabilistic classifier trained over the ${k} principal components — ${Math.round(acc*100)}% agreement with the training labels.` };
  }

  const ranked = FOODS.map((food,i) => ({ food, raw: rawScores[i] }));
  const pool = ranked.filter(r => r.food.diet === profile.diet && r.food.meal.includes(profile.meal));
  pool.sort((a,b) => b.raw - a.raw);
  const top = pool.slice(0, 3);

  const maxRaw = top.length ? top[0].raw : 1;
  const minRaw = pool.length ? pool[pool.length-1].raw : 0;
  const range = (maxRaw - minRaw) || 1;
  top.forEach(t => {
    t.match = Math.max(55, Math.min(99, Math.round(60 + ((t.raw - minRaw)/range) * 39)));
  });

  return {
    top,
    pipeline: {
      datasetSize: FOODS.length,
      featureCount: X[0].length,
      componentCount: k,
      explainedVar,
      algorithm: algoInfo,
      outputCount: top.length,
    }
  };
}

// ---------------------------------------------------------------
// 6. RENDER
// ---------------------------------------------------------------
const dtypeLabels = { type1:"Type 1 diabetes", type2:"Type 2 diabetes", gestational:"gestational diabetes", prediabetes:"prediabetes" };
const mealLabels = { breakfast:"breakfast", lunch:"lunch", dinner:"dinner" };

function renderResults(profile, results, pipeline){
  const grid = document.getElementById("cardsGrid");
  grid.innerHTML = "";

  document.getElementById("resultsHeading").textContent =
    `Your top ${mealLabels[profile.meal]} picks`;
  document.getElementById("resultsCount").textContent =
    `${results.length} of ${FOODS.length} dishes matched · ${ageBracketLabel(profile.age)} · ranked for ${dtypeLabels[profile.dtype]}`;

  const targetLine = document.getElementById("targetLine");
  if (targetLine){
    targetLine.textContent =
      `Targeting a ${mealWeightLabel[profile.meal]} ~${Math.round(calorieTarget(profile))} kcal ${mealLabels[profile.meal]} for this profile — every dish shown is still built around diabetes-friendly, low-GI choices.`;
  }

  const pipeEl = document.getElementById("pipelineStrip");
  if (pipeEl && pipeline){
    pipeEl.innerHTML = `
      <div class="pipe-step"><span class="pipe-num">1</span><div><b>Dataset</b><br>${pipeline.datasetSize} dishes × ${pipeline.featureCount} features</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step"><span class="pipe-num">2</span><div><b>Preprocessing</b><br>Z-score standardization</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step"><span class="pipe-num">3</span><div><b>Feature extraction</b><br>PCA → ${pipeline.componentCount} components (${Math.round(pipeline.explainedVar*100)}% variance kept)</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step"><span class="pipe-num">4</span><div><b>${pipeline.algorithm.name}</b><br>${pipeline.algorithm.detail}</div></div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step"><span class="pipe-num">5</span><div><b>Output</b><br>Top ${pipeline.outputCount} ${mealLabels[profile.meal]} matches</div></div>
    `;
  }

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
  const algorithm = document.getElementById("algorithm").value;

  const { top, pipeline } = runPipeline(profile, algorithm);
  const resultsSection = document.getElementById("results");
  resultsSection.hidden = false;
  renderResults(profile, top, pipeline);
  resultsSection.scrollIntoView({ behavior:"smooth", block:"start" });
});

// Run once on load with the default values so the page never feels empty
window.addEventListener("DOMContentLoaded", () => {
  form.requestSubmit();
});
