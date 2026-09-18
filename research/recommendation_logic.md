# Recommendation Logic

## Overview

DiaPlate uses a rule-based recommendation system implemented in JavaScript.

The current system does not use machine learning, deep learning, or model training.

## Main Stages

### 1. User Input

The system collects user information such as:

- Age
- Diabetes type
- Meal preference
- Dietary preference

### 2. Age Classification

The prototype groups users according to age-based conditional rules.

Example:

- Young Adult
- Adult
- Senior

The exact age boundaries should match the conditions implemented in the JavaScript code.

### 3. Food Filtering

Food items are filtered according to the user's selected preferences.

Possible filtering factors include:

- Meal category
- Dietary preference
- Food compatibility

### 4. Weighted Scoring

The system assigns scores to food items using selected nutritional and user-related factors.

Possible factors include:

- Diabetes-type compatibility
- Glycemic Index
- Fibre
- Age compatibility
- Micronutrient information
- Sugar penalty
- Sodium penalty

A simplified representation is:

Total Score =
Diabetes Fit Score
+ GI Score
+ Fibre Score
+ Age Compatibility Score
+ Micronutrient Score
- Sugar Penalty
- Sodium Penalty

## Scoring Limitations

The scoring weights are prototype design decisions unless they have been validated through formal research.

The score should not be interpreted as:
- A medical risk score
- A clinical recommendation
- A guarantee of blood glucose response
- A substitute for professional dietary advice

## 5. Sorting

After scoring, the food items are sorted in descending order.

The highest-scoring food items appear first.

Example JavaScript:

```javascript
scored.sort((a, b) => b.score - a.score);
