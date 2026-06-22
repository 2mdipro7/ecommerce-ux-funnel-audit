# Analytical Findings & Insights: Ecommerce UX Funnel Audit

Prepared June 2026

This document summarizes the final executed analysis of real ecommerce clickstream behavior. The project diagnoses where users lose momentum across product discovery, cart intent, and purchase behavior, then identifies category and product segments with the strongest UX intervention potential.

---

## 1. Dataset & Business Context

This project analyzes a real ecommerce clickstream dataset from an electronics store:

```text
Kaggle: eCommerce events history in electronics store
```

The cleaned dataset contains:

- **884,964 events**
- **490,398 sessions**
- **407,073 users**
- **793,589 product view events**
- **54,029 cart events**
- **37,346 purchase events**
- **$5,125,395.62 observed purchase revenue**

Event taxonomy: the dataset has `view`, `cart`, and `purchase` events. Cart is treated as the measurable purchase-intent stage, so the later-stage analysis focuses on **cart-to-purchase friction**.

---

## 2. Executive Summary

The data shows that the primary ecommerce UX bottleneck is not product visibility or final checkout completion. The largest friction point is the moment between product detail exposure and cart intent.

**Key Takeaways:**

1. **Product Persuasion Is the Primary Bottleneck:** Product detail behavior appears in **99.58%** of sessions, but only **8.45%** of product-detail sessions add to cart.
2. **The Largest Drop-Off Happens Before Cart:** The product-detail-to-cart stage loses **91.55%** of sessions, making it the clearest UX intervention point.
3. **Cart Intent Is Strong Commercial Signal:** Once users reach cart intent, **58.99%** purchase.
4. **Category Context Matters:** Product/category entry point is statistically associated with both purchase conversion and product-detail-to-cart behavior.
5. **High-Intent Non-Buyers Are Actionable:** The train/test model identifies likely buyers who fail to purchase, creating a practical queue for UX review and remarketing.

---

## 3. Funnel Findings

| Funnel Step | Sessions | Share of Sessions | Step Conversion | Drop-Off |
|---|---:|---:|---:|---:|
| Sessions | 490,398 | 100.00% | 100.00% | 0.00% |
| Product discovery | 490,398 | 100.00% | 100.00% | 0.00% |
| Product detail | 488,360 | 99.58% | 99.58% | 0.42% |
| Add to cart | 41,270 | 8.42% | 8.45% | 91.55% |
| Checkout-intent proxy | 41,270 | 8.42% | 100.00% | 0.00% |
| Purchase | 24,344 | 4.96% | 58.99% | 41.01% |

**Interpretation:** Users are reaching products, but product pages and category experiences are not consistently converting attention into cart intent. This points toward product-page clarity, merchandising, trust signals, pricing context, and taxonomy as higher-priority interventions than checkout-only optimization.

Key visual:

```text
outputs/01_overall_funnel.png
```

---

## 4. Behavioral Segmentation

Sessions were grouped into behavioral UX segments based on the deepest journey stage reached.

| Segment | Sessions | Users | Interpretation |
|---|---:|---:|---|
| Product Detail Browsers | 445,546 | 381,217 | Users view products but never add to cart. This is the largest friction segment. |
| Cart-Intent Non-Buyers | 20,508 | 19,298 | Users add to cart but do not purchase. This is the strongest recovery segment. |
| Purchasers | 24,344 | 21,304 | Users purchase and generate **$210.54 revenue/session**. |

Segment recommendations were synthesized from the funnel, statistical testing, product, and high-intent session outputs.

Output:

```text
outputs/ux_behavior_segments.csv
```

---

## 5. Statistical Testing Results

The statistical analysis is complete for the available executed dataset. The final testing suite covers category, cart behavior, engagement, journey depth, product breadth, and item price.

### 5.1 Normality Validation

Shapiro-Wilk checks were run on sampled session metrics. Engagement time, session duration, and event count were non-normal across tested outcome groups.

Because normality assumptions were not met, the analysis uses non-parametric tests such as Mann-Whitney U, Kruskal-Wallis, and Spearman correlation.

Output:

```text
outputs/normality_checks.csv
```

### 5.2 Completed Hypothesis Tests

| Hypothesis | Test | Result | Interpretation |
|---|---|---|---|
| Purchase conversion differs by category entry point | Chi-square | p < 0.001, Cramer's V = 0.102 | Category context is significantly associated with purchase behavior. |
| Product-detail-to-cart behavior differs by category | Chi-square | p < 0.001, Cramer's V = 0.159 | Product persuasion strength varies meaningfully by category. |
| Converted sessions have different engagement time | Mann-Whitney U | p < 0.001, rank-biserial = -0.737 | Converted sessions differ materially from non-converted sessions in engagement behavior. |
| Converted sessions have different session duration | Mann-Whitney U | p < 0.001, rank-biserial = -0.737 | Session duration separates converters from non-converters. |
| Cart-intent sessions have different event volume | Mann-Whitney U | p < 0.001, rank-biserial = -0.885 | Cart-intent sessions are behaviorally deeper than non-cart sessions. |
| Converted sessions involve different average item prices | Mann-Whitney U | p < 0.001, rank-biserial = -0.065 | Price differs statistically, though the effect size is small. |
| Journey depth differs by category entry point | Kruskal-Wallis | p < 0.001, epsilon squared = 0.025 | Category explains a modest but meaningful share of journey-depth variance. |
| Engagement differs by category entry point | Kruskal-Wallis | p < 0.001, epsilon squared = 0.017 | Engagement varies by category, with small effect size. |
| Event volume is associated with deeper journey progression | Spearman | rho = 0.545, p < 0.001 | Event activity is moderately associated with deeper funnel progression. |
| Unique product breadth is associated with journey progression | Spearman | rho = 0.181, p < 0.001 | Browsing more unique products is weakly but significantly associated with deeper journeys. |

**Statistical conclusion:** The evidence supports the core UX finding: conversion behavior varies by category, and deeper event activity is associated with deeper funnel progress. The strongest behavioral divide is between sessions that remain product-detail browsers and sessions that develop cart intent.

Output:

```text
outputs/statistical_tests.csv
outputs/08_statistical_test_summary.png
```

---

## 6. Predictive Modeling Results

A class-balanced logistic regression model was trained to predict purchase conversion using session-level behavioral and categorical features.

| Metric | Value |
|---|---:|
| Test rows | 122,600 |
| Purchase rate in test set | 4.96% |
| Accuracy | 94.39% |
| Precision | 46.82% |
| Recall | 95.43% |
| F1 | 62.82% |
| ROC AUC | 98.46% |
| Average precision | 74.51% |

**Interpretation:** The model is best used diagnostically, not as a production scoring system. Its high recall makes it useful for finding high-intent non-purchasing segments, but precision is naturally constrained by the low base purchase rate.

High-intent no-purchase segments concentrated around:

- `computers.components.videocards`
- `unknown`
- `electronics.telephone`
- `computers.peripherals.printer`
- `computers.components.cpu`

Outputs:

```text
outputs/model_performance.csv
outputs/model_feature_importance.csv
outputs/high_intent_no_purchase_segments.csv
```

---

## 7. Category and Product Discovery Findings

### 7.1 Category Performance

| Category Path | Sessions | Cart Rate | Purchase Rate | Revenue / Session |
|---|---:|---:|---:|---:|
| computers.components.videocards | 43,269 | 20.83% | 10.57% | $58.96 |
| computers.components.cpu | 11,525 | 13.26% | 5.98% | $20.06 |
| computers.peripherals.printer | 23,108 | 11.15% | 7.31% | $15.94 |
| electronics.telephone | 49,767 | 9.00% | 5.68% | $2.98 |
| unknown | 146,153 | 5.62% | 3.44% | $3.46 |

**Key Discovery:** `unknown` has the most traffic but weak revenue/session, while `computers.components.videocards` has fewer sessions but far stronger commercial performance. This suggests that taxonomy cleanup and better routing toward high-value categories could materially improve discovery quality.

### 7.2 Product-Level Findings

- `product_1821813`: **12,804 views**, **538 purchases**, **$213,844.24** revenue.
- `product_4099645`: **18.81% view-to-cart** and **10.27% view-to-purchase**.
- `product_893196`: **23.10% view-to-cart** and **13.40% view-to-purchase**, making it a hidden high-converter.

Output:

```text
outputs/product_discovery_scorecard.csv
outputs/05_product_friction_matrix.png
```

---

## 8. Strategic UX Recommendations

1. **Prioritize Product Detail Persuasion:** Since the largest loss happens before cart, improve product pages with clearer specifications, trust signals, reviews, return/shipping clarity, and stronger calls to action.
2. **Fix Category Taxonomy Gaps:** The `unknown` category has high traffic but weak commercial output. Metadata cleanup should improve routing, search relevance, and browsing confidence.
3. **Promote Hidden High-Converters:** Products with strong view-to-cart and view-to-purchase rates but lower visibility should be surfaced more prominently.
4. **Build Cart-Intent Recovery:** The **20,508 cart-intent non-buying sessions** are a strong remarketing and UX recovery opportunity.
5. **Use High-Intent Model Segments for UX Review:** Prioritize high-probability non-purchasing segments for page-level inspection and usability review.

---

## 9. Advisory Notes

1. **Event Scope:** The dataset tracks `view`, `cart`, and `purchase` behavior, so the analysis focuses on product discovery, cart intent, and purchase completion.
2. **Large Samples Make Small Effects Significant:** Some tests have extremely low p-values because the sample is large. Effect sizes should be interpreted alongside p-values.
3. **Model Is Diagnostic:** The model is intended for portfolio analytics and UX prioritization, not live production targeting.

---

## 10. Conclusion

The ecommerce journey is not failing because users cannot reach products. It is failing because product exposure does not reliably become cart intent. The strongest portfolio insight is that the highest-return UX intervention sits in the product persuasion layer: product detail content, taxonomy quality, comparison clarity, confidence cues, and merchandising of high-converting products.

The statistical testing, segmentation, and train/test modeling all support the same conclusion: deeper event behavior and category context are meaningfully connected to conversion progression, and the largest actionable opportunity is moving product-detail browsers into cart intent.
