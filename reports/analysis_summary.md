# Ecommerce UX Funnel Audit - Analysis Summary

## Dataset Snapshot
- Data source: **Kaggle eCommerce events history in electronics store**
- Sessions analyzed: **490,398**
- Users analyzed: **407,073**
- Session purchase rate: **4.96%**
- Observed purchase revenue: **$5,125,395.62**

## Funnel Diagnostic
- Largest step drop-off: **Add to cart** (91.55% drop-off from prior step).
- Full funnel table: `outputs/overall_funnel.csv`
- Funnel chart: `outputs/01_overall_funnel.png`

## Statistical Testing
- Completed significant tests at alpha=0.05: **10**
- Test details: `outputs/statistical_tests.csv`

## Predictive Modeling
- Purchase propensity model status: **completed**
- Model metrics: `outputs/model_performance.csv`
- Feature importance: `outputs/model_feature_importance.csv`
- High-intent no-purchase segments: `outputs/high_intent_no_purchase_segments.csv`

## UX Segmentation
- Segment table: `outputs/ux_behavior_segments.csv`
- High-intent no-purchase segments: `outputs/high_intent_no_purchase_segments.csv`

## Event Scope
The dataset tracks `view`, `cart`, and `purchase` events. Cart behavior is treated as the measurable purchase-intent stage, so later-stage findings should be read as cart-to-purchase friction.
