# Data Source Note

This project uses a real ecommerce clickstream dataset from an electronics store:

https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store

Rows after cleaning: 884,964

Sessions after aggregation: 490,398

Observed event types:

- view: 793,589
- cart: 54,029
- purchase: 37,346

Event taxonomy: this dataset has `view`, `cart`, and `purchase` events. The analysis treats cart behavior as the measurable purchase-intent stage and focuses later-stage findings on cart-to-purchase friction.
