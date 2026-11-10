# Classification report

Test set: Princeton 'Dark Patterns at Scale' dataset (1275 examples)

| System | Accuracy | Macro-F1 | ms / item (incl. model load) |
|---|---|---|---|
| regex | 0.636 | 0.389 | 0.1 |

## regex

```
                   precision    recall  f1-score   support

deceptive_urgency      1.000     0.635     0.777      1229
subscription_trap      0.968     0.652     0.779        46

        micro avg      0.999     0.636     0.777      1275
        macro avg      0.984     0.644     0.778      1275
     weighted avg      0.999     0.636     0.777      1275
```

## Real-World Failure Analysis (Princeton Dataset)

Below are representative failure cases from real-world shopping sites where the model missed dark patterns, along with root-cause explanations:

### Example 1: Social Proof (Activity Notification)
- **Text:** `Faith in Glendale, United States purchased a Choose Your Flavors 5 Bag Sample Pack (Whole Bean) about 1 hour ago`
- **Gold label:** `deceptive_urgency`
- **Predicted:** `safe`
- **Why it fails:** Phrasing variation: Uses past tense 'purchased a' instead of 'just bought' or 'sold in the last', missing the urgency lexicon's strict keyword patterns.

### Example 2: Social Proof (Activity Notification)
- **Text:** `31 people seeing this product right now`
- **Gold label:** `deceptive_urgency`
- **Predicted:** `safe`
- **Why it fails:** Synonym gap: Uses 'seeing' instead of the expected observation verbs ('looking', 'viewing', 'watching').

### Example 3: Scarcity (High-demand Message)
- **Text:** `An item you ordered is in high demand. No worries, we have reserved your order.`
- **Gold label:** `deceptive_urgency`
- **Predicted:** `safe`
- **Why it fails:** Implicit pressure: Creates artificial urgency ('in high demand', 'reserved your order') without explicit low-stock numbers ('only 2 left').

### Example 4: Sneaking (Hidden Costs)
- **Text:** `Merchandise: $69.99iService Charge: $14.99 Total Before Tax: $84.98 Estimated Tax: $0.00 Order Total: $84.98`
- **Gold label:** `subscription_trap`
- **Predicted:** `safe`
- **Why it fails:** Malformed scraped text: Concatenation without whitespace ('$69.99iService Charge') breaks regex word boundary \b(service|handling) fees?\b.

### Example 5: Obstruction (Hard to Cancel)
- **Text:** `If you wish to cancel your subscription, you can email us at getflat@flattummyco`
- **Gold label:** `subscription_trap`
- **Predicted:** `safe`
- **Why it fails:** Polite customer-service phrasing: Phrased conditionally ('If you wish to cancel') rather than aggressive recurring terms ('until cancelled', 'auto-renews').
