# Classification report

Test set: `test_set.csv` (70 examples)

| System | Accuracy | Macro-F1 | ms / item (incl. model load) |
|---|---|---|---|
| regex | 0.829 | 0.840 | 0.2 |
| setfit | 0.871 | 0.868 | 200.1 |

## regex

```
                        precision    recall  f1-score   support

     deceptive_urgency      1.000     1.000     1.000        10
unsubstantiated_claims      1.000     0.900     0.947        10
     subscription_trap      0.667     0.800     0.727        10
       confirm_shaming      1.000     0.800     0.889        10
          drip_pricing      1.000     0.700     0.824        10
       bait_and_switch      1.000     0.700     0.824        10
                  safe      0.529     0.900     0.667        10

              accuracy                          0.829        70
             macro avg      0.885     0.829     0.840        70
          weighted avg      0.885     0.829     0.840        70
```

## setfit

```
                        precision    recall  f1-score   support

     deceptive_urgency      0.909     1.000     0.952        10
unsubstantiated_claims      1.000     1.000     1.000        10
     subscription_trap      0.875     0.700     0.778        10
       confirm_shaming      1.000     1.000     1.000        10
          drip_pricing      0.769     1.000     0.870        10
       bait_and_switch      0.778     0.700     0.737        10
                  safe      0.778     0.700     0.737        10

              accuracy                          0.871        70
             macro avg      0.873     0.871     0.868        70
          weighted avg      0.873     0.871     0.868        70
```
