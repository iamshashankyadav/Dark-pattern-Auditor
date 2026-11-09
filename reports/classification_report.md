# Classification report

Test set: `test_set.csv` (40 examples)

| System | Accuracy | Macro-F1 | ms / item (incl. model load) |
|---|---|---|---|
| regex | 0.900 | 0.902 | 0.1 |
| zero_shot | 0.700 | 0.664 | 3475.1 |
| setfit | 0.950 | 0.950 | 1072.5 |

## regex

```
                        precision    recall  f1-score   support

     deceptive_urgency      1.000     1.000     1.000        10
unsubstantiated_claims      1.000     0.900     0.947        10
     subscription_trap      0.889     0.800     0.842        10
                  safe      0.750     0.900     0.818        10

              accuracy                          0.900        40
             macro avg      0.910     0.900     0.902        40
          weighted avg      0.910     0.900     0.902        40
```

## zero_shot

```
                        precision    recall  f1-score   support

     deceptive_urgency      1.000     0.800     0.889        10
unsubstantiated_claims      0.333     0.100     0.154        10
     subscription_trap      1.000     0.900     0.947        10
                  safe      0.500     1.000     0.667        10

              accuracy                          0.700        40
             macro avg      0.708     0.700     0.664        40
          weighted avg      0.708     0.700     0.664        40
```

## setfit

```
                        precision    recall  f1-score   support

     deceptive_urgency      1.000     0.900     0.947        10
unsubstantiated_claims      1.000     1.000     1.000        10
     subscription_trap      0.909     1.000     0.952        10
                  safe      0.900     0.900     0.900        10

              accuracy                          0.950        40
             macro avg      0.952     0.950     0.950        40
          weighted avg      0.952     0.950     0.950        40
```

## Full tiered pipeline

| Metric | Value |
|---|---|
| auto-decided | 39 |
| sent to human review | 1 |
| accuracy on auto-decided | 0.974 |
| passed at phase 1 (no transformer cost) | 7 |

## Phase 3 span extraction

Test set: `span_test.txt` (30 documents, 27 gold spans)

### regex (strict)

```
type       precision    recall        f1  support
micro          0.077     0.111     0.091       27
CLAIM          0.000     0.000     0.000        9
PRICING        0.000     0.000     0.000        9
URGENCY        0.300     0.333     0.316        9
```

### regex (lenient)

```
type       precision    recall        f1  support
micro          0.923     0.889     0.906       27
CLAIM          1.000     0.889     0.941        9
PRICING        0.941     0.889     0.914        9
URGENCY        0.800     0.889     0.842        9
```

### token_classifier (strict)

```
type       precision    recall        f1  support
micro          0.387     0.444     0.414       27
CLAIM          0.500     0.556     0.526        9
PRICING        0.250     0.222     0.235        9
URGENCY        0.385     0.556     0.455        9
```

### token_classifier (lenient)

```
type       precision    recall        f1  support
micro          0.871     0.963     0.915       27
CLAIM          1.000     1.000     1.000        9
PRICING        1.000     0.889     0.941        9
URGENCY        0.692     1.000     0.818        9
```
