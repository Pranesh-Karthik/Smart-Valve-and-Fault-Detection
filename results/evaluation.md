# Evaluation (synthetic data)

Train: 1200; test: 300. Seed: 42.

5-fold CV: 0.9542 +/- 0.0134. Test accuracy: 0.9667.

```text
               precision    recall  f1-score   support

   all_closed       0.93      0.95      0.94        60
  valve1_open       0.95      0.92      0.93        60
  valve2_open       0.97      0.98      0.98        60
  valve3_open       0.98      0.98      0.98        60
leak_detected       1.00      1.00      1.00        60

     accuracy                           0.97       300
    macro avg       0.97      0.97      0.97       300
 weighted avg       0.97      0.97      0.97       300
```

Scaling is fitted inside each CV fold. The hold-out set is excluded from CV.

Random window split; real repeated runs need grouped evaluation. Synthetic scores do not validate industrial performance.
