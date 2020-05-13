# Trained model

Code of model ensemble predicting DCN, MON, and RON of non-oxygenated and oxygenated hydrocarbons. In total, 40 multitask models trained on DCN, MON, and RON data are utilized to make a prediction, i.e., average predictions of all models.

* **predict_DCN_MON_RON** Predictions for a list of molecules, input via Data folder.

* **predict_DCN_MON_RON_single_mol** Predictions for a single molecule, input via command line, e.g., run

```
python predict_DCN_MON_RON_single_mol.py --mol 'CCc1ccc(OC)c(O)c1'
```

