# SMILES to molecular graphs

Code for preprocessing, i.e., converting SMILES strings of HCO-molecules into molecular graphs and assigning target properties. 

* **read_in_multitask** Read in list of molecules from csv file. Converting SMILES strings into molecular graphs and assign multiple target properties (DCN, MON, and RON) to one molecule. Store processed data in target directory as sdf and pt files.

* **read_in_singletask** Read in list of molecules from csv file. Converting SMILES strings into molecular graphs and assign one target property (DCN, MON, or RON) to one molecule. Store processed data in target directory as sdf and pt files.

* **single_molecule_conversion** Function to convert single SMILES string into molecular graph. Return molecular graph. 

