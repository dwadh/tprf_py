import data_gen as dg
import tprf_code
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
Generate a list of parameter combinations to be used.
Parameters: pf, pg, a, d
The naming convention is same as the one in paper and mentioned in data_gen.py
"""
p_values = [0.3, 0.9]
combos = []
combo = {}
for i in [0, 1]:
    for a in [0.3, 0.9]:
        for d in [0, 1]:
            combo = {"pf": p_values[i], "pg": p_values[1 - i], "a": a, "d": d}
            combos.append(combo)
combo = {}
combo['pf'] = 0
combo['pg'] = 0
combo['a'] = 0
combo['d'] = 0
combos.insert(0, combo)

# Define the base data dictionary with common parameters across all combinations
data_dict = {"T": 200,
             "N": 200,
             "relevant_factors": 1,
             "irr_factors": 4}

"""
Generate the data dictionary using the combinations created above
and the base dictionary (with common parameters).
Run the simulation with the generated data with each model.
The naming convention is same as the one in paper and mentioned in data_gen.py
"""

# Iterate over factor strength 1,2,3 (Weak, Moderate, Normal).
# #For moderate factors, use pervasive and non-pervasive

for strength in [4, 2.33]:
    if strength == 4:
        non_pervasive = [False, True]
    else:
        non_pervasive = [False]
    for j in non_pervasive:
        df = pd.DataFrame(columns=["pf", "pg", "a", "d", "3PRF1", "PCR5", "PCLAS"])

        # Iterate over each combination of parameters and update data dictionary
        for combo in combos:
            print (strength, j, combo)
            for key in combo:
                data_dict[key] = combo[key]
            data_dict["non_pervasive"] = j
            data_dict["strength"] = strength

            # Run the simulation n_iter times
            n_iter = 200
            r2_tprf_lst = []
            r2_pcr_lst = []
            r2_pclas_lst = []
            pcr_pclas = {"pcr": [],
                         "pclas": []}

            for i in tqdm(range(0, n_iter)):
                X, y = dg.data_generator(data_dict)
                Z = 1
                train_window = int(data_dict["T"] / 2)
                result = tprf_code.recursive_train(X, y, Z, train_window)
                r2_tprf_lst.append(result)
                for procedure in ("pcr", "pclas"):
                    try:
                        result = tprf_code.recursive_train_alternate(X, y, train_window, 5, procedure)
                        pcr_pclas[procedure].append(result)
                    except:
                        print("\n\nError:", strength, j, combo, X.shape, y.shape, procedure, i, "\n\n")
                        continue
            tprf_result = np.median(r2_tprf_lst)*100
            pcr_result = np.median(pcr_pclas['pcr'])*100
            pclas_result = np.median(pcr_pclas['pclas'])*100
            print ("\nTPRF: ", tprf_result, "\n")
            print ("\nPCR: ", pcr_result, "\n")
            print ("\nPCLAS: ", pclas_result, "\n")
            data_row = {"pf": data_dict["pf"], "pg": data_dict["pg"], "a": data_dict["a"], "d": data_dict["d"],
                        "3PRF1": tprf_result, "PCR5": pcr_result, "PCLAS": pclas_result}
            df = df.append(data_row, ignore_index=True)
        if strength == 4:
            if j:
                f_name = "moderate_non_pervasive.xlsx"
            else:
                f_name = "moderate.xlsx"
        elif strength == 9:
            f_name = "weak.xlsx"
        else:
            f_name = "normal.xlsx"
        save_dir = "data_200/" + f_name
        df.to_excel(save_dir)
        print ("Saved: ", save_dir)
