from sklearn.ensemble import RandomForestClassifier
from joblib import load
from DynamicSHAP import DSHAP
import numpy as np
        
if __name__ == "__main__":

    ### --- LOAD DATA --- ###
    X1 = np.load('dataset.npy')
    Y = np.load('outcome.npy')

    I, J, K = X1.shape
    dimensions = (I, J, K)

    X = np.empty((I, int(J*K)))
    for i in range(I):
        for j in range(J):
            for k in range(K):
                X[i,j+J*k] = X1[i,j,k]


    indicators = ["Clock time", "Aeration Rate", "Agitator Power", "Glucose Feed Rate", "Glucose Feed Temperature", \
                "Glucose Concentration", "Dissolved O2", "Biomass Concentration", "Penicillin Concentration", \
                " Bulk Volume", "Dissolved CO2", "pH", "Fermentor Temperature", "Generated Heat", "Acid Flowrate", \
                    "Base Flowrate", "Water Flowrate", "Cumulated Acid", "Cumulated Base", "Cumulated Glucose"]

    ### --- LOAD MODEL --- ###
    model = RandomForestClassifier(n_estimators=10000)
    model = load("RF_10000.joblib")

    ### --- PERFORM DSHAP --- ###
    dshap = DSHAP(model, X, dimensions, indicators, 1)
    dshap.fit()
    dshap.plot_continuous_DSHAP()
    dshap.plot_cumulative_DSHAP()