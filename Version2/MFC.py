import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def MFC():
    state_df1 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_12_processed.csv')
    state_df2 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_13_processed.csv')
    state_df3 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_14_processed.csv')
    state_df4 = pd.read_csv('/home/nlsde/RLmodel/Version3/src/w3_res_15_processed.csv')

    #Deal with NaN
    state_space_df1 = state_df1.fillna(state_df1.shift())
    state_space_df2 = state_df2.copy()
    state_space_df3 = state_df3.copy()
    state_space_df4 = state_df4.copy()

    for col in state_df2.columns:
        state_space_df2[col] = state_space_df2[col].fillna(state_space_df1[col])
        state_space_df3[col] = state_space_df3[col].fillna(state_space_df1[col])
        state_space_df4[col] = state_space_df4[col].fillna(state_space_df1[col])

    state_space_df1 = state_space_df1.ffill()
    state_space_df2 = state_space_df2.ffill()
    state_space_df3 = state_space_df3.ffill()
    state_space_df4 = state_space_df4.ffill()

    # Convert DataFrame to numpy arrays for KDE computation
    state_1_np = state_space_df1.to_numpy()
    state_2_np = state_space_df2.to_numpy()
    state_3_np = state_space_df3.to_numpy()
    state_4_np = state_space_df4.to_numpy()

    # Assuming state_1_np, state_2_np, state_3_np, state_4_np are numpy arrays of appropriate shapes
    state_space = []

    for i in range(len(state_1_np)):
        # Extract the data from each state for this iteration
        data1 = state_1_np[i]
        data2 = state_2_np[i]
        data3 = state_3_np[i]
        data4 = state_4_np[i]
        
        # Initialize a list to hold KDEs for each dimension
        kdelis = []
        
        # Loop over each of the 11 dimensions (1 to 10)
        for dim in range(11):
            # Collect the data for the current dimension across all states
            dimension_data = np.array([data1[dim], data2[dim], data3[dim], data4[dim]])
            
            # Reshape to 2D array (4 samples, 1 dimension)
            dimension_data_reshaped = dimension_data.reshape(1, -1)  # shape (1, 4)
            
            # Attempt to compute KDE for this dimension
            try:
                kde = gaussian_kde(dimension_data_reshaped) 
                data = kde.resample(1).flatten()
                sampled_data = data[0]
            except Exception as e:
                sampled_data = np.mean(dimension_data)
            
            # Append the KDE or mean to the list
            kdelis.append(sampled_data)
        
        # Append the list of KDEs/means for this state to the state_space
        state_space.append(kdelis)

    state_space_df = pd.DataFrame(state_space)
    return state_space_df, state_df1