import numpy as np
from scipy.stats import gaussian_kde

def compute_kde_or_mean(state_space_df1, state_space_df2, state_space_df3, state_space_df4):
    # Convert DataFrames to numpy arrays
    state_1_np = state_space_df1.to_numpy()
    state_2_np = state_space_df2.to_numpy()
    state_3_np = state_space_df3.to_numpy()
    state_4_np = state_space_df4.to_numpy()
    
    state_space = []  # Initialize the list to store processed states
    
    # Loop over each row (state) in the arrays
    for i in range(len(state_1_np)):
        # Extract the data from each state for this iteration
        data1 = state_1_np[i]
        data2 = state_2_np[i]
        data3 = state_3_np[i]
        data4 = state_4_np[i]
        
        # Initialize a list to hold KDEs (or means) for each dimension
        kdelis = []
        
        # Loop over each of the 11 dimensions (assuming there are 11 dimensions)
        for dim in range(11):
            # Collect the data for the current dimension across all states
            dimension_data = np.array([data1[dim], data2[dim], data3[dim], data4[dim]])
            
            # Check if the variance is too small (e.g., variance close to 0)
            if np.var(dimension_data) < 1e-6:
                #print(f"Low variance in dimension {dim} at state {i}, using mean instead of KDE.")
                sampled_data = np.mean(dimension_data)  # Fall back to mean
            else:
                # Attempt to compute KDE for this dimension
                try:
                    kde = gaussian_kde(dimension_data)  # KDE works with 1D data
                    sampled_data = kde.resample(1).flatten()[0]
                except Exception as e:
                    print(f"Error in KDE for dimension {dim} at state {i}: {e}")
                    sampled_data = np.mean(dimension_data)  # Fall back to mean if KDE fails
            
            # Append the KDE or mean to the list
            kdelis.append(sampled_data)
        
        # Append the list of KDEs/means for this state to the state_space
        state_space.append(kdelis)
    
    return np.array(state_space)  # Return as numpy array for consistency
