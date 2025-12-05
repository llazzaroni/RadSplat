import numpy as np

def find_peaks(density):
    # First find the peaks (the ones that are relevant)
    max_val = density.max()
    indices = []
    for i in range(1, len(density) - 1):
        if (density[i-1] < density[i]) and (density[i] > density[i+1]) and (density[i] > (max_val / 8)):
            indices.append(i)
    
    #print(indices)

    # Then prune them
    while len(indices) > 4:
        #print(indices)
        #print(indices)
        closest_indices = find_closest_indices(indices, len(density))
        #print(closest_indices)
        if density[closest_indices[0]] < density[closest_indices[1]]:
            index_to_remove = closest_indices[0]
        else:
            index_to_remove = closest_indices[1]
        
        indices.remove(index_to_remove)
        


    


    return indices

def find_closest_indices(indices, len_density):
    result = [-1, len_density]
    for i in range(1, len(indices)):
        if indices[i] - indices[i-1] < result[1] - result[0]:
            result[1] = indices[i]
            result[0] = indices[i-1]

    return result