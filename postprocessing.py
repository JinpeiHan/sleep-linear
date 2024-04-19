import os
import pickle
import numpy as np


# Dictionary to hold concatenated data based on patient_id
patient_data = {}
folder_path = './features'
output_folder = './processed_features'

valid_labels = {
    "Sleep stage W",
    "Sleep stage 1",
    "Sleep stage 2",
    "Sleep stage 3",
    "Sleep stage R"
}

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.p'):
        with open(os.path.join(folder_path, filename), 'rb') as file:
            # Load the data structured as [raw_data, patient_features, labels, patient_ids]
            raw_data, patient_features, labels, patient_ids = pickle.load(file)

            labels[labels == "Sleep stage 4"] = "Sleep stage 3"

            # Iterate through the unique patient_ids and concatenate data accordingly
            for pid in np.unique(patient_ids):
                idx = patient_ids == pid  # Indices of current patient_id
                data_tuple = (raw_data[idx], patient_features[idx], labels[idx], patient_ids[idx])

                if pid in patient_data:
                    # Concatenate data along the first dimension for existing patient_id
                    patient_data[pid] = tuple(np.concatenate([patient_data[pid][i], data_tuple[i]]) for i in range(4))
                else:
                    # Initialize data for new patient_id
                    patient_data[pid] = data_tuple

    # Remove entries with nan values in patient_features or labels
for pid, data in list(patient_data.items()):
    raw_data, patient_features, labels, patient_ids = data
    # Check for nan in patient_features and labels
    nan_mask = np.isnan(patient_features).any(axis=1) | np.isnan(labels).any(axis=1)
    # Filter out labels not in the valid set
    label_mask = np.isin(labels, list(valid_labels))

    # Combine the nan and label filters to determine valid entries
    valid_indices = ~nan_mask & label_mask
    if np.any(valid_indices):
        # Filter out the entries where any invalid values are present
        filtered_data = tuple(data_part[valid_indices] for data_part in data)
        output_file_path = os.path.join(output_folder, f'patient_{pid}.p')
        with open(output_file_path, 'wb') as output_file:
            pickle.dump(filtered_data, output_file)
    else:
        # If no valid data left, do not save any data for this patient
        continue
    #     patient_data[pid] = filtered_data
    # else:
    #     # If no valid data left, remove the patient entry
    #     del patient_data[pid]


