function[patient_results] = combine_results(mri_weight, eeg_weight)
    
    
    load('eeg_analysis.mat');
    load('eeg_test_labels.mat');
    
    %% Look at test labels that model generated 
    % Take an average for each patient
    for i=1:size(patient_IDs,2)
            
        if i == 1
            start_idx = 1;
            end_idx = patient_IDs(i).Trial_Num;
        else 
            start_idx = patient_IDs(i-1).Trial_Num + 1;
            end_idx = start_idx + patient_IDs(i).Trial_Num;
        end
        temp = mean(eeg_test_labels(start_idx:end_idx));
        temp_eeg_patient_labels = struct('patient_id', patient_IDs(i).Patient_ID,...
            'label', temp);
    
        eeg_patient_labels(i) = temp_eeg_patient_labels;
    end 
    
    
    load("mri_labels.mat");
    load("mri_analysis.mat");
    
    for i=1:size(patient_IDs,2)        
        if i == 1
            start_idx = 1;
            end_idx = patient_IDs(i).Trial_Num;
        else 
            start_idx = patient_IDs(i-1).Trial_Num + 1;
            end_idx = start_idx + patient_IDs(i).Trial_Num;
        end
        temp = mean(mri_labels(start_idx:end_idx));
        temp_mri_patient_labels = struct('patient_id', patient_IDs(i).Patient_ID,...
            'label', temp);
    
        mri_patient_labels(i) = temp_mri_patient_labels;
    end 
    
    
    %% Combine 
    for i=1:size(mri_patient_labels,2)
        combine = (mri_patient_labels(i).label)*mri_weight+(eeg_patient_labels(i).label)*eeg_weight/2
        patient_results(i) = combine;
    end 

end    