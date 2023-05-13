function [] = preprocess_eeg(train_ratio, path, split_by_patient)
    
    
    
    %% Find all folders in path 
    patient_folders = dir(path);
    
    num = 1;
    % loop thorugh patient folders 
    for i = 1:size(patient_folders)
        
        % Split patient folder name for later use
        parts = strsplit(patient_folders(i).name, '_');
    
        % loop through technique folders, skipping fodlers not containing
        % images
        if contains(patient_folders(i).name, 'Infarction')
    
            new_path = [path,patient_folders(i).name];
            file_list(num) = dir(fullfile(new_path, '*.csv'));
            num = num + 1;
        
        end
    end
    
    %% Read in all csv files
    for file = 1:size(file_list,2)
        filename = [file_list(file).folder, '/', file_list(file).name];
        temp = readtable(filename,'NumHeaderLines',10);
        temp = temp(:, 3:34); %remove non eeg data columns;
    
        parts = strsplit(file_list(file).folder, '_');
    
        temp_eeg_data = struct('Patient_ID', file_list(file).name, 'Designator', parts(4), ...
                        'Data', temp);
    
        eeg_data(file) = temp_eeg_data;
    
    end
    
    
    %% Split into training and test data
    if split_by_patient == true
        % iterate through each 

        for i = 1:size(eeg_data,2)
            data = eeg_data(i).Data;
            data = transpose(table2array(data));

             %round to nearest number divisible by 1000 which is 2 seconds
             cutoff = floor(size(data,2)/1000)*1000;

            n = floor(size(data,2)/1000);

            if eeg_data(i).Designator == "Conscious"
                labels = ones(n, 1) .* (-1).^(0:n-1)';
                if exist('test_set', 'var')
                    test_set = horzcat(test_set, data(:,1:cutoff));
                    test_labels = vertcat(test_labels, labels);
                else
                    test_set = data(:,1:cutoff);
                    test_labels = labels;
                end 
                
            else
                labels = -1 * ones(n,1);
                if exist('training_set', 'var')
                    training_set = horzcat(training_set, data(:,1:cutoff));
                    training_labels = vertcat(training_labels,labels);
                else
                    training_set = data(:,1:cutoff);
                    training_labels = labels;
                end 
            end   

            temp_patient_IDs = struct("Patient_ID", eeg_data(i).Patient_ID, 'Trial_Num', size(labels));
            patient_IDs(i) = temp_patient_IDs;

        end
    else 
        for i = 1:size(eeg_data,2)
            data = eeg_data(i).Data;
            data = transpose(table2array(data));
        
        
            %round to nearest number divisible by 1000 which is 2 seconds
            cutoff = floor(size(data,2)/1000)*1000;
        
            split_idx = (round(train_ratio*cutoff/1000))*1000; %make sure samples split into 2 second intervals
        
            if exist('training_set', 'var')
                test_set = horzcat(test_set, data(:,split_idx+1:cutoff));
                training_set = horzcat(training_set, data(:,1:split_idx));
            else
                test_set = data(:,split_idx+1:cutoff);
                training_set = data(:,1:split_idx);
            end 
        
            n = cutoff/1000;
        
            if eeg_data(i).Designator == "Conscious"
                % alternate between 1 and -1 for squeezing hand and not 
                %labels = ones(n, 1) .* (-1).^(0:n-1)';
                labels = ones((cutoff/1000),1);
            else
                % alternate between 1 and -1 for squeezing hand and not  
                %labels = ones(n, 1) .* (-1).^(0:n-1)';
                labels = -1 * ones((cutoff/1000),1);
            end 
        
            if exist('training_labels', 'var')
                test_labels = vertcat(test_labels, labels((split_idx/1000)+1:((cutoff)/1000)));
                training_labels = vertcat(training_labels, labels(1:(split_idx)/1000));
            else
                test_labels = labels((split_idx/1000)+1:((cutoff)/1000));
                training_labels = labels(1:(split_idx)/1000);
            end
        
            temp_patient_IDs = struct("Patient_ID", eeg_data(i).Patient_ID, 'Trial_Num', size(labels((split_idx/1000)+1:((cutoff)/1000)-1),1));
            patient_IDs(i) = temp_patient_IDs;
        
        end 
    end
    
    %% Split into trials 
    
    total_test_trials = size(test_set,2)/1000;
    total_train_trials = size(training_set,2)/1000;
    
    
    
    start = 1;
    finish = 1000;
    
    X = zeros(total_test_trials-1,32,1000);
    for i = 1:((total_test_trials)-1)
        X(i,:,:) = test_set(:,start:finish);
        start = start + 1000;
        finish = finish + 1000;
    end
    
    Y = test_labels(1:end-1);
    
    save('test.mat','X', "Y")
    
    %%
    
    clear X
    clear Y
    
    start = 1;
    finish = 1000;
    X = zeros((total_train_trials-1),32,1000);
    for i = 1:((total_train_trials)-1)
        X(i,:,:) = training_set(:,start:finish);
        start = start + 1000;
        finish = finish + 1000;
    end
    
    Y = training_labels(1:end-1);
    
    save('train.mat','X', "Y")
    
    
    save("eeg_analysis.mat", 'patient_IDs')
end
