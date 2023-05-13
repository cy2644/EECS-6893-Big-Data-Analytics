function predictedLabels = mri_svm(train_ratio,split_by_patient)
    
    
    %% Initialize vairble 
    num = 1;
    
    %% Load availble image data 
    load("image_data.mat")
    
    
    %% lets look at the makeup of the data
    pateint_entries = unique({image_data.Patient_ID});
    num_patient = numel(pateint_entries);
    
    image_type_entries = unique({image_data.Imaging_Type});
    num_imaging_type = numel(image_type_entries);
    
    technique_entries = unique({image_data.Technique});
    num_techniques = numel(technique_entries);
    
    
    %% split data into training and test sets
    train_num = 1;
    test_num = 1;
    if split_by_patient == true
        for patient = 1:num_patient %loop through patients
            for i=1:size(image_data,2)
                if strcmp(image_data(i).Patient_ID, pateint_entries(patient)) &&  mod(patient,2) == 0                
                    training_set(train_num) = image_data(i);
                    train_num = train_num + 1;
                end
    
                if strcmp(image_data(i).Patient_ID, pateint_entries(patient)) &&  mod(patient,2) == 1                
                    test_set(test_num) = image_data(i);
                    test_num = test_num + 1;
                end
                 
             end
    
         end
    
    
    else 
        for patient = 1:num_patient %loop through patients
            for imaging_type = 1:num_imaging_type %loop through imaging type
                for technique = 1:num_techniques % loop through techniques
                    %Loop through all entires in structure and check for conditions
                    for i=1:size(image_data,2)
                        if strcmp(image_data(i).Patient_ID, pateint_entries(patient))...
                               && strcmp(image_data(i).Imaging_Type, image_type_entries(imaging_type))...
                               && strcmp(image_data(i).Technique, technique_entries(technique))
                               
                            temp(num) = image_data(i);
                            num = num+1;
        
                        end
                    end
                    
                    if exist('temp', 'var')
                        n = train_ratio*size(temp,2);
                        if exist('training_set', 'var')
                            training_set = horzcat(temp(1:n), training_set);
                            test_set = horzcat(temp(n+1:end), test_set);
                        else
                            training_set = temp(1:n);
                            test_set = temp(n+1:end);
                        end
                    
                    trails = size(temp(n+1:end),2);

                    temp_patient_IDs = struct("Patient_ID", pateint_entries(patient), 'Trial_Num', trails);
                    patient_IDs(patient) = temp_patient_IDs;
                    
                    end

                    

                    clear temp;
                    num = 1;
                end
        
            end 
        end
    end
    %% Extract Features
    % Scale-Invariant Feature Transform (SIFT)
    for i = 1:size(training_set,2)
        % Convert the DICOM image to grayscale
        grayImg = mat2gray(training_set(i).Data);
        
        % Extract points and features
        points = detectSURFFeatures(grayImg);
        features = extractFeatures(grayImg, points);
        
    
        % Resize features because they are different for each image
        if i~= 1
            features = imresize(features, size(TRAIN_featureMatrix(:,:,i-1)));
        end
    
        % Store the features in the 3D matrix
        TRAIN_featureMatrix(:,:,i) = features;
    
        % Convert conscious and unconcious labels to 1 and -1
        if training_set(i).Designator == "Conscious"
            TRAIN_labels(i) = 1;
        else
            TRAIN_labels(i) = -1;
        end
    end
    
    % Reshape the 3D feature matrix into a 2D feature matrix
    TRAIN_featureMatrix2D = reshape(TRAIN_featureMatrix, [size(TRAIN_featureMatrix,1)*size(TRAIN_featureMatrix,2), size(TRAIN_featureMatrix,3)]).';
    
    for i = 1:size(test_set,2)
        % Convert the DICOM image to grayscale
        grayImg = mat2gray(test_set(i).Data);
        
        % Extract points and features
        points = detectSURFFeatures(grayImg);
        features = extractFeatures(grayImg, points);
    
        % Resize features because they are different for each image
        % and need to be the same size as the training set 
        if i~= 1
            features = imresize(features, size(TEST_featureMatrix(:,:,i-1)));
        end
    
        % Store the features in the 3D matrix
        TEST_featureMatrix(:,:,i) = features;
    
        % Convert conscious and unconcious labels to 1 and -1
        if test_set(i).Designator == "Conscious"
            TEST_labels(i) = 1;
        else
            TEST_labels(i) = -1;
        end
    end
    
    % Reshape the 3D feature matrix into a 2D feature matrix
    TEST_featureMatrix2D = reshape(TEST_featureMatrix, [size(TEST_featureMatrix,1)*size(TEST_featureMatrix,2), size(TEST_featureMatrix,3)]).';
    
    % Reshape the feature matrices to have the same column length
    % Find the number of columns in the smaller matrix
    numCols = min(size(TEST_featureMatrix2D, 2), size(TRAIN_featureMatrix2D, 2));
    
    % Reshape a and b to have the same number of columns
    TEST_featureMatrix2D = TEST_featureMatrix2D(:, 1:numCols);
    TRAIN_featureMatrix2D = TRAIN_featureMatrix2D(:, 1:numCols);
    
    
    %% Train Model
    
    params = struct('BoxConstraint', optimizableVariable('BoxConstraint', [0.01, 10], 'Transform', 'log'), ...
                    'KernelScale', optimizableVariable('KernelScale', [1e-5, 1e5], 'Transform', 'log'));
    
    
    % Set up hyperparameter optimization options
    opts = struct('Optimizer', 'bayesopt', ...
              'MaxObjectiveEvaluations', 30, ...
              'AcquisitionFunctionName', 'expected-improvement-plus', ...
              'ShowPlots', true);
    if split_by_patient == true
        svmModel = fitcsvm(TRAIN_featureMatrix2D, TRAIN_labels);
    else 
        % Train SVM model with hyperparameter optimization
        svmModel = fitcsvm(TRAIN_featureMatrix2D, TRAIN_labels, 'OptimizeHyperparameters', 'auto', ...
                   'HyperparameterOptimizationOptions', opts);
    end
    % Train SVM model
    % svmModel = fitcsvm(TRAIN_featureMatrix2D, TRAIN_labels, 'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', params);
    
    % Predict labels for test set
    predictedLabels = predict(svmModel, TEST_featureMatrix2D);
    
    
    
    %% Evaluate model accuracy
    predictedLabels = predictedLabels';
    sum = 0;
    for label = 1:size(TEST_labels,2)
        if TEST_labels(label) == predictedLabels(label)
            sum = sum + 1;
        end
    end 
    
    accuracy = sum/size(TEST_labels,2);
    
    
    disp(['Accuracy: ' num2str(accuracy)]);

    save("mri_analysis.mat", 'patient_IDs')

end

%accuracies(ratio) = accuracy;

















