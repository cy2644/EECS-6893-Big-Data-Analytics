function[] = preprocess_mri(path)
    
    %Initialize varibles
    %dcm_paths = struct();
    num = 1;
    
    % Find all folders in path 
    patient_folders = dir(path);
    
    % loop thorugh patient folders 
    for i = 1:size(patient_folders)
        
        % Split patient folder name for later use
        parts = strsplit(patient_folders(i).name, '_');
    
        % loop through technique folders, skipping fodlers not containing
        % images
        if contains(patient_folders(i).name, 'Infarction')
    
            new_path = [path,patient_folders(i).name];
            technique_folders = dir(new_path);
    
        
            % loop thorugh each and save dcm images
            for j = 1:size(technique_folders)
                newer_path = [new_path,'/',technique_folders(i).name];
                file_list = dir(fullfile(newer_path, '*.dcm'));
    
                % Split technique folder name for later use
                parts2 = strsplit(technique_folders(i).name, '_');
    
                % Save all relevant file info in new struct
                for k = 1:size(file_list)
                    temp = struct('Patient_ID', parts(1), 'Designator', parts(3), ...
                        'Imaging_Type', parts2(3), 'Technique', parts2(4), ...
                        'Path', newer_path, 'File_Name', file_list(k).name); 
                    
                    dcm_paths(num) = temp;
                    num = num + 1;
                    
                end 
            end
        end
    end
    
    %% Extract image data using paths 
    
    for i=1:length(dcm_paths)
        image_path = [dcm_paths(i).Path, '/', dcm_paths(i).File_Name];
        image = dicomread(image_path);
        temp_image_data = struct('Patient_ID', dcm_paths(i).Patient_ID, 'Designator', dcm_paths(i).Designator, ...
                        'Imaging_Type', dcm_paths(i).Imaging_Type, 'Technique', dcm_paths(i).Technique, ...
                        'Data', image, 'Number', i);
    
        image_data(i) = temp_image_data;
    
    end
    
    %% Save image data to mat file
    save('image_data', 'image_data')
end