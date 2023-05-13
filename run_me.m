%% Clear variables and set 
clear; clc; close all;

%Path to patient data
path = 'data/MRI-EEG-March2023/';
%fraction of data to be put into training group
train_ratio = .4;
split_by_patient = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Preprocess the EEG data. 
% The preprocess_eeg functions looks in "path" folder 
% for any folder with the following format patientID_infarction_Designation
% where patient_ID is an 8 diget valye and Designation is either unconcious
% or conscious. If the folder does not follow this format and contain the
% work infarction, it will be ignored. The preprocess_eeg functions grabs
% any .cvs file and saves it into a record with the patient id and
% designation. The data is parsed into 2 second segments and split into
% training and testing sets. The amount of data used for training can be
% changed with the train_ratio below. The function will save the training
% and test data and their labels in .mat files for use by other functions. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

preprocess_eeg(train_ratio, path, split_by_patient)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EEG ANALYSIS
% The eeg analysis function loads in the .mat files created during pre
% processing and creates an SVM model using features extracted by taking the 
% power spectral density. The analysis can be run in two different modes. 
%% INPUTS:
%%  There are two optional input arguments.
%%     1. 'manual'
%%        - When manual is selected you must specify an array of frequencies
%%          to be used as features. The frequencies must be a positive integer
%%          less than or equal to 500Hz. 
%%          The following example will use the power from frequencies 5,6,7 Hz
%%          from all 32 electrodes as the feature space.
%%          ex: eeg_analysis('manual', [5 6 7]) or
%%              eeg_analysis('manual', [6:7])
%%     2. 'auto'
%%        - When auto is selected you must specify how many frequencies you
%%          would like to sort through. The frequencies are chosen by the
%%          algorithm. Of the chosen frequencies, the algorithm will determine
%%          which ones are best to keep, anchd then discard the unwanted ones.
%%          The following example will sort through 4 different frequencies 
%%          from all 32 electrodes.
%%        ex: eeg_analysis('auto', 4)
%% OUTPUTS:
%%   eeg_test_labels is a vector of labels of the test data chosen by the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% eeg_analysis is being run in manual model and three frerquencies in the
% range of alpha, beta, and theta waves are chosen. 
eeg_test_labels = eeg_analysis('manual', [4 8 12]);

% The output lablels are then saves for use by the combined analysis
% process
save('eeg_test_labels.mat', 'eeg_test_labels')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PREPROCESS MRI
% The preprocess_mri functions looks in "path" folder 
% for any folder with the following format patientID_infarction_Designation
% where patient_ID is an 8 diget valye and Designation is either unconcious
% or conscious. If the folder does not follow this format and contain the
% work infarction, it will be ignored. The preprocess_mri functions grabs
% any .dcm file and saves it into a record with the patient id, technique
% type, designation, and other relevant information pulled from the folder title.
% See the provided data folder for formatting. If desired data is organized
% differently, the preprocessing file may need to be updated to accomodate
% the input data
% INPUTS:
%   train_ration => fraction of data to be designated to train the model
%   split_by_patient => if true, the training data will split the data in
%   half my patient (if there are 4 patients, complete data for 2 patients
%   will be designated for training and 2 will be designated for testing.
%   if false, then the mri data for each patient will be split into
%   training and test sets based on the training ratio. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

preprocess_mri(path)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MRI Support Vector Machine model 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_ratio = .6;
split_by_patient = false;

mri_labels = mri_svm(train_ratio, split_by_patient);
save('mri_labels.mat', 'mri_labels')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Combine EEG and MRI Results
% combine results looks at the results from the mri and eeg models and
% averages the results of each trial to generate a value between -1 and 1
% for each patient. Each value is then multiplied by the given weight to
% determing a combined value for each patient. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mri_weight = .5;
eeg_weight = .5;
patient_results = combine_results(mri_weight, eeg_weight);