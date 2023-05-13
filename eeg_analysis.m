
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  TITLE: EEG Analysis of Patients with TBI
%%
%%  FILENAME: eeg_analysis.m
%%
%%  AUTHOR: Jesse Kotsch
%%     
%%
%%  REVISION: 1.1
%%  DATE: 03/25/2023
%%
%%  INPUTS:
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
%%          ex: eeg_analysis('auto', 4)
%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%function [keep1, keep2, elim1, elim2, test_labels] = eeg_analysis(varargin)

function [test_labels] = eeg_analysis(varargin)



close all;clearvars -except varargin;clc;

trial_elimination = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Parse through the Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = inputParser;

% Parse 'manual'
errorMsg = 'Frequencies must be positive, and numeric, less than 501'; 
validationFcn = @(x) assert(isnumeric(x) && all(x > 0) && all(x <501),errorMsg);
addOptional(p,'manual',[4:7 9:11],validationFcn)

% Parse 'auto'
errorMsg = 'Frequencies must be positive, and numeric, less than 501'; 
validationFcn = @(x) assert(isnumeric(x) && all(x > 0) && all(x <501),errorMsg);
addOptional(p,'auto',4,validationFcn)

% Parse 'chan'
errorMsg = 'Channels must be integers within the range of 1 to 32'; 
validationFcn = @(x) assert(isnumeric(x) && all(x > 0) && all(x <65),errorMsg);
addOptional(p,'chan',[1:32],validationFcn)

parse(p,varargin{:})

valid_argnames  = {'manual', 'auto', 'chan'};
argwasspecified = ismember(valid_argnames, varargin(1:2:end));

% Load the raw data
raw_train_load     = load('train.mat');
raw_test_load      = load('test.mat');
%raw_test_sol_load = load('Competition_test_sol.mat');
[m,n,o] = size(raw_train_load.X);
[m2,n2,o2] = size(raw_test_load.X);

% raw_train = abs(raw_train_load.X(1:m/2,:,:));
% raw_test = abs(raw_train_load.X(m/2+1:m,:,:));
% labels = raw_train_load.Y(1:m/2,:);
% raw_test_sol = raw_test_load.Y(m/2+1:m,:);

raw_train = abs(raw_train_load.X);
raw_test = abs(raw_test_load.X);
labels = raw_train_load.Y;
raw_test_sol = raw_test_load.Y;

% Brain activity matrix (trials, electrode channel, samples of time series)

%ecog = raw_train.X;
%ecog_test = raw_test.X;




%ecog = raw_train_load.X(1:m/2,:,:);
%ecog_test = raw_train_load.X(m/2+1:m,:,:);

ecog = raw_train_load.X;
ecog_test = raw_test_load.X;

% Constants
C_FS          = 500;             % sampling frequency
C_LENGTH      = 1000;            % Number of samples collected per trial per electrode
C_SAMP        = 1000;            % Amount of samples to extract
C_TRIALS      = m+m2;            % Number of trials
C_SAMPLES     = C_TRIALS*1000;   % Number of samples
C_ELEC        = 32;              % Number of electrodes
C_FEAT        = p.Results.auto;  % Feature Count
C_CHAN        = p.Results.chan;  % Channels
C_CHAN_THRESH = 0.0;             % Power threshold for channel_selection function

%Plot first 4 channels from trial 1 for visualization
figure()
for i =1:4
   handles(i) = subplot(4,1,i);
   plot(reshape(ecog(1,i,:),[1,C_LENGTH]));
   set(gca,'xticklabel',[])
   grid on;
end
h=get(handles(1),'title');set(h,'String','First 4 Electrodes From Trial 1');
h=get(handles(4),'xlabel');set(h,'String','time [ms]');set(gca,'xticklabel',[0:500:C_LENGTH]);
h=get(handles(2),'ylabel');set(h,'String','muV');
clear i; clear h; clear handles;

% Power spectral density estimate
% In this project the professor has suggested that we extract features based
% on the power spectral density. We will then examine frequencies that seem to
% have greater importance.
for i=1:size(ecog,1)
   ecog_temp(:,:)=ecog(i,:,:);
   [pxx,f] = pwelch(ecog_temp',500,300,500,C_FS);
   ecog_psd(i,:,:)=20*log10(pxx');
end


old_ecog_psd = ecog_psd;
old_lables = labels;
gg = 1;
%for gauss_thresh = 2:-.1:0

gauss_thresh = .5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Trial Elimination
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if trial_elimination == true
    [train_data, new_labels, number_trials, keep1, keep2, elim1, elim2] = trial_elimination(old_ecog_psd, old_lables, raw_train, gauss_thresh);
    clear ecog;
    clear labels;
    clear C_TRIALS;
    clear ecog_temp;
    clear ecog_psd;
    clear pzz;
    clear f;

else
    train_data = raw_train;
    new_labels = labels;
    number_trials = C_TRIALS;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ecog = train_data;
labels = new_labels;
C_TRIALS = number_trials;             % Number of trials

%recalculate ecog psd
for i=1:size(ecog,1)
   ecog_temp(:,:)=ecog(i,:,:);
   [pxx,f] = pwelch(ecog_temp',500,300,500,C_FS);
   ecog_psd(i,:,:)=20*log10(pxx');
end

save('pxx.mat', 'pxx')

%Plot the power spectral density across the first channel for the first trial
figure();
subplot(2,1,1);
plot(f,reshape(ecog_psd(1,1,:),[1, size(ecog_psd,3)]));
title('Welch Power Spectral Density Estimate for Trial 1, Electrode 1');
xlabel('Frequency (Hz)');ylabel('PSD (dB/Hz)');grid on
% Plot the power spectral density across all channels for the first trial 
% figure();
subplot(2,1,2);
imagesc(f,[],reshape(ecog_psd(1,:,:),[C_ELEC, size(ecog_psd,3)]));colorbar
title('Welch Power Spectral Density Estimate for Trial 1');
xlabel('Frequency (Hz)');ylabel('ECoG Electrode');
clear i; clear ecog_temp; clear pxx; clear f;

% Select which channels we want to use
if argwasspecified(3)
   C_CHAN = C_CHAN;
   channel_selection(ecog_psd, labels, C_CHAN_THRESH);
else
   C_CHAN = channel_selection(ecog_psd, labels, C_CHAN_THRESH);
end

% Iterate through all frequencies in the Power Spectral Density plots.
% This function will find the label (one of two) assosiated to each PCA score, then
% an average location of both (two) point clouds is found. Then the distance between
% both point clouds is calculated. We want to find the largest distance between
% the two point clouds and use the cooresponding frequency as our extracted feature.
[sorted_dist, dist] = rate_features(ecog_psd, labels);

save('ecog_psd.mat', 'ecog_psd')

if argwasspecified(1)
   loop_length = 1;
else
   loop_length = C_FEAT;
end

for ii = 1:loop_length
   if argwasspecified(1)
      c_list(ii,:) = round(linspace(1,min(length(C_CHAN)*length(p.Results.manual),C_TRIALS),10)); % # of principal components in PCA
   else
      c_list(ii,:) = round(linspace(1,min(length(C_CHAN)*ii,C_TRIALS),10)); % # of principal components in PCA
   end
   
   for i = 1:ii
      if argwasspecified(1)
         for iii = 1:length(p.Results.manual)
            frequencies(ii,iii) = p.Results.manual(iii);
         end
      else
         frequencies(ii,i) = find(dist==sorted_dist(i));
      end
   end
   clear i;
   
   disp('**********************************************************************')
   msg = ['Using frequencies ',strjoin(arrayfun(@(x) num2str(x),frequencies(ii,:),...
         'UniformOutput',false),','),' as our extracted features.'];
   disp(msg)
   msg = ['Using channels ',strjoin(arrayfun(@(x) num2str(x),C_CHAN,...
         'UniformOutput',false),','),'.'];
   disp(msg)
   disp('**********************************************************************')
%    msg = ['The mean distance between the first two pca components with respect to label is '...
%           , 'num2str(round(val,2)), '.'];
%    disp(msg)
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%% Features
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   if argwasspecified(1)
      ecog_features = reshape(ecog_psd(:,C_CHAN,p.Results.manual), [length(labels) length(C_CHAN)*length(p.Results.manual)]);
   else
      ecog_features = reshape(ecog_psd(:,C_CHAN,frequencies(ii,:)), [length(labels) length(C_CHAN)*ii]); % Extract frequencies
   end
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   
   centered = bsxfun(@minus, ecog_features, mean(ecog_features));
   [U, S, V] = svd(centered);
   scores = U*S;
   % Find actual classification of the training data (used for visualization)
   BrainActivation = scores(find(labels ==  1),:);
   NoBrainActivation  = scores(find(labels == -1),:);
   
   % Plot the PCA 1st vs. 2nd score. Note that i've also added the actual
   % classification for visual assurance that we have chosen a good set of
   % features for classification.
   figure();
   subplot(2,2,1.5);
   scatter(BrainActivation(:,1),BrainActivation(:,2),'r*');hold on;
   scatter(NoBrainActivation(:,1),NoBrainActivation(:,2),'b*');
   title(['PCA 1st vs. 2nd Score (',...
         strjoin(arrayfun(@(x) num2str(x),frequencies(ii,:),...
         'UniformOutput',false),','),' Hz)']);
   xlabel('1st component score');ylabel('2nd component score');
   clear NoBrainActivation; clear BrainActivation;
   subplot(2,1,2);
   % Plot the the Square-Rooted Eigenvalue Spectrum
   scatter(1:length(diag(S)),diag(S),'go');hold on;grid on;
   title(['Square-Rooted Eigenvalue Spectrum (',...
         strjoin(arrayfun(@(x) num2str(x),frequencies(ii,:),...
         'UniformOutput',false),','),' Hz)']);
   xlabel('component number');ylabel('Square-Root Eigenvalue');
   clear tmp;
   
   clear idx; clear BrainActivation; clear NoBrainActivation; clear msg;
   clear meanx_t; clear meanx_f; clear meany_t; clear meany_f;
   
   j=1;
   for i = c_list(ii,:)
      % Here we will apply PCA to reduce the dimensionality of the data while keeping
      % most of the explained variance
      centered = bsxfun(@minus, ecog_features, mean(ecog_features));
      [U, S, V] = svd(centered);
      
      % Reduce the dimensionality of the data while preserving
      % the variance.
      % tmp = cumsum(diag(S))/sum(diag(S));
      diag_S = diag(S);
      tmp = sum(diag_S(1:i))/sum(diag_S);
      msg = ['-PCA trial #',num2str(j),];
      disp(msg)
      msg = ['Reducing dimensionality while preserving ',num2str(round(tmp*100,2)),'% of the variance.'];
      disp(msg)
      msg = ['Reduced features from ', num2str(length(diag_S)), ' dimensions to ', num2str(i), ' dimensions.'];
      disp(msg)
      clear msg;
      V = V(:,1:i);
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %% SVM
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      A_train = (V*(V'*(ecog_features')))';
      Hyper   = dykstra_svm(A_train, labels, 100000);
      
      comp_labels = A_train*Hyper;
      comp_labels = sign(comp_labels);
      
      
      nmis_list1(ii,j) = sum(comp_labels~=labels);
      error_rate = nmis_list1(ii,j)/length(labels)*100;
      
      if j==1
         error_rate_prev = error_rate;
      elseif (error_rate < error_rate_prev)
         % Capture properties
         preserved_variances{ii,j} = round(tmp*100,2);
         hyper_planes{ii,j} = Hyper;
         preserved_frequenciess{ii,j} = frequencies(ii,:);
         train_comp_labelss{ii,j} = comp_labels;
      end
      preserved_error(ii,j) = error_rate;
      
      error_rate_prev = error_rate;
      j = j+1;
      
      msg = ['Classified with an error rate of ', num2str(round(error_rate,2)), '% '];
      disp(msg)
      clear msg;
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   end
end

% g_thresh(gg) = gauss_thresh;
% e_rate(gg) = error_rate;
% 
% 
% 
% gg = gg + 1

%end

%figure(100)
%plot(g_thresh, e_rate)
% xlabel('Threshold')
% ylabel('Error Rate')

% Use the best results
[row, col] = find(preserved_error==min(preserved_error(:)));
row = row(1);col = col(1);

preserved_variance    = cell2mat(preserved_variances(row,col));
Hyper                 = cell2mat(hyper_planes(row,col));
preserved_frequencies = cell2mat(preserved_frequenciess(row,col));
train_comp_labels     = cell2mat(train_comp_labelss(row,col));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Misclassification Rate for Training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure();
list = {};
for i = 1:loop_length
   plot(c_list(i,:),nmis_list1(i,:)/length(labels)*100,'*-');hold on;
      if argwasspecified(1)
            msg = ['(Frequency ',strjoin(arrayfun(@(x) num2str(x),frequencies(i,:),...
                  'UniformOutput',false),','),')'];
      else
            msg = ['(Frequency ',strjoin(arrayfun(@(x) num2str(x),frequencies(i,1:i),...
                  'UniformOutput',false),','),')'];
      end
      list{end+1} = msg;
end
grid on;
legend(list)
%title({'Performance of Classification Using',
      %'SVM Followed by Projection (Training Set)'})
xlabel('$\textrm{\# of principal components }$', 'interpreter', 'latex')
ylabel('Misclassification rate (%)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Misclassification Rate for Training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('**********************************************************************')
msg = ['Using frequencies ',strjoin(arrayfun(@(x) num2str(x),preserved_frequencies,...
      'UniformOutput',false),','),' as our extracted features.'];
disp(msg)
disp('**********************************************************************')

% Calculate Welch Power Spectral Density Estimate for test data
for i=1:size(ecog_test,1)
   ecog_temp(:,:)=ecog_test(i,:,:);
   [pxx,f] = pwelch(ecog_temp',500,300,500,C_FS);
   ecog_test_psd(i,:,:)=20*log10(pxx');
end

%Plot the Classification Results for the test data
figure();
names = {'BrainActivation';'NoBrainActivation'};
plot(find(train_comp_labels==1), ones(numel(find(train_comp_labels==1))),...
     'b*', 'Marker','*'), box off, axis([0 100 -2 2])
hold on;
plot(find(train_comp_labels==-1), -1*ones(numel(find(train_comp_labels==-1))),...
     'r*', 'Marker','*'), box off, axis([0 size(ecog_psd,1) -2 2])
set(gca, 'ytick',[-1 1], 'yticklabel',names)
grid on;
title(['Classification of Training Data (Error Rate of ', num2str(round(preserved_error(row,col),2)),'%)']);
xlabel('Trial #');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test Set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract test features
ecog_test_features = reshape(ecog_test_psd(:,C_CHAN,preserved_frequencies),...
                            [size(ecog_test_psd,1) length(C_CHAN)*length(preserved_frequencies)]);

% Here we will apply PCA to reduce the dimensionality of the data while keeping
% most of the explained variance

% Find the -inf values using isnan
is_inf = isinf(ecog_test_features);

% Replace -inf with 0
ecog_test_features(is_inf) = 0;

centered = bsxfun(@minus, ecog_test_features, mean(ecog_test_features));


[U, S, V] = svd(centered);
[coeff,score,latent] = pca(centered);


% Reduce the dimensionality of the data while preserving preserved_variance
% percent of the variance.
tmp_s = cumsum(diag(S))/sum(diag(S));

idx = min(find(tmp_s > preserved_variance/100));
if preserved_variance==100
   idx = min(find(tmp_s == preserved_variance/100));
end
msg = ['Reducing dimensionality while preserving ',num2str(round(tmp_s(idx)*100,2)),'% of the variance.'];
disp(msg)
msg = ['Projecting onto the principal space of ', num2str(idx), ' dimensions.'];
disp(msg)
V = V(:,1:idx); % Reduce dimensionality

% Classify the test labels
A_test = (V*(V'*(ecog_test_features')))';
test_labels = A_test*Hyper;
test_labels = sign(test_labels);

% Plot the PCA 1st vs. 2nd score.
scores = U*S;

figure();
% Plot the first vs second score
subplot(2,2,1.5);

scatter(scores(:,1), scores(:,2));
% title(['Test PCA 1st vs. 2nd Score (',...
%       strjoin(arrayfun(@(x) num2str(x),preserved_frequencies,...
%       'UniformOutput',false),','),' Hz)']);
xlabel('1st component score');ylabel('2nd component score');
% Plot the Square-Rooted Eigenvalue Spectrum
subplot(2,1,2);
tmp = diag(S);
scatter(1:idx,tmp(1:idx), 'go');hold on;
scatter(idx+1:length(tmp),tmp(idx+1:end), 'ko');grid on;
% title(['Square-Rooted Eigenvalue Spectrum (', num2str(round(tmp_s(idx)*100,2)),'% of variance)']);
xlabel('component number');ylabel('Square-Root Eigenvalue');
clear tmp;


final_error_rate = ((sum(test_labels~=raw_test_sol))/size(test_labels,1))*100;
msg = ['Final Error Misclassification = ', num2str(final_error_rate), '%'];
disp(msg)

%Plot the Classification Results for the test data
figure();
names = {'BrainActivation';'NoBrainActivation'};
plot(find(test_labels==1), ones(numel(find(test_labels==1))),...
     'b*', 'Marker','*'), box off, axis([0 100 -2 2])
hold on;
plot(find(test_labels==-1), -1*ones(numel(find(test_labels==-1))),...
     'r*', 'Marker','*'), box off, axis([0 size(ecog_test_psd,1) -2 2])
set(gca, 'ytick',[-1 1], 'yticklabel',names)
grid on;
title(['Classification of Test Data (Error Rate of ', num2str(final_error_rate), '%)']);
xlabel('Trial #');

msg = ['Ratio of NoBrainActivation/BrainActivation = ',...
      num2str(length(find(test_labels==1))/length(find(test_labels==-1)))];
disp(msg)


% g_thresh(gg) = gauss_thresh;
% e_rate(gg) = final_error_rate;



gg = gg + 1;
end
%end

function [xhat,classificationMargin] = dykstra_svm(A,s,iterN)
    s = reshape(s,length(s),1);
    xhat = dykstra_halfspace(diag(s)*A,ones(size(A,1),1),iterN);
    classificationMargin = 1/sqrt(xhat'*xhat);
end

function projection = dykstra_halfspace(A, b, iterN)
    [m, n] = size(A);
    v0 = zeros(n, 1);
    Z = zeros(n, m);
    count = 1;
    while(count < iterN) % stopping conditions here
        ii = mod(count, m);
        if (ii == 0)
            ii = m;
        end
        % Projecting on the halfspace {x: a'x <= b}
        v1 = halfspace_projection(A(ii, :).', b(ii), v0 - Z(:, ii));
        % Updating Z
        Z(:, ii) = v1 - (v0 - Z(:, ii));
        count = count + 1;
        v0 = v1;
    end
    projection = v0;
end

function projection = halfspace_projection(a, b, v)
% code for computing projection of v on the halfspace {x: a'x >= b}
    s = a.'*v - b;
    if s >= 0
        projection = v;
    else
        projection = v - ((s/(a.'*a))*a);
    end
end

function [sorted_dist, dist] = rate_features(ecog_psd,labels)
% Iterate through all frequencies in the Power Spectral Density plots.
% This function will find the label (one of two) assosiated to each PCA score, then
% an average location of both (two) point clouds is found. Then the distance between
% both point clouds is calculated. We want to find the largest distance between
% the two point clouds and use the cooresponding frequency as our extracted feature.
for i = 1:size(ecog_psd,3)
   ecog_features(:,:)=ecog_psd(:,:,i); % Extract frequency
   [U, S, V] = svd(ecog_features);
   scores = U*S; % Calculate the scores
   BrainActivation = scores(find(labels ==  1),:); % Find labels assosiated BrainActivation scores
   NoBrainActivation  = scores(find(labels == -1),:); % Find labels assosiated NoBrainActivation scores
   % Find x,y means of two point clouds
   meanx_t = mean(BrainActivation(:,1));meany_t = mean(BrainActivation(:,2));
   meanx_f = mean(NoBrainActivation(:,1));meany_f = mean(NoBrainActivation(:,2));
   % Find average distance between the mean of the two point clouds
   dist(i) = abs(norm([meanx_t meany_t] - [meanx_f meany_f]));
end

[val, idx] = max(dist);
sorted_dist = sort(dist,'descend');
end

function [sel_chan] = channel_selection(ecog_psd,labels, threshold)
   sum1 = mean(ecog_psd(:,:,6:11),3);
   sum_NoBrainActivation = mean(sum1(find(labels>0),:));
   sum_BrainActivation = mean(sum1(find(labels<0),:));
   figure();bar(1:32,abs(sum_NoBrainActivation), 'r'); hold on;bar(1:32,abs(sum_BrainActivation), 'b')
   legend('Average NoBrainActivation Power', 'Average BrainActivation Power')
   title('Average Power Per Electrode (Frequency Band 6-12 Hz)')
   xlabel('ECoG Electrode'); ylabel('Magnitude of Average Power (dB)')
   figure();b = bar(1:32,abs([sum_NoBrainActivation(:), sum_BrainActivation(:)]), 'grouped');
   b(1).FaceColor = 'b';b(2).FaceColor = 'r';
   legend('Average NoBrainActivation Power', 'Average BrainActivation Power')
   title('Average Power Per Electrode')
   xlabel('ECoG Electrode'); ylabel('Magnitude of Average Power (dB)')
   set(gca, 'XTick', [1:32])
   
   sel_chan = find(abs(sum_NoBrainActivation-sum_BrainActivation)>threshold);
   sum_NoBrainActivation_ch = [sum_NoBrainActivation(sel_chan)];
   sum_BrainActivation_ch = [sum_BrainActivation(sel_chan)];
   figure();b = bar(sel_chan,abs([sum_NoBrainActivation_ch(:), sum_BrainActivation_ch(:)]), 'grouped');
   b(1).FaceColor = 'b';b(2).FaceColor = 'r';
   legend('Average NoBrainActivation Power', 'Average BrainActivation Power')
   title('Selected Frequency Channels');
   xlabel('ECoG Electrode'); ylabel('Magnitude of Average Power (dB)');
   
end

% This function will rate the best frequencies to use for the selected channels
% that you input.
function [sorted_freq] = rate_freq(ecog_psd, labels);

NoBrainActivation_power = zeros(1,size(ecog_psd,3));
BrainActivation_power = zeros(1,size(ecog_psd,3));
diff_power2=0;
for i = 1:size(ecog_psd,2) % Iterate through Channels
% for i = [4,9,10,11,12,14,18,21,22,23,24,29,30,31,32,37,38,39,40,45,46,47,48,52,53,54,55,56,59,60,61] % 0.3
% for i = [12,18,21,22,23,29,30,31,32,37,38,39,40,46,47,48,53,60] % 0.9
% for i = [12,18,21,22,23,29,30,31,37,38,39,40,46,47,48] % 1.0
% for i = [21,29,30,31,37,38,39,40,46] % 2.0

   for ii = 1:size(ecog_psd,1)  % Iterate through Trials
      if labels(ii)==1
         NoBrainActivation_power = NoBrainActivation_power + reshape(ecog_psd(ii, i, :),[1, size(ecog_psd,3)]);
      elseif labels(ii)==-1
         BrainActivation_power = BrainActivation_power + reshape(ecog_psd(ii, i, :),[1, size(ecog_psd,3)]);
      end
      diff_power2 = diff_power2+NoBrainActivation_power-BrainActivation_power;
   end

diff_power = abs(BrainActivation_power-NoBrainActivation_power);
end



end