function[train_data, new_labels, number_trials, keep_BrainActivation_trials, keep_NoBrainActivation_trials, eliminate_NoBrainActivation, eliminate_BrainActivation] = trial_elimination(ecog_psd, labels, raw_train, thresh)
    [m,n,o] = size(raw_train);
    [mm,nn,oo] = size(ecog_psd);
    for ch = 1:n
        for j = 1:m
           for i = 1:oo
              temp_avg_ecog_psd(j,i) =  ecog_psd(j,ch,i);
           end
           avg_ecog_psd(j,ch) = mean(temp_avg_ecog_psd(j,:));
        end
    end
    % Split NoBrainActivation and BrainActivation Trials 
    row1 = 1;
    row2 = 1;
    for i = 1:m  
        if labels(i,1) == 1
            avg_NoBrainActivation_psd(row1,:) = avg_ecog_psd(i,:);
            NoBrainActivation_trialnum(row1) = i;
            row1 = row1 + 1;
        end
        if labels(i,1) == -1
            avg_BrainActivation_psd(row2,:) = avg_ecog_psd(i,:);
            BrainActivation_trialnum(row2) = i;
            row2 = row2 + 1;
        end    
    end
    % Plot NoBrainActivation gaussian and only keep trials within one standard deviation
    % of the mean
    figure()
    x = avg_NoBrainActivation_psd(:,1);
    stdev_NoBrainActivation = std(avg_NoBrainActivation_psd);
    x1 = x + stdev_NoBrainActivation(1,1);
    x2 = x - stdev_NoBrainActivation(1,1);
    x1 = x + thresh;
    x2 = x - thresh;

    % Find the -inf values using isnan
    is_inf = isinf(x);

    % Replace -inf with 0
    x(is_inf) = 0;



    pd = fitdist(x,'Normal');
    y = normpdf(x,pd.mu,pd.sigma);
    scatter(x,y)
    title('Normal Distribution of NoBrainActivation Trials in Classifying Set')
    xlabel('Average Power (dB/Hz)');ylabel('Probability');
    for ch = 1:n
      j = 1;
      k = 1;
      x = avg_NoBrainActivation_psd(:,ch);

      % Find the -inf values using isnan
      is_inf = isinf(x);

      % Replace -inf with 0
      x(is_inf) = 0;

      for trial = 1:row1-1
      pd = fitdist(x,'Normal');
      x1 = pd.mu + thresh;
      x2 = pd.mu - thresh;
          if (abs(avg_NoBrainActivation_psd(trial,ch)) < abs(x2)) && (abs(avg_NoBrainActivation_psd(trial,ch)) > abs(x1))
              eliminate_NoBrainActivation(trial,ch) = NoBrainActivation_trialnum(trial);
              j = j + 1;
          else 
              eliminate_NoBrainActivation(trial,ch) = -1;
              k = k + 1;
          end
      end
    end
    keep_cnt = zeros(row1-1,1);
    for trial = 1:row1-1
        for ch = 1:n
            if eliminate_NoBrainActivation(trial,ch) == -1
            else
                keep_cnt(trial) = keep_cnt(trial) + 1;
            end      
        end
    end
    i = 1;
    for trial = 1:row1-1
        if keep_cnt(trial) > 0
            keep_NoBrainActivation_trials(i) = NoBrainActivation_trialnum(trial);
            i = i + 1;
        end 
    end     
    figure()
    x = avg_BrainActivation_psd(:,1);
    stdev_BrainActivation = std(avg_BrainActivation_psd);
    x1 = x + stdev_BrainActivation(1,1);
    x2 = x - stdev_BrainActivation(1,1);
    x1 = x + thresh;
    x2 = x - thresh;
    % Find the -inf values using isnan
    is_inf = isinf(x);

    % Replace -inf with 0
    x(is_inf) = 0;
    pd = fitdist(x,'Normal');
    y = normpdf(x,pd.mu,pd.sigma);
    scatter(x,y)
    title('Normal Distribution of BrainActivation Trials in Classifying Set')
    xlabel('Average Power (dB/Hz)');ylabel('Probability');
    for ch = 1:n
      j = 1;
      k = 1;
      x = avg_BrainActivation_psd(:,ch);
      % Find the -inf values using isnan
      is_inf = isinf(x);

      % Replace -inf with 0
      x(is_inf) = 0;
      for trial = 1:row2-1
      pd = fitdist(x,'Normal');
      x1 = pd.mu + thresh;
      x2 = pd.mu - thresh;
          if (abs(avg_BrainActivation_psd(trial,ch)) < abs(x2)) && (abs(avg_BrainActivation_psd(trial,ch)) > abs(x1))
              eliminate_BrainActivation(trial,ch) = BrainActivation_trialnum(trial);
              j = j + 1;
          else 
              eliminate_BrainActivation(trial,ch) = -1;
              k = k + 1;
          end
      end
    end
    keep_cnt2 = zeros(row2-1,1);
    for trial = 1:row2-1
        for ch = 1:n
            if eliminate_BrainActivation(trial,ch) == -1
            else
                keep_cnt2(trial) = keep_cnt2(trial) + 1;
            end      
        end
    end
    i = 1;
    for trial = 1:row2-1
        if keep_cnt2(trial,1) > 0
            keep_BrainActivation_trials(i) = BrainActivation_trialnum(1,trial);
            i = i + 1;
        end 
    end 
   if max(keep_cnt2 > 0)  
        % Output only the wanted trails
        [m1,n1] = size(keep_BrainActivation_trials);

        for i = 1:n1
           temp = keep_BrainActivation_trials(i);
           train_data(i,:,:) =  raw_train(temp,:,:);
           new_labels(i,1) =  labels(temp);
        end
   else 
       n1 = 0;
   end   
   if max(keep_cnt > 0) 
        [mm1,nn1] = size(keep_NoBrainActivation_trials);

        for i = 1:nn1
           temp = keep_NoBrainActivation_trials(1,i);
           train_data(i+n1,:,:) =  raw_train(temp,:,:);
           new_labels(i+n1,1) =  labels(temp);
        end 
   else 
       nn1 = 0;
   end   
   if max(keep_cnt > 0) || max(keep_cnt2 > 0)
       number_trials = n1 + nn1;
   else 
       train_data =  raw_train;
       new_labels =  labels;
       number_trials = m;
   end 

end
 


  
