% Takes in subject number, creates EEG feature matrix

%function [] = extract_features(ts, bands, fs, tau, dE)
%    
%end

function [] = get_feature_matrix(subj)
    subj = char(subj);

    X_npy_file = strcat('/projectnb/cs542/ryanmars/data_downsampled_mat/s',subj,'_X.npy');
    y_npy_file = strcat('/projectnb/cs542/ryanmars/data_downsampled_mat/s',subj,'_y.npy');
    lst_npy_file = strcat('/projectnb/cs542/ryanmars/data_downsampled_mat/s',subj,'_lst.npy');
    
    X = readNPY(X_npy_file);
    y = readNPY(y_npy_file);
    %lst = readNPY(lst_npy_file); %Index exceeds matrix dimensions??
    
    %dlmwrite(strcat('/projectnb/cs542/ryanmars/data_downsampled_mat/s',subj,'_X.mat'),X);
    
    %bands = [0.5, 4, 7, 12, 30]; %Frequency bands for delta, theta, alpha, beta, gamma respectively
    %fs = 40; %Sampling Rate in Hz
    %tau = 10; %embed_seq lag (integer)
    %dE = 10; %embedding dimension (integer)
    
    %nfiles = size(X,1);
    %nchannels = size(X,3);
    %nfeatures = 8 + length(bands); 
    
    %feature_mat = zeros(nfiles, nfeatures * nchannels);
    
    %for i=1:nfiles
    %    for j=1:nchannels
    %        ts = X(i, :, j);
    %        feature_mat(i, (j*nfeatures+1):(j+1)*nfeatures) = extract_features(ts, bads, fs, tau, dE);        
    %    end
    %end
end