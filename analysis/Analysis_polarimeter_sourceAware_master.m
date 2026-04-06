function analysis_polarimeter_sourceAware_master
    % Auto-detect source types and event tags from CSV filenames in dataset-1603
    % Logic for processing DP coherent and SP sources
    % Output summary CSVs and plots
    
    % Define time window
    time_window = [0, 60];
    
    % Get list of CSV files from dataset-1603
    csv_files = dir('dataset-1603/*.csv');
    
    for file = csv_files'
        [sourceType, eventTag, replicate] = parse_filename_tokens(file.name);
        % Process each source type accordingly...
        
        if startsWith(file.name, 'DPQAM16-200G') || startsWith(file.name, 'DPQPSK-200G')
            % DP coherent sources processing
            % Compute variation-based metrics
        elseif startsWith(file.name, 'SP-PURE') || startsWith(file.name, 'SP-AGIL') || startsWith(file.name, '10GE')
            % SP sources processing
            % Compute metrics
        end
    end
    
    % Write summary CSVs and save plots
end
