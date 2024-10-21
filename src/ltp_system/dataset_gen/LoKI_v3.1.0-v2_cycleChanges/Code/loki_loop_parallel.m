% ... (previous code)

% Open the file for reading
fid = fopen('loop_config.txt', 'r');

% Read the integer n_simulations from the first line
n_simulations = fscanf(fid, '%d', 1);

% Read the string outputfolder from the second line
folder = fscanf(fid, '%s', 1);
fclose(fid);

% Define number of workers
numWorkers = 8;
if numWorkers >= feature('numcores')
    error('Num of workers is higher than num of Cores');
end

% Create a parallel pool with the specified number of workers
pool = parpool(numWorkers);

tic; % start timer

parfor i = 0:(n_simulations-1) % parallel for loop
    address_string = strcat(folder, num2str(i), '.in');
    disp(address_string);

    try
        loki(address_string, i);
    catch ME
        % Handle the error (display a message or log it)
        fprintf('Error in iteration %d: %s\n', i, ME.message);
        continue; % Skip to the next iteration
    end
end

disp(toc); % Display execution time
delete(pool);
