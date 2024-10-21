clear all
close all
clc

% input file with default configuration
inputSubFolder = 'Oxygen';
inputFile1 = 'oxygenlpp_chem_setup_novib_exp_comp.in';
inputFile2 = 'oxygenlpp3_noflow_4.chem';

% experimental values to compare with
R = 1e-2;                                     % discharge radii in m
targetI = 30e-3;   % target discharge current in A
%p =  7.5* 133.332;                                  % gas pressure in Pa
%p_torr = 7.5;
p =  [0.5 1 2 4 8].* 133.332;                                  % gas pressure in Pa
p_torr = [0.5 1 2 4 8];
% p =  [4 7.5].* 133.332;                                  % gas pressure in Pa
% p_torr = [4 7.5];
% p =  [1.15 2 3 4 5 6].* 133.332;                                  % gas pressure in Pa

% EN= 1.255 *p_torr.^2  -12.28*p_torr +84.42; %in K 20mA room 7.4 sccm
% EN= 1.49 *p_torr.^2  -13.35*p_torr +90.93;  %in K 50mA room 7.4 sccm
% EN=  1.176 *p_torr.^2  -10.54 * p_torr +   69.34; %in K 20mA dryice 7.4 sccm
% EN=  0.9795 *p_torr.^2  -9.32 * p_torr +  73.75;  %in K 50mA dryice 7.4 sccm
% EN =[60 60 60 60 60 60 60];

% Tg =  -5.777  *p_torr.^2 + 85.19 *p_torr + 321.3 ; %gas temperature in K 20mA room 7.4 sccm
%Tg = -9.948  *p_torr.^2 + 123.1 *p_torr + 377.6; %gas temperature in K 50mA room 7.4 sccm

% Tg =  -6.037  *p_torr.^2 + 74.86 *p_torr +  277.3; %gas temperature in K 20mA dryice 7.4 sccm
% Tg = -8.792  *p_torr.^2 + 112.1 *p_torr + 329.7 ; %gas temperature in K 50mA dryice 7.4 sccm
%Tg = 612.5
Tg= [347.3818285 375.884153 407.143803 452.360879 515.832917];
% Tg= [512.849 612.5];

%lossprob = 1.8e-3.*exp(-948./Tg);
lossprob = [0.002122431 0.001608795 0.001553127 0.001600322 0.002005327];
%lossprob3 = [6e-5 1.3e-4 5e-4 1e-3];
% lossprob = [8.73e-4 1.16e-3];
% lossprob3 = [5e-4 1e-3];
%lossprob = 1.16e-3;

% cycle for working conditions
for idx = 1:length(p) %5:5
  
  % initial values of electron density (m-3) and relative error in discharge current
  ne = 4e+15;
  neAll = ne;
  errorAll = [];
    
  while 1
    
    % write new setup file
    defaultFileID = fopen(['Input' filesep inputSubFolder filesep inputFile1], 'r');
    currentFileID = fopen(['Input' filesep 'oxygen_chem_setupbis.in'], 'wt');
    while ~feof(defaultFileID)
      line = fgetl(defaultFileID);
      if strncmp(line, '  electronDensity: ', 19)
        fprintf(currentFileID, '%s%.15e\n', line(1:19), ne);
      elseif strncmp(line, '  gasPressure: ', 15)
        fprintf(currentFileID, '%s%.6f\n', line(1:15), p(idx));
      elseif strncmp(line, '  gasTemperature: ', 18)
        fprintf(currentFileID, '%s%.6f\n', line(1:18), Tg(idx));
      elseif strncmp(line, '  chamberRadius: ', 17)
        fprintf(currentFileID, '%s%f\n', line(1:17), R);
%       elseif strncmp(line, '  reducedField: ', 16)
%         fprintf(currentFileID, '%s%f\n', line(1:16), EN(idx));
%       elseif strncmp(line, '      - CO(X,v=*) = boltzmannPopulation@', 40)
%         fprintf(currentFileID, '%s%f\n', line(1:36), TCO);
%       elseif strncmp(line, '      - CO2/CO2_vibpop_', 23)
%         fprintf(currentFileID, '%s%.2f%s\n', line(1:19), vibpop(idx),'.txt');
      else
        fprintf(currentFileID, '%s\n', line);
      end
    end
    fclose(defaultFileID);
    fclose(currentFileID);
    
    % write new chem file
    defaultFileID = fopen(['Input' filesep inputSubFolder filesep inputFile2], 'r');
    currentFileID = fopen(['Input' filesep 'Oxygen' filesep 'oxygen_O_loss_prob_change.chem'], 'wt');
    while ~feof(defaultFileID)
      line = fgetl(defaultFileID);
      if strncmp(line, 'O(3P) + wall -> 0.5O2(X)                    | multicomponentTransportChantry   | x                               |', 84)
        fprintf(currentFileID, 'O(3P) + wall -> 0.5O2(X)                    | multicomponentTransportChantry   |  %.16e | \n' , lossprob(idx));
      %elseif strncmp(line, 'O(3P) + wall -> 0.3333333333333333O3(X)                             | gasOnGasDiffOxygen        |', 84)
      %  fprintf(currentFileID, 'O(3P) + wall -> 0.3333333333333333O3(X)                             | gasOnGasDiffOxygen        | %.16e | \n' , lossprob3(idx));
      else
        fprintf(currentFileID, '%s\n', line);
      end
    end
    fclose(defaultFileID);
    fclose(currentFileID);
    
    % run loki
    loki('oxygen_chem_setupbis.in');
    
    % analise results
    resultsFileID = fopen(['Output' filesep 'outputO2' filesep 'swarmParameters.txt'], 'r');
    while ~feof(resultsFileID)
      line = fgetl(resultsFileID);
      if strncmp(line, '                 Drift velocity = ', 34)
        vd = str2double(line(35:54));
        break;
      end
    end
    fclose(resultsFileID);
    currentI = 1.6021766208e-19*ne*vd*pi*R*R;
    
    % evaluate new error
    errorAll(end+1) = (currentI-targetI)/targetI;
    
    % evaluate new electron density if needed exit loop otherwise
    if abs(errorAll(end)) > 1e-5
      if length(errorAll) == 1
        if errorAll(end) > 0
          ne = ne/2;
        else
          ne = ne*2;
        end
        neAll(end+1) = ne;
      elseif length(errorAll) == 2
        ne = interp1(errorAll, neAll, 0, 'linear', 'extrap');
        neAll(end+1) = ne;
      elseif length(errorAll) > 2
        ne = interp1(errorAll, neAll, 0, 'spline', 'extrap');
        neAll(end+1) = ne;
      end
    else
      break
    end
    
  end
  
  % move final output folder to a save location
  movefile(['Output' filesep 'outputO2'],['Output' filesep 'ISTref_O2bquenching_Herz_kutasiO2b_novib' sprintf('%.1f', p(idx)./133.332) 'Torr_' sprintf('%.1f', targetI*1000) 'mA']);

end
exit;