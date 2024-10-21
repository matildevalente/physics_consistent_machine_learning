clear all;
close all;

% save the script directory
scriptDirectory = pwd;
addpath(pwd);

% DEFINE THE EEDF TYPE: boltzmann, prescribedEedf or boltzmannMC
eedfType = 'boltzmann'; 

% DEFINE THE THERMAL MODEL
includeThermalModel = true;
boundary = 'wall';
wallFraction = 0.5;
intConvCoeff = 100;
extConvCoeff = 15;

% input file with default configuration
inputSubFolder = 'Oxygen';
inputRefFileIn = 'oxygen_chem_setup_novib.in';
inputRefFileChem = 'oxygen_novib.chem';
outputSubFolder = 'OxygenDCSimulations/novib_butWithVibHeat';

% experimental values 
chamberRadius = 1E-2;
inputExpData = readtable(['Input' '/' inputSubFolder '/' 'experimentalDataRussiansLPP_lossProbBooth2020.xlsx']);

%select current to be simulated
% indexesToSimulate = inputExpData.Current_mA_==30;
indexesToSimulate = true(size(inputExpData.Current_mA_));

% read experimental values
pressure_torr = inputExpData.Pressure_torr_(indexesToSimulate);
pressure = pressure_torr.*133.32;
current_mA = inputExpData.Current_mA_(indexesToSimulate);
current = current_mA.*1E-3;
Twall_deg = inputExpData.Twall(indexesToSimulate);
Twall = Twall_deg+273.15;
sccmFlow = inputExpData.Flow_sccm_(indexesToSimulate);
expTgas = inputExpData.Tgas_K_(indexesToSimulate);
expEN = inputExpData.E_N_Td_(indexesToSimulate);
gammaO = inputExpData.gamma_O(indexesToSimulate);

nSimulations = length(pressure_torr);
dataToSave = [];
for i=1:nSimulations
  
  % evaluate output folder
  outputFolder = [outputSubFolder '/' 'I' num2str(current_mA(i)) 'mA_p' num2str(pressure_torr(i)) 'torr'];
    
  % write new setup file
  refFileID = fopen(['Input' '/' inputSubFolder '/' inputRefFileIn], 'r');
  tempFileID = fopen(['Input' '/' inputSubFolder '/' 'temp.in'], 'w');
  while ~feof(refFileID)
    line = fgetl(refFileID);
    if contains(line,'gasPressure:')  
      fprintf(tempFileID, '  gasPressure: %.15e\n', pressure(i));
    elseif contains(line,'gasTemperature:')
      fprintf(tempFileID, '  gasTemperature: %.15e\n', expTgas(i));        
    elseif contains(line,'wallTemperature:')
      fprintf(tempFileID, '  wallTemperature: %.15e\n', Twall(i));
    elseif contains(line,'chamberRadius:')
      fprintf(tempFileID, '  chamberRadius: %.15e\n', chamberRadius);
    elseif contains(line,'reducedField:')
      fprintf(tempFileID, '  reducedField: %.15e\n', expEN(i));
    elseif contains(line,'dischargeCurrent:')
      fprintf(tempFileID, '  dischargeCurrent: %.15e\n', current(i));    
    elseif contains(line,'eedfType:')
      fprintf(tempFileID, '  eedfType: %s\n', eedfType); 
    elseif contains(line,'thermalModel:')
      fprintf(tempFileID, '%s\n', line);
      line = fgetl(refFileID);
      fprintf(tempFileID, '    isOn: %s\n',string(includeThermalModel));
    elseif contains(line,'boundary:')
      fprintf(tempFileID, '    boundary: %s\n',boundary);
    elseif contains(line,'wallFraction:')  
      fprintf(tempFileID, '    wallFraction: %.15e\n',wallFraction);
    elseif contains(line,'intConvCoeff:')  
      fprintf(tempFileID, '    intConvCoeff: %.15e/1.6021766208e-19\n',intConvCoeff);
    elseif contains(line,'extConvCoeff:')  
      fprintf(tempFileID, '    extConvCoeff: %.15e/1.6021766208e-19\n',extConvCoeff);    
    elseif contains(line,inputRefFileChem)
      fprintf(tempFileID, '    - Oxygen/temp.chem\n');  
    elseif contains(line,'folder:')
      fprintf(tempFileID, '  folder: %s\n', outputFolder);
    else
      fprintf(tempFileID, '%s\n', line);  
    end
  end
  fclose(refFileID);
  fclose(tempFileID);
  
  % write new chem file
  refFileID = fopen(['Input' '/' inputSubFolder '/' inputRefFileChem], 'r');
  tempFileID = fopen(['Input' '/' inputSubFolder '/' 'temp.chem'], 'w');
  while ~feof(refFileID)
    line = fgetl(refFileID);
    if contains(line, 'O(3P) + wall -> 0.5O2(X)')
      fprintf(tempFileID, 'O(3P) + wall -> 0.5O2(X)                    | multicomponentTransportChantry  |  %.16e, nearWallTemperature | 5.12*(1-0.5)\n' , gammaO(i));
    elseif contains(line, 'e -> O2(X)') && contains(line,'inFlow')                                   
      fprintf(tempFileID, 'e -> O2(X)					| inFlow	|%.16e*101325/1.38064852e-23*1e-6/60/273.15 ,1 | \n' , sccmFlow(i));  
    elseif contains(line, 'O(3P) + wall -> 0.5O2(X,v=0)')
      fprintf(tempFileID, 'O(3P) + wall -> 0.5O2(X,v=0)                    | multicomponentTransportChantry   |  %.16e, nearWallTemperature | 5.12*(1-0.5)\n' , gammaO(i));
    elseif contains(line, 'e -> O2(X,v=0)') && contains(line,'inFlow')                                   
      fprintf(tempFileID, 'e -> O2(X,v=0)					| inFlow	|%.16e*101325/1.38064852e-23*1e-6/60/273.15 ,1 | \n' , sccmFlow(i));          
    else
      fprintf(tempFileID, '%s\n', line);  
    end
  end
  fclose(refFileID);
  fclose(tempFileID);
  
  % run loki
  loki([inputSubFolder '/' 'temp.in']);

  % copy this chem file to the output, for later cross-checks
  copyfile('Input/Oxygen/temp.chem', ['Output' '/' outputFolder '/' 'temp.chem']);     
  
%     system(['wsl ./lokimc ' inputSubFolder '/' 'temp.in' ' 2']); 
  
  % calculate the new current
  swarmFileID = fopen(['Output' '/' outputFolder '/' 'swarmParameters.txt']);
  while ~feof(swarmFileID)
    line = fgetl(swarmFileID);
    if contains(line,'Drift velocity')
      driftVelocity = textscan(line,'                       Drift velocity = %f (m/s)',1);
      driftVelocity = driftVelocity{1};
    elseif contains(line, 'Reduced electric field')
      modelEN = textscan(line,'               Reduced electric field = %f (Td)');
      modelEN = modelEN{1};
    elseif contains(line, 'Electron temperature')
      electronTemperature = textscan(line,'                 Electron temperature = %f (eV)');
      electronTemperature = electronTemperature{1};
    end
  end
  fclose(swarmFileID);
  
  % read the final simulation results
  chemSolutionTime = readtable(['Output' '/' outputFolder '/' 'chemSolutionTime.txt']);
  
  electronDensity = chemSolutionTime.x_O2___X___m__3_(end) + chemSolutionTime.x_O___gnd___m__3_(end)-chemSolutionTime.x_O___gnd___m__3__1(end);

  speciesModifiedNames = chemSolutionTime.Properties.VariableNames;
  speciesModifiedNames = speciesModifiedNames(2:end);
  speciesNames = chemSolutionTime.Properties.VariableDescriptions;
  speciesNames = speciesNames(2:end);
  lineToSave = [pressure_torr(i) current_mA(i) Twall(i) sccmFlow(i) expTgas(i) expEN(i) gammaO(i) modelEN electronTemperature electronDensity];
  for name = speciesModifiedNames
    lineToSave(end+1) = chemSolutionTime.(name{1})(end);   
  end
  dataToSave = [dataToSave; lineToSave];
  headers = ["p(Torr)" "I(mA)" "Twall" "Flow(sccm)" "Tgas_exp(K)" "E/N_exp(Td)" "gammaO" "E/N(Td)" "Te(eV)" "n_e(m-3)" string(speciesNames)];     
  tableToSave = array2table(dataToSave,'VariableNames',headers);
  writetable(tableToSave,['Output' '/' outputSubFolder '/' 'summaryResults.xlsx']);
end