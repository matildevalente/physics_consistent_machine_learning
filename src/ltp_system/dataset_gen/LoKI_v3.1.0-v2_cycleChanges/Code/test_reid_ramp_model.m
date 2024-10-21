% --- DATA FOR THE TESTS ---
k0 = 6e-20;                               % constant elastic cross section (m^2)
k = 10e-20;                               % slope of inelastic cross section ramp (m^2eV-1)
ineThreshold = 0.2;                       % inelastic cross section threshold (eV)
Tg = 0;                                   % gas temperature to be considered in the calculations
Muam = 4;                                 % mass of the heavy species to be considered in the calculations (in UAM)
decades = 13;                             % decades of decay considered for the eedf
cellNumber = 1000;                        % number of energy cells to be considered in the calculations
reducedFieldStr = 'logspace(0,2,100)';   % range of reduced electric fields to be considered in the calculations

% --- LOCAL COPY OF CONSTANTS USED IN THE TESTS ---
kb = Constant.boltzmann;
me = Constant.electronMass;
eCharge = Constant.electronCharge;
uam = Constant.unifiedAtomicMass;

% --- CREATE OUTPUT FOLDER STRUCTURE ---
mkdir('Output', 'testRampModel');
mkdir(['Output' filesep 'testRampModel'], ['slope_' sprintf('%e', k)]);

% --- THEORETICAL CALCULATIONS ---
M = Muam*uam;

for i = 1:length(k)
  
  % --- WRITE CROSS SECTION FILE ---
  fid = fopen('Input/ModelGasRamp/cs_LXCat.txt', 'w');
  fprintf(fid, 'SPECIES: e / A\n');
  fprintf(fid, 'PROCESS: E + A -> E + A, Elastic\n');
  fprintf(fid, 'PARAM.:  m/M = %.14e, complete set\n', me/M);
  fprintf(fid, 'COMMENT: [e + A(gnd) -> e + A(gnd), Elastic]\n');
  fprintf(fid, 'UPDATED: 2018-06-05 11:00:00\n');
  fprintf(fid, 'COLUMNS: Energy (eV) | Cross section (m2)\n');
  fprintf(fid, '-----------------------------\n');
  fprintf(fid, '%f %.5e\n', 0.0, k0);
  fprintf(fid, '%f %.5e\n', 1000.0, k0);
  fprintf(fid, '-----------------------------\n\n');
  fprintf(fid, 'SPECIES: e / A\n');
  fprintf(fid, 'PROCESS: E + A -> E + A(exc), Excitation\n');
  fprintf(fid, 'PARAM.:  E = %f eV, complete set\n', ineThreshold);
  fprintf(fid, 'COMMENT: [e + A(gnd) -> e + A(exc), Excitation]\n');
  fprintf(fid, 'UPDATED: 2018-06-05 11:00:00\n');
  fprintf(fid, 'COLUMNS: Energy (eV) | Cross section (m2)\n');
  fprintf(fid, '-----------------------------\n');
  fprintf(fid, '%f %.5e\n', ineThreshold, 0.0);
  fprintf(fid, '%f %.5e\n', 1000.0, k(i)*(1000-ineThreshold));
  fprintf(fid, '-----------------------------\n');
  fclose(fid);
  
  % --- WRITE SETUP FILE ---
  outputFolderStr = sprintf('testRampModel%sslope_%e', filesep, k(i));
  fid = fopen('Input/ModelGasRamp/setup.in', 'w');
  fprintf(fid, 'workingConditions:\n');
  fprintf(fid, '  gasPressure: 133\n');
  fprintf(fid, '  gasTemperature: %f\n', Tg);
  fprintf(fid, '  electronDensity: 0\n');
  fprintf(fid, '  electronTemperature: 0\n');
  fprintf(fid, '  chamberLength: 0\n');
  fprintf(fid, '  chamberRadius: 0\n');
  fprintf(fid, '  reducedField: %s\n', reducedFieldStr);
  fprintf(fid, '  excitationFrequency: 0\n');
  fprintf(fid, 'electronKinetics:\n');
  fprintf(fid, '  isOn: true\n');
  fprintf(fid, '  eedfType: boltzmann\n');
  fprintf(fid, '  LXCatFiles:\n');
  fprintf(fid, '    - ModelGasRamp/cs_LXCat.txt\n');
  fprintf(fid, '  ionizationOperator:\n');
  fprintf(fid, '    mode: conservative\n');
  fprintf(fid, '  electronElectronCollisions: no\n');
  fprintf(fid, '  gasProperties:\n');
  fprintf(fid, '    mass:\n');
  fprintf(fid, '      - A = %f*1.660539040e-27\n', Muam);
  fprintf(fid, '    fraction:\n');
  fprintf(fid, '      - A = 1\n');
  fprintf(fid, '  stateProperties:\n');
  fprintf(fid, '    population:\n');
  fprintf(fid, '      - A(gnd) = 1.0\n');
  fprintf(fid, '  energyGrid:\n');
  fprintf(fid, '    maxEnergy: %f\n', 1);
  fprintf(fid, '    cellNumber: %d\n', cellNumber);
  fprintf(fid, '    smartGrid:\n');
  fprintf(fid, '      minEedfDecay: %d\n', decades);
  fprintf(fid, '      maxEedfDecay: %d\n', decades+1);
  fprintf(fid, '      updateFactor: 0.05\n');
  fprintf(fid, 'chemistry:\n');
  fprintf(fid, '  isOn: false\n');
  fprintf(fid, 'gui:\n');
  fprintf(fid, '  isOn: true\n');
  fprintf(fid, '  refreshFrequency: 1\n');
  fprintf(fid, 'output:\n');
  fprintf(fid, '  isOn: true\n');
  fprintf(fid, '  folder: %s\n', outputFolderStr);
  fprintf(fid, '  dataFiles:\n');
  fprintf(fid, '    - eedf\n');
  fprintf(fid, '    - swarmParameters\n');
  fprintf(fid, '    - powerBalance\n');
  fprintf(fid, '    - lookUpTable\n');
  fclose(fid);
  
  % --- CALL LOKI TOOL ---
  loki('ModelGasRamp/setup.in');
  
%   fid = fopen(['Output' filesep outputFolderStr sprintf('%e', k) filesep 'data.txt'], 'w');
%   fprintf(fid, '################################\n');
%   fprintf(fid, '# DATA CONSIDERED FOR THE TESTS#\n');
%   fprintf(fid, '################################\n\n');
%   fprintf(fid, '# elastic momentum transfer constant rate coefficient = %e (m3s-1)\n', k0);
%   fprintf(fid, '#                                     gas temperature = %f (K)\n', Tg);
%   fprintf(fid, '#                           mass of the heavy species = %f (uam)\n', Muam);
%   fprintf(fid, '#               decades of decay ensured for the eedf = %d \n', decades);
%   fprintf(fid, '#                 number of cells for the energy grid = %d \n\n', cellNumber);
%   fprintf(fid, 'E/N(Td)              Te_th(eV)            Te_sim(eV)           RelError\n');
%   values(4:4:4*length(Te)) = abs(Te-TeSimulation)./Te;
%   values(3:4:4*length(Te)) = TeSimulation;
%   values(2:4:4*length(Te)) = Te;
%   values(1:4:4*length(Te)) = reducedFieldStr;
%   fprintf(fid, '%#.14e %#.14e %#.14e %#.14e \n', values);
%   fclose(fid);
  
end