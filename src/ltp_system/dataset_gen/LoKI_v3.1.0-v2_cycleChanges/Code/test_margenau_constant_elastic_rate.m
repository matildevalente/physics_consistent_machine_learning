% --- DATA FOR THE TESTS ---
k0 = 1e-12;                   % elastic momentum transfer constant rate coefficient
Tg = 300;                     % gas temperature to be considered in the calculations
Muam = 40;                    % mass of the heavy species to be considered in the calculations (in UAM)
decades = 15;                 % decades of decay considered for the eedf
cellNumber = 50;              % number of energy cells to be considered in the calculations
reducedField = (0:1:50);      % range of reduced electric fields to be considered in the calculations

% --- LOCAL COPY OF CONSTANTS USED IN THE TESTS ---
kb = Constant.boltzmann;
me = Constant.electronMass;
eCharge = Constant.electronCharge;
uam = Constant.unifiedAtomicMass;

% --- CREATE OUTPUT FOLDER STRUCTURE ---
mkdir('Output', 'testMargenauConstantElasticRate');
mkdir(['Output' filesep 'testMargenauConstantElasticRate'], ['cellNumber_' sprintf('%d', cellNumber)]);

% --- THEORETICAL CALCULATIONS ---
M = Muam*uam;
Te = kb*Tg/eCharge+M/(3*me^2*eCharge)*(eCharge*reducedField*1e-21/k0).^2;
TeSimulation = zeros(size(Te));

for i = 1:length(reducedField)
  
  % --- EVALUATE ENERGY GRID OF THE SIMULATION ---
  maxEnergy = str2double(sprintf('%.2e', (decades/log10(exp(1)))*Te(i)));
  energy = linspace(0, maxEnergy, cellNumber+1);  
  energy(cellNumber+2) = 2*energy(end)-energy(end-1);
  
  % --- WRITE CROSS SECTION FILE ---
  fid = fopen('Input/ModelGasConstantElasticCollisionRate/elastic_cs_LXCat.txt', 'w');
  fprintf(fid, 'SPECIES: e / A\n');
  fprintf(fid, 'PROCESS: E + A -> E + A, Elastic\n');
  fprintf(fid, 'PARAM.:  m/M = 0.0000195921, complete set\n');
  fprintf(fid, 'COMMENT: [e + A(gnd) -> e + A(gnd), Elastic]\n');
  fprintf(fid, 'UPDATED: 2018-06-05 11:00:00\n');
  fprintf(fid, 'COLUMNS: Energy (eV) | Cross section (m2)\n');
  fprintf(fid, '-----------------------------\n');
  % write first point to avoid divergence of the analytical cross section
  fprintf(fid, '%.8e %.15e\n', 0.0, 1e-12*sqrt(9.10938215e-31/(2*1.6021766208e-19*(energy(2)-energy(1))))*(2*sqrt(2)-1));
  % write value of the cross section at the same energy points that are going to be used in the calculations
  for j = 2:cellNumber+2
    fprintf(fid, '%.8e %.15e\n', energy(j), 1e-12*sqrt(9.10938215e-31/(2*1.6021766208e-19*energy(j))));
  end
  fprintf(fid, '-----------------------------\n');
  fclose(fid);
  
  % --- WRITE SETUP FILE ---
  outputFolderStr = sprintf('testMargenauConstantElasticRate%scellNumber_%d%sreducedField_%f', filesep, cellNumber, ...
    filesep, reducedField(i));
  fid = fopen('Input/ModelGasConstantElasticCollisionRate/setup.in', 'w');
  fprintf(fid, 'workingConditions:\n');
  fprintf(fid, '  gasPressure: 133\n');
  fprintf(fid, '  gasTemperature: %f\n', Tg);
  fprintf(fid, '  electronDensity: 0\n');
  fprintf(fid, '  electronTemperature: 0\n');
  fprintf(fid, '  chamberLength: 0\n');
  fprintf(fid, '  chamberRadius: 0\n');
  fprintf(fid, '  reducedField: %f\n', reducedField(i));
  fprintf(fid, '  excitationFrequency: 0\n');
  fprintf(fid, 'electronKinetics:\n');
  fprintf(fid, '  isOn: true\n');
  fprintf(fid, '  eedfType: boltzmann\n');
  fprintf(fid, '  LXCatFiles:\n');
  fprintf(fid, '    - ModelGasConstantElasticCollisionRate/elastic_cs_LXCat.txt\n');
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
  fprintf(fid, '    maxEnergy: %f\n', maxEnergy);
  fprintf(fid, '    cellNumber: %d\n', cellNumber);
  fprintf(fid, 'chemistry:\n');
  fprintf(fid, '  isOn: false\n');
  fprintf(fid, 'gui:\n');
  fprintf(fid, '  isOn: false\n');
  fprintf(fid, '  refreshFrequency: 1\n');
  fprintf(fid, 'output:\n');
  fprintf(fid, '  isOn: true\n');
  fprintf(fid, '  folder: %s\n', outputFolderStr);
  fprintf(fid, '  dataFiles:\n');
  fprintf(fid, '    - eedf\n');
  fprintf(fid, '    - swarmParameters\n');
  fprintf(fid, '    - powerBalance\n');
  fclose(fid);
  
  % --- CALL LOKI TOOL ---
  loki('ModelGasConstantElasticCollisionRate/setup.in');
  
  % --- READ SIMULATION RESULTS ---
  fid = fopen(['Output' filesep outputFolderStr filesep 'swarmParameters.txt'], 'r');
  fgetl(fid);
  fgetl(fid);
  fgetl(fid);
  fgetl(fid);
  fgetl(fid);
  fgetl(fid);
  fgetl(fid);
  info = strsplit(fgetl(fid));
  TeSimulation(i) = str2double(info{5});
  fclose(fid);
  
  fid = fopen(['Output' filesep outputFolderStr filesep 'eedf.txt'], 'r');
  fgetl(fid);
  data = fscanf(fid,'%e',[3 inf]);
  theoreticalEEDF = exp(-data(1,:)/Te(i))/sum(sqrt(data(1,:)).*exp(-data(1,:)/Te(i)))/(data(1,2)-data(1,1));
  relErrorEEDF(i) = sum(abs(theoreticalEEDF-data(2,:))./theoreticalEEDF)/cellNumber;
  fclose(fid);
  
end

close all
fid = fopen(['Output' filesep 'testMargenauConstantElasticRate' filesep 'cellNumber_' sprintf('%d', cellNumber) ...
  filesep 'summary.txt'], 'w');
fprintf(fid, '################################\n');
fprintf(fid, '# DATA CONSIDERED FOR THE TESTS#\n');
fprintf(fid, '################################\n\n');
fprintf(fid, '# elastic momentum transfer constant rate coefficient = %e (m3s-1)\n', k0);
fprintf(fid, '#                                     gas temperature = %f (K)\n', Tg);
fprintf(fid, '#                           mass of the heavy species = %f (uam)\n', Muam);
fprintf(fid, '#               decades of decay ensured for the eedf = %d \n', decades);
fprintf(fid, '#                 number of cells for the energy grid = %d \n\n', cellNumber);
fprintf(fid, 'E/N(Td)              Te_th(eV)            Te_sim(eV)           RelError(Te)         RelError(EEDF)\n');
values(5:5:5*length(Te)) = relErrorEEDF;
values(4:5:5*length(Te)) = abs(Te-TeSimulation)./Te;
values(3:5:5*length(Te)) = TeSimulation;
values(2:5:5*length(Te)) = Te;
values(1:5:5*length(Te)) = reducedField;
fprintf(fid, '%#.14e %#.14e %#.14e %#.14e %#.14e \n', values);
fclose(fid);
plot(reducedField, (Te-TeSimulation)./Te, reducedField, relErrorEEDF);