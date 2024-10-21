function [rateCoeff, dependent] = powerGasElecTemp(~, ~, ~, ~, ~, ~, workCond, ~, rateCoeffParams)
% arrhenius evaluates a reaction rate coefficients using the following expression:

  Tg = workCond.gasTemperature; % in K
  Te = workCond.electronTemperature; %in eV
  a = rateCoeffParams{1};
  b = rateCoeffParams{2};
  c = rateCoeffParams{3};

  rateCoeff = a * Tg^b * Te^c;
  
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', true, 'onElectronKinetics', true);

end
