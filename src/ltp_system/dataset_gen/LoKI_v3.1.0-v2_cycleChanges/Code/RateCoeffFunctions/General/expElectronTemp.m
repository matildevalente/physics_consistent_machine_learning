function [rateCoeff, dependent] = expElectronTemp(~, ~, ~, ~, ~, ~, workCond, ~, rateCoeffParams)
% expElectronTemp evaluates a reaction rate coefficients using the following expression:

  Te = workCond.electronTemperature; % in eV
  a = rateCoeffParams{1};
  b = rateCoeffParams{2};

  rateCoeff = a * exp(Te/b);
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', true);

end