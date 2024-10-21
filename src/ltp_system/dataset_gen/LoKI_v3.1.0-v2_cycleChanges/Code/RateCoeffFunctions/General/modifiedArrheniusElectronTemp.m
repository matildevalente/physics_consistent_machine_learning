function [rateCoeff, dependent] = modifiedArrheniusElectronTemp(~, ~, ~, ~, ~, ~, workCond, ~, rateCoeffParams)
% modifiedArrheniusElectronTemp evaluates a reaction rate coefficients using the following expression:

  Te = workCond.electronTemperature; % in eV
  a = rateCoeffParams{1};
  b = rateCoeffParams{2};
  c = rateCoeffParams{3};

  rateCoeff = a * Te^b * exp(c/Te);
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', true);

end