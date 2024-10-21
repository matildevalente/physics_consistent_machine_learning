function [rateCoeff, dependent] = modifiedArrheniusCharEnergyTemp(~, ~, ~, ~, ~, ~, ~, eTransProp, rateCoeffParams)
% modified Arrhenious char energy arrhenius evaluates a reaction rate coefficients using the following expression:

  charEnergy = eTransProp.reducedDiffCoeff/eTransProp.reducedMobility;
  a = rateCoeffParams{1};
  b = rateCoeffParams{2};
  c = rateCoeffParams{3};

  rateCoeff = a * charEnergy^b * exp(c/charEnergy);
  
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', true);

end
