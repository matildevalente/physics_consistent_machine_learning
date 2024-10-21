function [rateCoeff, dependent] = einsteinCoeffFromOscStrength(~, ~, ~, reactionArray, reactionID, ~, ~, ~, ...
  rateCoeffParams)
% evaluates einstein coefficients for radiative transitions using the
% following expression

  a = Constant.vacuumPermeability*2*pi*Constant.electronCharge^4/(Constant.electronMass*...
      Constant.planck^2*Constant.speedOfLight);
  fji = rateCoeffParams{1};
  ui = reactionArray(reactionID).reactantArray.energy;
  gi = reactionArray(reactionID).reactantArray.statisticalWeight;
  uj = reactionArray(reactionID).productArray.energy;
  gj = reactionArray(reactionID).productArray.statisticalWeight;

  du_ij = ui - uj;

  rateCoeff = a * du_ij^2 * gj/gi * fji;
  
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', false);

end
