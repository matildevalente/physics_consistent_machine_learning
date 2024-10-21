function [rateCoeff, dependent] = orbitalExchange(~, ~, ~, reactionArray, reactionID, ~, workCond, ~, rateCoeffParams)
  % orbital Exchange evalutes excitation transfer reactions with quantum
  % orbital number changes M.Santos et al, J. Phys. D: Appl. Phys, 47,  
  % 265201, 2014
  
  Tg = workCond.gasTemperature*Constant.boltzmannInEV; % in eV  
  
  ui = reactionArray(reactionID).reactantArray.energy;
  gi = reactionArray(reactionID).reactantArray.statisticalWeight;
  uj = reactionArray(reactionID).productArray.energy;
  duij = ui-uj;
  
  a = rateCoeffParams{1};
  b = rateCoeffParams{2};
  
  if duij < Tg
   rateCoeff = a * (duij/Tg)^(b) / gi;
  else
   c = rateCoeffParams{3};
   rateCoeff = a * (duij/Tg)^(b) * exp(-duij*c/Tg)/gi;
  end
 
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', true, 'onElectronKinetics', false);

end
