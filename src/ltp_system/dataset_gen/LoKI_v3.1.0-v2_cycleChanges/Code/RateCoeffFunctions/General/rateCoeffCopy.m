function [rateCoeff, dependent] = rateCoeffCopy(~, ~, ~, reactionArray, ~, ~, ~, ~, ...
    rateCoeffParams)
% rateCoeffCopy evaluates de rate coefficient by setting it equal to another
% reaction's rate coefficient accordingly with the reaction's description given
% as input argument

  flagLoopBreak = 0;

  %create react descrip from arguments given
  %e.g. when e + O2(X,v=0) -> 2e + O2(+,X) is given as argument
  %the loop concatenates 'e + O2(X' + ',' + 'v=0) -> 2e + O2(+' + ',' + 'X)'
  for i=1:size(rateCoeffParams,2)
    if i ~= 1
      reactDescrip = strcat(reactDescrip, ',', rateCoeffParams{i});
    else
      reactDescrip = rateCoeffParams{1};
    end
  end

  for ID=1:length(reactionArray)
    %looking for reactionID that matches reaction descrip. given as arg.
    if strcmp(reactionArray(ID).description, reactDescrip)
      %getting the inelastic rateCoeff of the reaction ID
      rateCoeff = reactionArray(ID).eedfEquivalent.ineRateCoeff;
      flagLoopBreak = 1;
      break;
    end
  end

  if flagLoopBreak == 0
    error('Reaction description given as input argument was not found');
  end

  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', false);
end
