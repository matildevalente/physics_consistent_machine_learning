function [rateCoeff, dependent] = scaledEedf(~, ~, ~, reactionArray, reactionID, ~, ~, ~, ...
    rateCoeffParams)
% scaledEedf scales the rate coefficient of an eedf reaction

  persistent equivalentCollision;
  currentReaction = reactionArray(reactionID);
  
  if isempty(equivalentCollision)
    equivalentCollision = cell((length(reactionArray)));
  end  

  if isempty(equivalentCollision{reactionID})
    % reconstruct the collision description
    % e.g. when e + O2(X,v=0) -> e + e + O2(+,X) is given as argument
    % the loop concatenates 'e + O2(X' + ',' + 'v=0) -> e + e + O2(+' + ',' + 'X)'
    collisionDescription = currentReaction.description;
    % remove spaces
    collisionDescription = collisionDescription(~isspace(collisionDescription));
    % find the equivalent collision in the target gas array
    targetGas = currentReaction.reactantArray(1).gas.eedfEquivalent;
    for collision = [targetGas.collisionArray targetGas.collisionArrayExtra]
      % get the candidate full description (collision,type)
      description = collision.description;
      % remove the type part
      commasPositions = strfind(description,',');
      description = description(1:(commasPositions(end)-1));
      % check if the candidate description coincides
      if strcmp(description, collisionDescription)
        equivalentCollision{reactionID} = collision;
        break;
      end  
    end  
    % check if the equivalent collision was found
    if isempty(equivalentCollision{reactionID})
      error('Could not find collision:\n%s\nNeeded to evaluate ''scaledEedf'' for reaction\n%s',...
        collisionDescription,currentReaction.description);
    elseif currentReaction.isReverse && ~equivalentCollision{reactionID}.isReverse
      error(['Error when using ''scaledEedf'' for reaction\n%s',...
        '\nThe collision to be copied has no reverse.'],currentReaction.description);
    end  
  end  

  equivColl = equivalentCollision{reactionID};
  scalingFactor = rateCoeffParams{1};
  % save the forward rate coeff
  rateCoeff = scalingFactor*equivColl.ineRateCoeff;
  % save the backwards rate coeff, if it has inverse
  if currentReaction.isReverse
    currentReaction.backRateCoeff = scalingFactor*equivColl.supRateCoeff;
  end  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', true);
end
