function [rateCoeff, dependent] = pseudoEedfCopy(~, ~, ~, reactionArray, reactionID, ~, ~, ~, ...
    rateCoeffParams)
% pseudoEedfCopy evaluates the rate coefficient of a reaction by setting it equal to another
% collision's rate coefficient accordingly to the collision's description given
% as input argument

  persistent equivalentCollision;
  currentReaction = reactionArray(reactionID);
  
  if isempty(equivalentCollision)
    equivalentCollision = cell((length(reactionArray)));
  end  

  if isempty(equivalentCollision{reactionID})
    % reconstruct the collision description
    % e.g. when e + O2(X,v=0) -> e + e + O2(+,X) is given as argument
    % the loop concatenates 'e + O2(X' + ',' + 'v=0) -> e + e + O2(+' + ',' + 'X)'
    collisionDescription = rateCoeffParams{1};
    for i=2:length(rateCoeffParams)
      collisionDescription = [collisionDescription ',' rateCoeffParams{i}];
    end
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
      error('Could not find collision:\n%s\nNeeded to evaluate ''pseudoEedfCopy'' for reaction\n%s',...
        collisionDescription,currentReaction.description);
    elseif currentReaction.isReverse && ~equivalentCollision{reactionID}.isReverse
      error(['Error when using ''pseudoEedf_copy'' for reaction\n%s',...
        '\nThe collision to be copied has no reverse.'],currentReaction.description);
    end  
  end  

  equivColl = equivalentCollision{reactionID};
  % save the forward rate coeff
  rateCoeff = equivColl.ineRateCoeff;
  % save the backwards rate coeff, if it has inverse
  if currentReaction.isReverse
    currentReaction.backRateCoeff = equivColl.supRateCoeff;
  end  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', true);
end
