% LoKI-B solves a time and space independent form of the two-term 
% electron Boltzmann equation (EBE), for non-magnetised non-equilibrium 
% low-temperature plasmas excited by DC/HF electric fields from 
% different gases or gas mixtures.
% Copyright (C) 2018 A. Tejero-del-Caz, V. Guerra, D. Goncalves, 
% M. Lino da Silva, L. Marques, N. Pinhao, C. D. Pintassilgo and
% L. L. Alves
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.

classdef Reaction < handle
  %Reaction Class that stores the information of a certain chemical reaction
  %   Class that stores the information of a chemical reaction read from
  %   a ".chem" file. The information stored here (in particular the rate
  %   coefficient) is to be used in a chemistry solver to obtain the evolution
  %   heavy species densities.

  properties

    ID = -1;    % ID that identifies the reaction in the reactions array 
    
		type = '';            % string with the type of rate coefficient as defined in the ".chem" file
    rateCoeffFuncHandle;  % function handle to the function that evaluates the rate coefficient of the reaction

    reactantElectrons = 0;              % number of electrons in the left hand side of the reaction
    productElectrons = 0;               % number of electrons in the right hand side of the reaction
    reactantArray = ChemState.empty;    % handle to the reactants of the reaction
    reactantStoiCoeff = [];             % array of stoichiometric coefficients for the reactants of the reaction 
    catalystArray = ChemState.empty;    % handle to the catalysts of the reaction
    catalystStoiCoeff = [];             % array of stoichiometric coefficients for the catalysts of the reaction 
    productArray = ChemState.empty;     % handle to the products of the reaction
    productStoiCoeff = [];              % array of stoichiometric coefficients for the products of the reaction 
    
    isTimeDependent = false;      % true when the rate coefficient is implicit or explicitly time dependent
    
    isReverse = false;        % true when super elastic collision is defined
    isTransport = false;      % true when the reaction is a transport reaction (volume species + surface species -> ...)
    isGasStabilised = false;  % true when the reaction is stabilised by a collision with a gas species (any)
    
    rateCoeffParams = {};   % parameters needed to evaluate the reaction rate coefficient
    forwRateCoeff = [];      % rate coefficient of the direct reaction (->)
    backRateCoeff = [];      % rate coefficient of the inverse reaction (<-)
    
    enthalpy = 0.0;         % reaction enthalpy needed to solve the thermal balance equation (in case activated)
    
    eedfEquivalent = Collision.empty;
    
  end

  methods (Access = public)

    function reaction = Reaction(type, reactantElectrons, productElectrons, reactantArray, reactantStoiCoeff, ...
        catalystArray, catalystStoiCoeff, productArray, productStoiCoeff, isReverse, isTransport, isGasStabilised, ...
        rateCoeffParams, enthalpy)

      persistent lastID;
      if isempty(lastID)
        lastID = 0;
      end
      
      lastID = lastID + 1;
      reaction.ID = lastID;
      reaction.type = type;
      reaction.rateCoeffFuncHandle = str2func(type);
      reaction.reactantElectrons = reactantElectrons;
      reaction.productElectrons = productElectrons;
      reaction.reactantArray = reactantArray;
      reaction.reactantStoiCoeff = reactantStoiCoeff;
      reaction.catalystArray = catalystArray;
      reaction.catalystStoiCoeff = catalystStoiCoeff;
      reaction.productArray = productArray;
      reaction.productStoiCoeff = productStoiCoeff;
      reaction.isReverse = isReverse;
      reaction.isTransport = isTransport;
      reaction.isGasStabilised = isGasStabilised;
      reaction.rateCoeffParams = rateCoeffParams;
      if ~isempty(enthalpy)
        reaction.enthalpy = enthalpy;
      end
      
      for reactant = reactantArray
        if ~isempty(reactant.childArray)
          error(['Error when creating reaction: ''%s''.\nState %s has an inner distribution, hence I don''t know' ...
            'how to destroy it.\nPlease rewrite the reaction in terms of %s''s ''childs'''], reaction.description, ...
            reactant.name, reactant.name);
        end
        reactant.reactionsDestruction(end+1) = reaction;
      end
      for product = productArray
        if ~isempty(product.childArray)
          error(['Error when creating reaction: ''%s''.\nState %s has an inner distribution, hence I don''t know' ...
            'how to create it.\nPlease rewrite the reaction in terms of %s''s ''childs'''], reaction.description, ...
            product.name, product.name);
        end
        product.reactionsCreation(end+1) = reaction;
      end

    end
    
    function disp(reaction)
      
      fprintf('ID: %d\n', reaction.ID);
      fprintf('description: %s\n', reaction.description);
      paramStr = '';
      for i = 1:length(reaction.rateCoeffParams)
        if isnumeric(reaction.rateCoeffParams{i})
          paramStr = sprintf('%s%g, ', paramStr, reaction.rateCoeffParams{i});
        else
          paramStr = sprintf('%s%s, ', paramStr, reaction.rateCoeffParams{i});
        end
      end
      fprintf('parameters: %s\n', paramStr(1:end-2));
      
    end
    
    function reactionStr = description(reaction)
      
      if reaction.reactantElectrons == 0
        reactionStr = '';
      elseif reaction.reactantElectrons == 1
        reactionStr = 'e + ';
      else
        reactionStr = sprintf('%de + ', reaction.reactantElectrons);
      end
      for i = 1:length(reaction.reactantArray)
        if reaction.reactantStoiCoeff(i) == 1
          reactionStr = sprintf('%s%s + ', reactionStr, reaction.reactantArray(i).name);
        else
          reactionStr = sprintf('%s%d%s + ', reactionStr, reaction.reactantStoiCoeff(i), reaction.reactantArray(i).name);
        end
      end
      catalystStr = '';
      for i = 1:length(reaction.catalystArray)
        if reaction.catalystStoiCoeff(i) == 1
          catalystStr = sprintf('%s%s + ', catalystStr, reaction.catalystArray(i).name);
        else
          catalystStr = sprintf('%s%d%s + ', catalystStr, reaction.catalystStoiCoeff(i), reaction.reactantArray(i).name);
        end
      end
      reactionStr = sprintf('%s%s', reactionStr, catalystStr);
      if reaction.isGasStabilised
        reactionStr = sprintf('%sGas + ', reactionStr);
      end
      if reaction.isTransport && length(reaction.reactantArray)==1
        reactionStr = sprintf('%sWall + ', reactionStr);
      end
      if reaction.isReverse
        reactionStr = sprintf('%s<-> ', reactionStr(1:end-2));
      else
        reactionStr = sprintf('%s-> ', reactionStr(1:end-2));
      end
      if reaction.productElectrons == 1
        reactionStr = sprintf('%se + ', reactionStr);
      elseif reaction.productElectrons > 1
        reactionStr = sprintf('%s%de + ', reactionStr, reaction.productElectrons);
      end
      for i = 1:length(reaction.productArray)
        if reaction.productStoiCoeff(i) == 1
          reactionStr = sprintf('%s%s + ', reactionStr, reaction.productArray(i).name);
        else
          reactionStr = sprintf('%s%d%s + ', reactionStr, reaction.productStoiCoeff(i), reaction.productArray(i).name);
        end
      end
      reactionStr = sprintf('%s%s', reactionStr, catalystStr);
      if reaction.isGasStabilised
        reactionStr = sprintf('%sGas + ', reactionStr);
      end
      reactionStr = reactionStr(1:end-3);
      
    end
    
    function reactionStr = descriptionExtended(reaction)
      
      reactionStr = sprintf('%s, %s', reaction.description, reaction.type);
      
    end
    
    function linkWithElectronKineticsCollision(reaction, electronKineticsCollision)
      
      % save handle to the equivalent electron kinetic gas
      reaction.eedfEquivalent = electronKineticsCollision;
      electronKineticsCollision.chemEquivalent = reaction;
      
    end
    
    function evaluateEnthalpy(reaction)
    % evaluateEnthalpy evaluates the reaction enthalpy as de energy difference of the involved states
      
      % evaluate energy difference between reactant and product states
      enthalpyAux = 0;
      for i = 1:length(reaction.reactantArray)
        state = reaction.reactantArray(i);
        if ~isempty(state.energy)
          enthalpyAux = enthalpyAux + reaction.reactantStoiCoeff(i)*state.energy;
        else
          error(['Energy of state %s not found when evaluating the enthalpy of reaction:\n%s\nas states energy ' ...
            'difference.\nPlease, fix the problem and run the code again.'], state.name, reaction.description);
        end
      end
      for i = 1:length(reaction.productArray)
        state = reaction.productArray(i);
        if ~isempty(state.energy)
          enthalpyAux = enthalpyAux - reaction.productStoiCoeff(i)*state.energy;
        else
          error(['Energy of state %s not found when evaluating the enthalpy of reaction:\n%s\nas states energy ' ...
            'difference.\nPlease, fix the problem and run the code again.'], state.name, reaction.description);
        end
      end
      % save enthalpy value in reaction properties
      reaction.enthalpy = enthalpyAux;
      
    end
    
  end

end
