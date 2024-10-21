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

classdef ChemGas < Gas
  
  properties
    
    isVolumeSpecies;                  % boolean variable indicating if the gas (and its states) is a volume specie
    isSurfaceSpecies;                 % boolean variable indicating if the gas (and its states) is a surface specie
    
    reactionArray = Reaction.empty;   % array of handles to the reactions in wich this gas is involved
    
    eedfEquivalent = EedfGas.empty;   % handle to the equivalent EedfGas (in case it exists) 
    
  end
  
  events
    
  end
  
  methods
    
    function gas = ChemGas(gasName)
      
      persistent lastID;
      if isempty(lastID)
        lastID = 0;
      end
      lastID = lastID + 1;
      gas.ID = lastID;
      gas.name = gasName;
      gas.stateArray = ChemState.empty;
      
      % find if the species is in the volume or at the surface (keyword 'wall_' case insensitive)
      if length(gasName)>5 && strcmpi(gasName(1:5), 'wall_')
        gas.isVolumeSpecies = false;
        gas.isSurfaceSpecies = true;
      else
        gas.isVolumeSpecies = true;
        gas.isSurfaceSpecies = false;
      end
      
    end
    
    function linkWithElectronKineticsGas(chemGas, eedfGas)
      
      % save handle to the equivalent electron kinetic gas
      chemGas.eedfEquivalent = eedfGas;
      eedfGas.chemEquivalent = chemGas;
      
      % copy properties of the equivalent electron kinetic gas (avoiding 'ID' and 'stateArray' properties)
      eedfGasProperties = fields(eedfGas)';
      chemGasProperties = fields(chemGas)';
      for property = eedfGasProperties
        if ~strcmp(property{1},'ID') && ~strcmp(property{1}, 'stateArray') && any(strcmp(property, chemGasProperties))
          chemGas.(property{1}) = eedfGas.(property{1});
        end
      end
      
    end
    
    function updateElectronKineticsEquivalentPopulations(gas)
    % updateElectronKineticsEquivalentPopulations is a function that updates the populations of the equivalent
    % gases that are used to solve the electron kinetics. The function avoids the electron kinetics gases for which
    % the user did not specify a cross section set (dummy gases). Because of the same reason the function also
    % avoids to update electronic or ionic states for which the user did not specify a non zero population.
      
      % avoid gases that does not have an eedf equivalent or gases whos states are not target of e-impact collisions
      if ~isempty(gas.eedfEquivalent) && ~isempty(gas.eedfEquivalent.collisionArray)
        % loop over all the states of a gas
        for state = gas.stateArray
          % avoid states without an eedf equivalent state and states who are not target of any e-impact collision
          if ~isempty(state.eedfEquivalent) && ~isempty(state.eedfEquivalent.collisionArray)
            state.eedfEquivalent.population = state.population;
          end
        end
      end
      
    end
    
    function checkThermalModelData(gas)
    % checkThermalModelData checks for the data needed for the thermal module to be activated to be present in the gas
    % properties (heat capacity and heat conductivity)
      
      % check for the definition of the gas mass
      if isempty(gas.mass)
        error(['A value for the mass is not found for gas %s.\n' ...
          'Thermal model can not be activated without it.\n' ...
          'Please check your setup file.'], gas.name);
      end
      
      % check for the definition of the heat conductivity
      if isempty(gas.thermalConductivity)
        error(['A value for the thermal conductivity is not found for gas %s.\n' ...
          'Thermal model can not be activated without it.\n' ...
          'Please check your setup file.'], gas.name);
      end
      
      % check for the definition of the heat capacity
      if isempty(gas.heatCapacity)
        error(['A value for the heat capacity is not found for gas %s.\n' ...
          'Thermal model can not be activated without it.\n' ...
          'Please check your setup file.'], gas.name);
      end
      
    end
    
  end
  
  methods (Static)
    
    function updateElectronKineticsFractions(gasArray)
    % updateElectronKineticsFractions is a function that updates (properly reescaling) the fractions of the
    % equivalent electron kinetics gases. The function avoids gases that do not have an electron kinetics
    % equivalent with a cross section set (dummy gases).
      
      % find gases to update and evaluate the fractions norm
      gasesToUpdate = [];
      fractionNorm = 0;
      for i = 1:length(gasArray)
        if ~isempty(gasArray(i).eedfEquivalent) && ~isempty(gasArray(i).eedfEquivalent.collisionArray)
          gasesToUpdate(end+1) = i;
          fractionNorm = fractionNorm + gasArray(i).fraction;
        end
      end
      
      % update gases
      for i = gasesToUpdate
        gasArray(i).eedfEquivalent.fraction = gasArray(i).fraction/fractionNorm;
      end
      
    end
    
  end
  
end