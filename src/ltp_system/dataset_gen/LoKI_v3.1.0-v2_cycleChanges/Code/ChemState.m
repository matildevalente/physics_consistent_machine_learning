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

classdef ChemState < State
  
  properties
    
    reactionsCreation = Reaction.empty;     % handle to the reactions where this state is created (forward reaction)
    reactionsDestruction = Reaction.empty;  % handle to the reactions where this state is destroyed (forward reaction)
    
    eedfEquivalent = EedfState.empty;       % handle to the corresponding equivalent eedf state (in case it exists)
    
  end
  
  events
    
  end
  
  methods (Access = public)
    
    function state = ChemState(gas, ionCharg, eleLevel, vibLevel, rotLevel)
      persistent lastID;
      if isempty(lastID)
        lastID = 0;
      end
      lastID = lastID + 1;
      state.ID = lastID;
      state.gas = gas;
      state.ionCharg = ionCharg;
      state.eleLevel = eleLevel;
      state.vibLevel = vibLevel;
      state.rotLevel = rotLevel;
      if isempty(ionCharg)
        if isempty(rotLevel)
          if isempty(vibLevel)
            state.type = 'ele';
          else
            state.type = 'vib';
          end
        else
          state.type = 'rot';
        end
      else
        state.type = 'ion';
      end
      state.parent = ChemState.empty;
      state.siblingArray = ChemState.empty;
      state.childArray = ChemState.empty;
      state.addFamily;
      gas.stateArray(end+1) = state;
      state.evaluateName;
    end
    
    function isVolumeSpecies = isVolumeSpecies(state)
    % isVolumeSpecies return a boolean value that is true if the parent gas of the state is defined as a volume species
    % and false otherwise
      
      isVolumeSpecies = state.gas.isVolumeSpecies;

    end

    function isSurfaceSpecies = isSurfaceSpecies(state)
    % isSurfaceSpecies return a boolean value that is true if the parent gas of the state is defined as a surface species
    % and false otherwise
      
      isSurfaceSpecies = state.gas.isSurfaceSpecies;

    end

    function linkWithElectronKineticsState(chemState, eedfState)
      
      % save handle to the equivalent electron kinetic state
      chemState.eedfEquivalent = eedfState;
      eedfState.chemEquivalent = chemState;
      
      % copy properties of the equivalent electron kinetic state (avoiding 'ID', 'gas', 'parent', 'siblingArray' and 
      % 'childArray'  properties)
      eedfStateProperties = fields(eedfState)';
      chemStateProperties = fields(chemState)';
      for property = eedfStateProperties
        if ~strcmp(property{1},'ID') && ~strcmp(property{1}, 'gas') && ~strcmp(property{1}, 'parent') && ...
            ~strcmp(property{1}, 'siblingArray') && ~strcmp(property{1}, 'childArray') && ...
            any(strcmp(property, chemStateProperties))
          chemState.(property{1}) = eedfState.(property{1});
        end
      end
      
    end
    
  end
  
end
