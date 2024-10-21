function energy = morseOscillatorEnergy(state, ~, ~)
  % morseOscillator (have to be writen)
  
  
  if ~strcmp(state.type, 'vib')
    error('Trying to asign morse oscillator energy to non vibrational state %s. Check input file', state.name);
  elseif isempty(state.gas.harmonicFrequency)
    error(['Unable to find harmonicFrequency to evaluate the energy of the state %s with function ' ...
      '''harmonicOscillatorEnergy''.\nCheck input file'], state.name);
  elseif isempty(state.gas.anharmonicFrequency)
    error(['Unable to find anharmonicFrequency to evaluate the energy of the state %s with function ' ...
      '''morseOscillatorEnergy''.\nCheck input file'], state.name);
  end
  
  v = str2double(state.vibLevel);
  energy = Constant.planckReducedInEV*(state.gas.harmonicFrequency*(v+0.5)-state.gas.anharmonicFrequency*(v+0.5)^2);
  
end
