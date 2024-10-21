function energy = harmonicOscillatorEnergy(state, ~, ~)
  % harmonicOscillator (have to be writen)
  
  if ~strcmp(state.type, 'vib')
    error('Trying to asign harmonic oscillator energy to non vibrational state %s. Check input file', state.name);
  elseif isempty(state.gas.harmonicFrequency)
    error(['Unable to find harmonicFrequency to evaluate the energy of the state %s with function ' ...
      '''harmonicOscillatorEnergy''.\nCheck input file'], state.name);
  end
  
  v = str2double(state.vibLevel);
  energy = Constant.planckReducedInEV*state.gas.harmonicFrequency*(v+0.5);
  
end
