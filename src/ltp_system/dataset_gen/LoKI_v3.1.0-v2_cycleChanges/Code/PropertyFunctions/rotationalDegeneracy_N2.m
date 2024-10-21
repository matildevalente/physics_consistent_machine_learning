function statisticalWeight = rotationalDegeneracy_N2(state, ~, ~)
  % rotationalDegeneracy_N2 (have to be writen)
  
  if ~strcmp(state.type, 'rot')
    error(['Trying to asign rotational degeneracy to non rotational state %s. Check input file', state.name]);
  end
  
  J = str2double(state.rotLevel);
  statisticalWeight = 3*(1+0.5*(1+(-1)^J))*(2*J+1);
  
end
