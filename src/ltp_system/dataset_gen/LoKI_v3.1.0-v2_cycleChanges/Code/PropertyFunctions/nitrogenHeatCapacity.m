function heatCapacity = nitrogenHeatCapacity(gas, ~, workCond)
  % atomicNitrogenHeatCapacity (have to be writen)
  
  % heat capacity for nitrogen (SI units)
  heatCapacity = 29.1 + 2494.2/(553.4*sqrt(pi/2))*exp(-2*((workCond.gasTemperature-1047.4)/553.4)^2);
  % change energy units to eV
  heatCapacity = heatCapacity/Constant.electronCharge;
  % store value on gas object properties
  gas.heatCapacity = heatCapacity;
  
  
end
