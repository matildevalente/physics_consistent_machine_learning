function heatCapacity = oxygenHeatCapacity(gas, ~, workCond)
  % oxygenHeatCapacity (have to be writen)
  
  % heat capacity for oxygen (SI units)
  heatCapacity = 28.8 + 6456.2/(788.3*sqrt(pi/2))*exp(-2*((workCond.gasTemperature-1006.9)/788.3)^2);
  % change energy units to eV
  heatCapacity = heatCapacity/Constant.electronCharge;
  % store value on gas object properties
  gas.heatCapacity = heatCapacity;
  
  
end
