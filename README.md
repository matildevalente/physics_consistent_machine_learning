# Physics-consistent machine learning 

Data-driven machine learning models often require extensive datasets, which can be costly or inaccessible, and their predictions may fail to comply with established physical laws. Current approaches for incorporating physical priors mitigate these issues by penalizing deviations from known physical laws, as in physics-informed neural networks, or by designing architectures that automatically satisfy specific invariants. However, penalization approaches do not guarantee compliance withphysical constraints for unseen inputs, and invariant-based methods can lack flex-ibility and generality. We propose a novel physics-consistent machine learningmethod that directly enforces compliance with physical principles by projectingmodel outputs onto the subspace defined by these laws. This ensures that predic-tions inherently adhere to the chosen physical constraints, improving reliabilityand interpretability. Our method is demonstrated on two systems: a spring-masssystem and a low-temperature reactive plasma. Compared to purely data-drivenmodels, our approach significantly reduces errors in physical law compliance,enhances predictive accuracy of physical quantities, and outperforms alternativeswhen working with simpler models or limited datasets. The proposed projection-based technique is versatile and can function independently or in conjunctionwith existing physics-informed neural networks, offering a powerful, general, andscalable solution for developing fast and reliable surrogate models of complexphysical systems, particularly in resource-constrained scenarios.

For more information, please refer to the following: (https://doi.org/10.48550/arXiv.2502.15755)

## Requirements
The required dependencies are listed in `requirements.txt`. Install them using pip:
```bash
pip install -r requirements.txt
```


# Usage
1. Configure the system parameters in the config file
2. Or run the main system directly using the default parameters
3. To run the main programs use 
```bash
python LowTemperaturePlasma.py
```
```bash
python SpringMassSystem.py
```




### Project Structure
```
.
├── configs/                            # Configuration files for both systems
├── data/                               # Dataset directory
├── output/                             # Checkpoints directory
├── src/                                # Methods implementation & plots & tables
├── clean.py                            # Script to clear the directories to help running from scratch
├── main_ltp_system.py                  # Main LTP system implementation
└── main_spring_mass_system.py          # Main spring-mass system implementation
```


## Contact
Contact us through matilde.valente@tecnico.ulisboa.pt


