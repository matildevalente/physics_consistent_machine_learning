# Paper

projection_paper/
│
├── data/                           # Store raw and processed data
│   ├── ltp_system/
│   └── spring_mass_system/
├── configs/                        # Configuration files for each problem
│   ├── ltp_system_config.yaml
│   └── spring_mass_system_config.yaml
├── src/                            # Source code directory
│   ├── ltp_system/                   
│   │   ├── dataset_gen/            # Directory for dataset generation 
│   │   ├── figures/                # Stores the plots in .eps and .pdf formats
│   │   ├── __init__.py             # Combined PINN model and training 
│   │   ├── pinn.py                 # Combined PINN model and training 
│   │   ├── nn.py                   # Combined NN model and training 
│   │   ├── projection.py           # Projection method implementation
│   │   ├── plotter.py              # Contains logic to creates the plots
│   │   ├── utils.py                # Utility file
│   │   └── data_prep.py            # Data preprocessing 
│   │
│   └── spring_mass_system/                   
│       ├── dataset_gen/            # Directory for dataset generation 
│       ├── figures/                # Stores the plots in .eps and .pdf formats
│       ├── __init__.py             # Combined PINN model and training 
│       ├── pinn.py                 # Combined PINN model and training 
│       ├── nn.py                   # Combined NN model and training 
│       ├── projection.py           # Projection method implementation
│       ├── plotter.py              # Contains logic to creates the plots
│       ├── utils.py                # Utility file
│       └── data_prep.py            # Data preprocessing 
│
├── output/                         # Save model checkpoints, results, and plots
├── main_ltp_system.py              # Main script for LTP system
├── main_spring_mass_system.py      # Main script for Spring Mass System
└── requirements.txt                # Python dependencies

to activate the venv in the projection_paper folder:
conda activate /Users/matildevalente/anaconda3/envs/ML1/

