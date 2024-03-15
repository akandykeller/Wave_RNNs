<h1 align='center'> Traveling Waves Encode the Recent Past <br> and Enhance Sequence Learning </h1>

This repository contains the implementation to reproduce the experiments 
of the *ICLR 2024* submission: "Traveling Waves Encode the Recent Past <br> and Enhance Sequence Learning".

To install the requirements:
`conda env create -f environment.yml`

This repo is equipped with Weights & Biases tracking by default. To enable it you must first install wandb separately: 
`pip install wandb`

Then, enter your Project Name, Entitiy Name, and Logging directory into the initalization of each task file:
`wandb.init(name=args.run_name,
            project='PROJECT_NAME', 
            entity='ENTITY_NAME', 
            dir='WANDB_DIR',
            config=args)`

For more information on making a Weight & Biases account see [(creating a weights and biases account)](https://app.wandb.ai/login?signup=true) and the associated [quickstart guide](https://docs.wandb.com/quickstart).
