wandb-sync:
	wandb sync --sync-all

clean-data:
	rm -rf ./data/processed

clean-pycache:
	find . -name "*pycache*" -type d -exec rm -rf {} +

clean-wandb:
	rm -rf wandb

clean-all:
	find . -name "*pycache*" -type d -exec rm -rf {} +
	rm -rf wandb
