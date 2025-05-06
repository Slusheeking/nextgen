from datasets import load_dataset
import os

# Available configurations
configs = [
    'sentences_allagree',    # Sentences with 100% agreement among annotators
    'sentences_75agree',     # Sentences with at least 75% agreement
    'sentences_66agree',     # Sentences with at least 66% agreement
    'sentences_50agree'      # Sentences with at least 50% agreement
]

# Create base output directory
base_output_dir = "./financial_phrasebank_dataset"
os.makedirs(base_output_dir, exist_ok=True)

# Download and save each configuration
for config in configs:
    print(f"Downloading configuration: {config}")
    
    # Load dataset with specific configuration
    dataset = load_dataset("takala/financial_phrasebank", config)
    
    # Create configuration-specific output directory
    config_output_dir = os.path.join(base_output_dir, config)
    
    # Save the dataset
    dataset.save_to_disk(config_output_dir)
    
    print(f"Dataset {config} downloaded and saved to {config_output_dir}")
    print("Dataset information:")
    print(dataset)
    print("\n" + "="*50 + "\n")

print("All Financial PhraseBank dataset configurations downloaded successfully.")
