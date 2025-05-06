# Financial PhraseBank Dataset

## Dataset Configurations

The Financial PhraseBank dataset is available in four different configurations, based on the level of annotator agreement:

1. **sentences_allagree** (2,264 sentences)
   - Contains sentences with 100% agreement among annotators
   - Highest confidence level
   - Most conservative dataset

2. **sentences_75agree** (3,453 sentences)
   - Contains sentences with at least 75% agreement among annotators
   - High confidence level
   - More sentences than the all-agree configuration

3. **sentences_66agree** (4,217 sentences)
   - Contains sentences with at least 66% agreement among annotators
   - Moderate confidence level
   - Larger dataset with more varied annotations

4. **sentences_50agree** (4,846 sentences)
   - Contains sentences with at least 50% agreement among annotators
   - Lowest confidence level
   - Largest dataset, includes more diverse and potentially more ambiguous sentences

## Dataset Structure

Each configuration contains a single training split with two features:
- `sentence`: The text of the financial sentence
- `label`: The sentiment or classification label

## Recommended Usage

- For high-precision tasks: Use `sentences_allagree`
- For balanced performance: Use `sentences_75agree`
- For maximum data coverage: Use `sentences_50agree`

## Download Location

The datasets are saved in the following directories:
- `./financial_phrasebank_dataset/sentences_allagree`
- `./financial_phrasebank_dataset/sentences_75agree`
- `./financial_phrasebank_dataset/sentences_66agree`
- `./financial_phrasebank_dataset/sentences_50agree`
