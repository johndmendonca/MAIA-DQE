# Emotion Recognition in Conversations: Benchmark Code

## Train:

```bash
python cli.py train -f configs/maia.yaml
```

## Interact:
Fun command to interact with with a trained model.

```bash
python cli.py interact --experiment experiments/{experiment_id}/
```

## Testing:

```bash
python cli.py test --experiment experiments/{experiment_id}/
```