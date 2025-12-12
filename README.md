# CSG4Rec
This is the pytorch implementation of our paper:
> CSG4Rec: Collaborative-Similarity Guided Semantic IDs Tokenization for Generative Sequential Recommendation

## Requirements
This project is developed with the following environment:
- Python 3.9.23
- PyTorch 2.8.0 (CUDA 12.8)

```
torch==2.8.0+cu128
accelerate
bitsandbytes
deepspeed
evaluate
peft
sentencepiece
tqdm
transformers
scikit-learn
k-means-constrained
```

## CSGT
### Train
```
bash CSGT/train_tokenizer.sh 
```
### Tokenize
```
bash CSGT/tokenize.sh 
```

## Generation
### Train
```
cd Generation
bash run_train.sh
```
### Test
```
cd Generation
bash run_test.sh
```
