## UKB
Website: https://ixa2.si.ehu.es/ukb/  
Direct Download Link: https://ixa2.si.ehu.es/ukb/embeddings/RWSGwn.emb.tar.gz  
Pre Processing: None  

## FastText
Website: https://fasttext.cc/docs/en/english-vectors.html  
Direct Download Link: https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip  
Pre Processing: None  


## Attract-Repel
Website: https://github.com/nmrksic/attract-repel  
Direct Download Link: https://drive.google.com/open?id=0B_pyA_IW4g-jZzBIZXpYS1RseFk  
Pre Processing:  


```
python glove_2_word2vec.py --embedding_path embeddings/en-de-it-ru.txt --output_path embeddings/en-AT.txt
python remove_prefix.py --embedding_path embeddings/en-AT.txt --output_path embeddings/en-AT.vec --prefix_separator _ --allowed_prefixes en
```


## LexVec
Website: https://github.com/alexandres/lexvec  
Direct Download Link: https://www.dropbox.com/s/mrxn933chn5u37z/lexvec.commoncrawl.ngramsubwords.300d.W.pos.vectors.gz?dl=1  
Pre Processing: None  


## Paragram
Website: 
- https://github.com/jwieting/paragram-word  
- http://www.cs.cmu.edu/~jwieting/  
Direct Download Link:
- ws353 (We use this one): https://drive.google.com/uc?id=0B9w48e1rj-MOLVdZRzFfTlNsem8&export=download  
- sl999: https://drive.google.com/uc?id=0B9w48e1rj-MOck1fRGxaZW1LU2M&export=download  

Pre Processing:
```
python3 clean.py --embedding_path embeddings/paragram_300_ws353.txt --output_path embeddings/paragram_ws353.txt
python3 clean.py --embedding_path embeddings/paragram_300_sl999.txt --output_path embeddings/paragram_sl999.txt
python3 glove_2_word2vec.py --embedding_path embeddings/paragram_ws353.txt --output_path embeddings/paragram_ws353.vec
python3 glove_2_word2vec.py --embedding_path embeddings/paragram_sl999.txt --output_path embeddings/paragram_sl999.vec
sed -i '$ d' paragram_sl999.vec
sed -i '$ d' paragram_ws353.vec
sed -i 's/^ *//' paragram_sl999.vec
sed -i 's/^ *//' paragram_ws353.vec


python3 #Launch a python3 terminal from the root directory of this repository
from MetaVec.embedding import Embedding
emb = Embedding.from_file("embeddings/paragram_sl999.vec")
emb.save("embeddings/paragram_sl999.vec")
emb = Embedding.from_file("embeddings/paragram_ws353.vec")
emb.save("embeddings/paragram_ws353.vec")

```

## JOINTChyb
Website: http://ixa2.si.ehu.es/ukb/bilingual_embeddings.html  
Direct Download Link: http://ixa2.si.ehu.es/ukb/embeddings/JOINTC-HYB.tar.bz2  
Pre Processing: None  

## Numberbatch
Website: https://github.com/commonsense/conceptnet-numberbatch  
Direct Download Link: https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz  
Pre Processing: None  

## LexSub
Website: https://github.com/aishikchakraborty/LexSub  
Direct Download Link: https://drive.google.com/file/d/1DtMLJzvwfak25PoRrt5Wq4_3w9x7r8Yy/view?usp=sharing  
Pre Processing: The file we want is: emb_glove_retro_syn_hyp_mer_300_300_100_cosine_wn_v2.txt  

```
python3 glove_2_word2vec.py --embedding_path embeddings/emb_glove_retro_syn_hyp_mer_300_300_100_cosine_wn_v2.txt --output_path embeddings/LexSub.vec  
```


## GloVe
Website: https://nlp.stanford.edu/projects/glove/  
Direct Download Link: http://nlp.stanford.edu/data/glove.840B.300d.zip  
Pre Processing:  
```
python3 glove_2_word2vec.py --embedding_path embeddings/glove.840B.300d.txt --output_path embeddings/glove.vec --normalize
```

## Word2Vec (GoogleNews)
Website: https://code.google.com/archive/p/word2vec/  
Direct Download Link: https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download 
Pre Processing: 

```
python3 
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.save_word2vec_format('w2v.vec', binary=False)
```
