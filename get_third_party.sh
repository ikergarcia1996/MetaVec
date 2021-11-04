dependencies=$1


cd MetaVec || exit
echo "Downloading VecMap"
git clone https://github.com/artetxem/vecmap.git

if [ "$dependencies" = "all" ];
then
cd ..
echo "Downloading Word Embeddings Benchmarks"
git clone https://github.com/ikergarcia1996/word-embeddings-benchmarks.git
cd word-embeddings-benchmarks || exit
python setup.py install

echo "Downloading Jiant V1"
git clone https://github.com/ikergarcia1996/jiant-v1-legacy
cd jiant-v1-legacy || exit
git submodule update --init --recursive

echo "Downloading Glue taks for Jiant V1"
python scripts/download_glue_data.py --data_dir data --tasks all

echo "Downloading NLTK and SPACY requeriments for Jiant"
python -m nltk.downloader perluniprops nonbreaking_prefixes punkt
python -m spacy download en

fi