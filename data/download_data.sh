echo 'Paraphraser'
wget -O Paraphraser.zip 'http://paraphraser.ru/download/get?file_id=1'
unzip Paraphraser.zip -d Paraphraser && rm -f Paraphraser.zip
wget -O ParaphraserGold.zip 'http://paraphraser.ru/download/get?file_id=5'
unzip ParaphraserGold.zip -d Paraphraser && rm -f ParaphraserGold.zip

echo 'MSRVid'
wget http://files.deeppavlov.ai/datasets/STS2012_MSRvid_translated.tar.gz
tar zxvf STS2012_MSRvid_translated.tar.gz && rm -f rm -f STS2012_MSRvid_translated.tar.gz

echo 'XNLI'
wget http://www.nyu.edu/projects/bowman/xnli/XNLI-1.0.zip
unzip XNLI-1.0.zip && rm -f XNLI-1.0.zip
rm -rf __MACOSX

echo 'Rusentiment'
git clone --depth=1 --branch=master https://github.com/text-machine-lab/rusentiment Rusentiment
rm -rf !$/.git
mv Rusentiment/Dataset/* Rusentiment
rm -rf Rusentiment/Dataset
