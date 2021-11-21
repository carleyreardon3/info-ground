download_dir="/projectnb/cs591-mm-ml/reardonc/info-ground/flickr30kJP_downloads"
git clone https://github.com/nlab-mpg/Flickr30kEnt-JP.git $download_dir

annotations_zip="${download_dir}/Sentences_jp_v2.tgz"
tar -xvf $annotations_zip -C $download_dir
