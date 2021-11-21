download_dir="/projectnb/cs591-mm-ml/reardonc/info-ground/flickr30k_downloads"
git clone https://github.com/BryanPlummer/flickr30k_entities.git $download_dir

annotations_zip="${download_dir}/annotations.zip"
unzip $annotations_zip -d $download_dir
