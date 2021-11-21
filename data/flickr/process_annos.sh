echo "Processing train set"
python3 -m data.flickr.write_annos_to_json --subset train
echo "Processing val set"
python3 -m data.flickr.write_annos_to_json --subset val
echo "Processing test set"
python3 -m data.flickr.write_annos_to_json --subset test
