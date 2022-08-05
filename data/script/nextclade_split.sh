for file in ncbi_dataset/data/split/*
do
    echo "applying nextcalde on $file"
    ./nextclade run --input-dataset reference/  --output-tsv=$(echo output/$(basename $file).tsv)  $file --verbose 
done