root="/users/kcochran/projects/cs197_cross_species_domain_adaptation"

data_dir="${root}/data"

# conservation tracks from UCSC Genome Browser
# e.g. https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/

wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw -O "${data_dir}/human/conservation.phyloP.bigWig"
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw -O "${data_dir}/human/conservation.phastCons.bigWig" 

wget https://hgdownload.soe.ucsc.edu/goldenPath/mm10/phyloP60way/mm10.60way.phyloP60way.bw -O "${data_dir}/mouse/conservation.phyloP.bigWig"
wget https://hgdownload.soe.ucsc.edu/goldenPath/mm10/phastCons60way/mm10.60way.phastCons.bw -O "${data_dir}/mouse/conservation.phastCons.bigWig"

