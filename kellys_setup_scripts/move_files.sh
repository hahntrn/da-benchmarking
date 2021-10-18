#!/bin/bash

set -e

old_dir="/users/kcochran/projects/domain_adaptation/data"
old_dir_raw="/users/kcochran/projects/domain_adaptation/raw_data"

# this directory should already be created via soft-linking
new_dir="/users/kcochran/projects/cs197_cross_species_domain_adaptation/data"


tfs=( "CTCF" "CEBPA" "Hnf4a" "RXRA" )
specieses=( "mm10" "hg38" )


files_to_move=( "chr3toY_pos_shuf.bed" "chr1_random_1m.bed" "chr2.bed" "chr3toY_neg_shuf_run1_10E.bed" "chr3toY_neg_shuf_run1_11E.bed" "chr3toY_neg_shuf_run1_12E.bed" "chr3toY_neg_shuf_run1_13E.bed" "chr3toY_neg_shuf_run1_14E.bed" "chr3toY_neg_shuf_run1_15E.bed" "chr3toY_neg_shuf_run1_1E.bed" "chr3toY_neg_shuf_run1_2E.bed" "chr3toY_neg_shuf_run1_3E.bed" "chr3toY_neg_shuf_run1_4E.bed" "chr3toY_neg_shuf_run1_5E.bed" "chr3toY_neg_shuf_run1_6E.bed" "chr3toY_neg_shuf_run1_7E.bed" "chr3toY_neg_shuf_run1_8E.bed" "chr3toY_neg_shuf_run1_9E.bed" "chr3toY_shuf_run1_10E.bed" "chr3toY_shuf_run1_11E.bed" "chr3toY_shuf_run1_12E.bed" "chr3toY_shuf_run1_13E.bed" "chr3toY_shuf_run1_14E.bed" "chr3toY_shuf_run1_15E.bed" "chr3toY_shuf_run1_1E.bed" "chr3toY_shuf_run1_2E.bed" "chr3toY_shuf_run1_3E.bed" "chr3toY_shuf_run1_4E.bed" "chr3toY_shuf_run1_5E.bed" "chr3toY_shuf_run1_6E.bed" "chr3toY_shuf_run1_7E.bed" "chr3toY_shuf_run1_8E.bed" "chr3toY_shuf_run1_9E.bed" )


for tf in "${tfs[@]}"; do
  for species in "${specieses[@]}"; do
    old_root="$old_dir/${species}/${tf}"
    new_root="$new_dir/${species}/${tf}"
    mkdir -p "$new_root"
    for filename in "${files_to_move[@]}"; do
      echo "$tf" "$species" "$filename"
      if [[ -f "$new_root/${filename}.gz" ]] ; then
        rm "$new_root/${filename}.gz"
      fi

      cp "$old_root/$filename" "$new_root/$filename"
      gzip "$new_root/$filename"
    done
  
    echo "$tf" "$species"
    cp "$old_dir_raw/$species/$tf/mgps_out_${tf}.bed" "$new_root/raw_peaks.bed"
    gzip "$new_root/raw_peaks.bed"
  done
done

# renaming directories to be more intuitive
for species in "${specieses[@]}"; do
  mv "$new_dir/${species}/Hnf4a" "$new_dir/${species}/HNF4A"
done

mv "$new_dir/mm10" "$new_dir/mouse"
mv "$new_dir/human" "$new_dir/human"
