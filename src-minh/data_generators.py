import gzip
import torch
import random
import numpy as np
from pyfaidx import Fasta
from torch.utils.data import Dataset

import os
GENOMES = { "mouse" : "/users/kcochran/genomes/mm10_no_alt_analysis_set_ENCODE.fasta",
            "human" : "/users/kcochran/genomes/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta" }

ROOT = "/users/kcochran/projects/cs197_cross_species_domain_adaptation/"
DATA_DIR = ROOT + "data/"

SPECIES = ["mouse", "human"]

TFS = ["CTCF", "CEBPA", "HNF4A", "RXRA"]


class ValGenerator(Dataset):
    letter_dict = {
        'a':[1,0,0,0], 'c':[0,1,0,0], 'g':[0,0,1,0], 't':[0,0,0,1], 'n':[0,0,0,0],
        'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1], 'N':[0,0,0,0]}

    def __init__(self, species, tf, return_labels = True, batchsize=1000):
        self.valfile = DATA_DIR + species + "/" + tf + "/chr1_random_1m.bed.gz"
        self.converter = Fasta(GENOMES[species])
        self.batchsize = batchsize  # arbitrarily large number that will fit into memory
        self.return_labels = return_labels
        self.get_coords_and_labels()


    def __len__(self):
        return self.steps_per_epoch


    def get_coords_and_labels(self):
        with gzip.open(self.valfile) as f:
            coords_tmp = [line.decode().split()[:4] for line in f]  # expecting bed file format
        
        self.labels = [int(coord[3]) for coord in coords_tmp]
        self.coords = [(coord[0], int(coord[1]), int(coord[2])) for coord in coords_tmp]  # no strand consideration
        
        self.steps_per_epoch = int(len(self.coords) / self.batchsize)

    def convert(self, coords):
        seqs_onehot = []
        for coord in coords:
            chrom, start, stop = coord
            seq = self.converter[chrom][start:stop].seq
            seq_onehot = np.array([self.letter_dict.get(x,[0,0,0,0]) for x in seq])
            seqs_onehot.append(seq_onehot)

        seqs_onehot = np.array(seqs_onehot)
        return seqs_onehot


    def __getitem__(self, batch_index):	
        # First, get chunk of coordinates
        batch_start = batch_index * self.batchsize
        batch_end = (batch_index + 1) * self.batchsize
        coords_batch = self.coords[batch_start : batch_end]

        # if train_steps calculation is off, lists of coords may be empty
        assert len(coords_batch) > 0, len(coords_batch)

        # Second, convert the coordinates into one-hot encoded sequences
        onehot = self.convert(coords_batch)

        # array will be empty if coords are not found in the genome
        assert onehot.shape[0] > 0, onehot.shape[0]

        onehot = torch.tensor(onehot, dtype=torch.float).permute(0, 2, 1)
        
        if self.return_labels:
            labels = self.labels[batch_start : batch_end]
            labels = torch.tensor(labels, dtype=torch.float)
            return onehot, labels
        else:
            return onehot
        
        
class TrainGenerator(Dataset):
    letter_dict = {
        'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],
        'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],
        'T':[0,0,0,1],'N':[0,0,0,0]}

    def __init__(self, species, tf):
        self.posfile = DATA_DIR + species + "/" + tf + "/chr3toY_pos_shuf.bed.gz"
        self.negfile = DATA_DIR + species + "/" + tf + "/chr3toY_neg_shuf_run1_1E.bed.gz"
        self.converter = Fasta(GENOMES[species])
        self.batchsize = 400
        self.halfbatchsize = self.batchsize // 2
        self.current_epoch = 1

        self.get_coords()
        self.on_epoch_end()


    def __len__(self):
        return self.steps_per_epoch


    def get_coords(self):
        with gzip.open(self.posfile) as posf:
            pos_coords_tmp = [line.decode().split()[:3] for line in posf]  # expecting bed file format
            self.pos_coords = [(coord[0], int(coord[1]), int(coord[2])) for coord in pos_coords_tmp]  # no strand consideration
        with gzip.open(self.negfile) as negf:
            neg_coords_tmp = [line.decode().split()[:3] for line in negf]
            self.neg_coords = [(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]
            
        self.steps_per_epoch = int(len(self.pos_coords) / self.halfbatchsize)
        print(self.steps_per_epoch)
                

    def convert(self, coords):
        seqs_onehot = []
        for coord in coords:
            chrom, start, stop = coord
            seq = self.converter[chrom][start:stop].seq
            seq_onehot = np.array([self.letter_dict.get(x,[0,0,0,0]) for x in seq])
            seqs_onehot.append(seq_onehot)

        seqs_onehot = np.array(seqs_onehot)
        return seqs_onehot


    def __getitem__(self, batch_index):	
        # First, get chunk of coordinates
        pos_coords_batch = self.pos_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]
        neg_coords_batch = self.neg_coords[batch_index * self.halfbatchsize : (batch_index + 1) * self.halfbatchsize]

        # if train_steps calculation is off, lists of coords may be empty
        assert len(pos_coords_batch) > 0, len(pos_coords_batch)
        assert len(neg_coords_batch) > 0, len(neg_coords_batch)

        # Second, convert the coordinates into one-hot encoded sequences
        pos_onehot = self.convert(pos_coords_batch)
        neg_onehot = self.convert(neg_coords_batch)

        # seqdataloader returns empty array if coords are empty list or not in genome
        assert pos_onehot.shape[0] > 0, pos_onehot.shape[0]
        assert neg_onehot.shape[0] > 0, neg_onehot.shape[0]

        # Third, combine bound and unbound sites into one large array, and create label vector
        # We don't need to shuffle here because all these examples will correspond
        # to a simultaneous gradient update for the whole batch
        all_seqs = np.concatenate((pos_onehot, neg_onehot))
        labels = np.concatenate((np.ones(pos_onehot.shape[0],), np.zeros(neg_onehot.shape[0],)))

        all_seqs = torch.tensor(all_seqs, dtype=torch.float).permute(0, 2, 1)
        labels = torch.tensor(labels, dtype=torch.float)
        assert all_seqs.shape[0] == self.batchsize, all_seqs.shape[0]
        return all_seqs, labels


    def on_epoch_end(self):
        # switch to next set of negative examples
        prev_epoch = self.current_epoch
        next_epoch = prev_epoch + 1

        # update file where we will retrieve unbound site coordinates from
        prev_negfile = self.negfile
        next_negfile = prev_negfile.replace(str(prev_epoch) + "E", str(next_epoch) + "E")
        self.negfile = next_negfile

        # load in new unbound site coordinates
        with gzip.open(self.negfile) as negf:
            neg_coords_tmp = [line.decode().split()[:3] for line in negf]
            self.neg_coords = [(coord[0], int(coord[1]), int(coord[2])) for coord in neg_coords_tmp]

        # then shuffle positive examples
        random.shuffle(self.pos_coords)
        
        
