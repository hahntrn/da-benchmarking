import argparse
from tqdm import tqdm

# TEST tqdm
import time
d = {0:1, 1:2, 2:3, 3:4, 4:5}
l = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
def fn():
    for k,v in tqdm(l):
        time.sleep(0.1)

for i in tqdm(range(20)):
    fn()

# %%capture cap --no-stderr
# print('yay')


parser = argparse.ArgumentParser(description='Train with CORAL loss')
parser.add_argument('--lr', type=float)
parser.add_argument('--alpha', type=float)
parser.add_argument('--n-epochs', type=int)

args = parser.parse_args()
print(args)
print(args.lr)
print(args.n_epochs)
