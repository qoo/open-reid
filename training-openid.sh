export PYTHONPATH=~/open-reid

pip install scikit-learn
pip install metric_learn

python examples/softmax_loss.py -d viper -b 64 -j 2 -a inception --logs-dir logs/softmax-loss/viper-inception


  File "build/bdist.linux-x86_64/egg/reid/trainers.py", line 33, in train
IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number

sudo apt install python3-pip -y

pip3 install scikit-learn
pip3 install metric_learn
pip3 install h5py

python3 examples/softmax_loss.py -d viper -b 64 -j 2 -a inception --logs-dir logs/softmax-loss/viper-inception

Traceback (most recent call last):
  File "examples/softmax_loss.py", line 217, in <module>
    main(parser.parse_args())
  File "examples/softmax_loss.py", line 147, in main
    trainer.train(epoch, train_loader, optimizer)
  File "/root/open-reid/reid/trainers.py", line 33, in train
    losses.update(loss.data[0], targets.size(0))
IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number

https://github.com/layumi/Person_reID_baseline_pytorch/issues/33
version =  torch.__version__

if int(version[2]) > 3: # for the new version like 0.4.0 and 0.5.0
    running_loss += loss.item()
else :  # for the old version like 0.3.0 and 0.3.1
    running_loss += loss.data[0]

 * Finished epoch  49  top1:  8.5%  best: 10.0%

Test with best model:
=> Loaded checkpoint 'logs/softmax-loss/viper-inception/model_best.pth.tar'
Extract Features: [1/10]	Time 1.226 (1.226)	Data 1.098 (1.098)	
Extract Features: [2/10]	Time 0.126 (0.676)	Data 0.000 (0.549)	
Extract Features: [3/10]	Time 0.089 (0.480)	Data 0.000 (0.366)	
Extract Features: [4/10]	Time 0.100 (0.385)	Data 0.000 (0.275)	
Extract Features: [5/10]	Time 0.112 (0.331)	Data 0.000 (0.220)	
Extract Features: [6/10]	Time 0.155 (0.301)	Data 0.000 (0.183)	
Extract Features: [7/10]	Time 0.141 (0.278)	Data 0.000 (0.157)	
Extract Features: [8/10]	Time 0.136 (0.261)	Data 0.000 (0.137)	
Extract Features: [9/10]	Time 0.113 (0.244)	Data 0.000 (0.122)	
Extract Features: [10/10]	Time 4.280 (0.648)	Data 0.000 (0.110)	
Mean AP: 8.1%
CMC Scores    allshots      cuhk03  market1501
  top-1           2.7%        5.1%        2.7%
  top-5          10.1%       19.5%       10.1%
  top-10         17.7%       30.4%       17.7%



# test triploss, best case
python3 examples/triplet_loss.py -d cuhk03 -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/cuhk03-resnet50

RuntimeError: Please download the dataset manually from https://docs.google.com/spreadsheet/viewform?usp=drive_web&formkey=dHRkMkFVSUFvbTJIRkRDLWRwZWpONnc6MA#gid=0 to /root/open-reid/examples/data/cuhk03/raw/cuhk03_release.zip

wget -O /root/open-reid/examples/data/cuhk03/raw/cuhk03_release.zip https://www.dropbox.com/s/ezlfz6xccpydkbu/cudnn-9.0-linux-x64-v7.3.0.29.tar?dl=0 https://www.dropbox.com/s/g63s3lix7x6p5ln/cuhk03_release.zip?dl=0


  File "/root/open-reid/reid/loss/triplet.py", line 27, in forward
    dist_ap = torch.cat(dist_ap)
RuntimeError: zero-dimensional tensor (at position 0) cannot be concatenated

https://github.com/Cysu/open-reid/issues/69

What worked for me is replacing torch.cat with torch.stack, but I am not entirely sure if this solution is unproblematic.

# 0.4.1
python examples/softmax_loss.py -d viper -b 64 -j 2 -a inception --logs-dir logs/softmax-loss/viper-inception
ok

python examples/triplet_loss.py -d cuhk03 -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/cuhk03-resnet50

    dist_ap = torch.cat(dist_ap)
RuntimeError: zero-dimensional tensor (at position 0) cannot be concatenated

#0.3.1

python examples/triplet_loss.py -d cuhk03 -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/cuhk03-resnet50


Epoch: [149][21/21]     Time 0.659 (0.792)      Data 0.000 (0.150)      Loss 0.043 (0.033)      Prec 96.09% (97.53%)
Mean AP: 77.2%
CMC Scores    allshots      cuhk03  market1501
  top-1          62.4%       81.1%       82.3%
  top-5          74.3%       96.1%       90.6%
  top-10         80.3%       98.1%       93.9%

#1.0.0
Epoch: [149][21/21]     Time 0.626 (0.825)      Data 0.001 (0.166)      Loss 0.026 (0.026)      Prec 0.00% (0.00%)

Mean AP: 78.3%
CMC Scores    allshots      cuhk03  market1501
  top-1          63.2%       81.6%       81.5%
  top-5          76.0%       96.7%       89.9%
  top-10         81.9%       98.0%       93.6%


# mac
# The version of Anaconda may be different depending on when you are installing`
curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-MacOSX-x86_64.sh
sh Anaconda3-5.2.0-MacOSX-x86_64.sh
# and follow the prompts. The defaults are generally good.`

# mac for open re 
(base) Chenghungs-MacBook-Pro:repo chenghungyeh$ conda install pytorch torchvision -c pytorch
(base) Chenghungs-MacBook-Pro:repo chenghungyeh$ /anaconda3/bin/pip install metric_learn



python examples/triplet_loss.py --height 160 --width 160 -d album -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/album-resnet50

Running album.py
Traceback (most recent call last):
  File "examples/triplet_loss.py", line 220, in <module>
    main(parser.parse_args())
  File "examples/triplet_loss.py", line 144, in main
    trainer.train(epoch, train_loader, optimizer)
  File "/root/open-reid/reid/trainers.py", line 27, in train
    for i, inputs in enumerate(data_loader):
  File "/usr/local/lib/python2.7/dist-packages/torch/utils/data/dataloader.py", line 417, in __iter__
    return DataLoaderIter(self)
"  File "/usr/local/lib/python2.7/dist-packages/torch/utils/data/dataloader.py", line 242, in __init__
    self._put_indices()"
  File "/usr/local/lib/python2.7/dist-packages/torch/utils/data/dataloader.py", line 290, in _put_indices
    indices = next(self.sample_iter, None)
  File "/usr/local/lib/python2.7/dist-packages/torch/utils/data/sampler.py", line 119, in __iter__
    for idx in self.sampler:
  File "/root/open-reid/reid/utils/data/sampler.py", line 25, in __iter__
    indices = torch.randperm(self.num_samples)


python examples/triplet_loss.py --height 160 --width 160 -d viper -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/album-resnet50



python examples/triplet_loss.py --height 160 --width 160 -d album -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/album-resnet50


  File "examples/triplet_loss.py", line 220, in <module>
    main(parser.parse_args())
  File "examples/triplet_loss.py", line 144, in main
    trainer.train(epoch, train_loader, optimizer)
  File "/root/open-reid/reid/trainers.py", line 27, in train
    for i, inputs in enumerate(data_loader):
  File "/usr/local/lib/python2.7/dist-packages/torch/utils/data/dataloader.py", line 281, in __next__
    return self._process_next_batch(batch)
  File "/usr/local/lib/python2.7/dist-packages/torch/utils/data/dataloader.py", line 301, in _process_next_batch
    raise batch.exc_type(batch.exc_msg)
IOError: Traceback (most recent call last):
  File "/usr/local/lib/python2.7/dist-packages/torch/utils/data/dataloader.py", line 55, in _worker_loop
    samples = collate_fn([dataset[i] for i in batch_indices])
  File "/root/open-reid/reid/utils/data/preprocessor.py", line 20, in __getitem__
    return self._get_single_item(indices)
  File "/root/open-reid/reid/utils/data/preprocessor.py", line 27, in _get_single_item
    img = Image.open(fpath).convert('RGB')
  File "/usr/local/lib/python2.7/dist-packages/PIL/Image.py", line 2634, in open
    fp = builtins.open(filename, "rb")
IOError: [Errno 2] No such file or directory: u'/root/open-reid/examples/data/album/images/00006068_00_0006.jpg'

fixed


mbine-trainval --logs-dir examples/logs/triplet-loss/album-resnet50
Running album.py
/Users/chenghungyeh/repo/facenet_data/datasets/orb_sample_result_train_160/*/*.jpg
Traceback (most recent call last):
  File "examples/triplet_loss.py", line 220, in <module>
    main(parser.parse_args())
  File "examples/triplet_loss.py", line 92, in main
    args.combine_trainval)
  File "examples/triplet_loss.py", line 29, in get_data
    dataset = datasets.create(name, root, split_id=split_id)
  File "/root/open-reid/reid/datasets/__init__.py", line 47, in create
    return __factory[name](root, *args, **kwargs)
  File "/root/open-reid/reid/datasets/album.py", line 27, in __init__
    self.load(num_val)
  File "/root/open-reid/reid/utils/data/dataset.py", line 54, in load
    .format(num))
ValueError: num_val exceeds total identities 0

bug
Extract Features: [1/2] Time 3.500 (3.500)  Data 3.228 (3.228)  
Extract Features: [2/2] Time 3.048 (3.274)  Data 0.000 (1.614)  
Traceback (most recent call last):
  File "examples/triplet_loss.py", line 220, in <module>
    main(parser.parse_args())
  File "examples/triplet_loss.py", line 147, in main
    top1 = evaluator.evaluate(val_loader, dataset.val, dataset.val)
  File "/root/open-reid/reid/evaluators.py", line 120, in evaluate
    return evaluate_all(distmat, query=query, gallery=gallery)
  File "/root/open-reid/reid/evaluators.py", line 82, in evaluate_all
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
  File "/root/open-reid/reid/evaluation_metrics/ranking.py", line 114, in mean_ap
    raise RuntimeError("No valid query")
RuntimeError: No valid query




