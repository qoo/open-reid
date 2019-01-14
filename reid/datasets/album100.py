from __future__ import print_function, absolute_import
import os.path as osp
import os

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class Album100(Dataset):
    url = 'http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip'
    md5 = '1c2d9fc1cc800332567a0da25a1ce68c'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        print('Running album.py')
        super(Album100, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. " +
        #                        "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        # if self._check_integrity():
        #     print("Files already downloaded and verified")
        #     return

        import hashlib
        from glob import glob
        from scipy.misc import imsave, imread
        from six.moves import urllib
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'VIPeR.v1.0.zip')
        # if osp.isfile(fpath) and \
        #    hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
        #     print("Using downloaded file: " + fpath)
        # else:
        #     print("Downloading {} to {}".format(self.url, fpath))
        #     urllib.request.urlretrieve(self.url, fpath)

        # Extract the file
        # exdir = osp.join(raw_dir, 'VIPeR')
        # if not osp.isdir(exdir):
        #     print("Extracting zip file")
        #     with ZipFile(fpath) as z:
        #         z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)
        # cameras = [sorted(glob(osp.join(exdir, 'cam_a', '*.bmp'))),
        #            sorted(glob(osp.join(exdir, 'cam_b', '*.bmp')))]
        # assert len(cameras[0]) == len(cameras[1])
        identities = []
        # exdir: training dataset path
        # exdir='/Users/chenghungyeh/repo/facenet_data/datasets/orb_sample_result_train_160'
        exdir='/root/orb_sample_result_train_160'
        exdir='/root/orb_sample_result_train_all_160'


        input_dir=exdir
        input_filename = os.path.join(input_dir, '*')
        input_filename = os.path.join(input_filename, '*.jpg')
        print(input_filename)
        cluster_dict = {}
        cnt = 0
        for f in glob(input_filename):
            # print(f)
            filename = f.split('/')[-1]
            extention = f.split('.')[-1]
            prefix = f.split('.')[0]
            clusterID = prefix.split('/')[-2]
            albumID = prefix.split('/')[-1]
            if clusterID not in cluster_dict:
                cluster_dict[clusterID] = [albumID]
            else:
                cluster_dict[clusterID] += [albumID]
        #     cnt += 1
        #     if cnt > 5:
        #         break
        # print(cluster_dict)

        # cameras = sorted(glob(osp.join(exdir, '*', '*.jpg')))
        # 1 Album format
        # for pid, person in enumerate(cluster_dict):
        #     images = []
        #     # view-0
        #     photos=[]
        #     for image_id, image in enumerate(cluster_dict[person]):
        #         fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 0, image_id)
        #         # imsave(osp.join(images_dir, fname), imread(osp.join(exdir, person, image+ '.jpg')))
        #         photos.append(fname)
        #     images.append(photos)
        #     identities.append(images)

        # multi Album format
        for pid, person in enumerate(cluster_dict):
            identity = []
            for image_id, image in enumerate(cluster_dict[person]):
                camera = []
                fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, image_id, 0)
                # imsave(osp.join(images_dir, fname), imread(osp.join(exdir, person, image+ '.jpg')))
                camera.append(fname)
                identity.append(camera)
            identities.append(identity)

        # for pid, (cam1, cam2) in enumerate(zip(*cameras)):
        #     images = []
        #     # view-0
        #     fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 0, 0)
        #     imsave(osp.join(images_dir, fname), imread(cam1))
        #     images.append([fname])
        #     # view-1
        #     fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 1, 0)
        #     imsave(osp.join(images_dir, fname), imread(cam2))
        #     identities.append(images)

        # Save meta information into a json file
        meta = {'name': 'Album', 'shot': 'multiple', 'num_cameras': 1,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))
        # # Randomly create ten training and test split
        num = len(identities)
        splits = []
        for _ in range(10):
            pids = np.random.permutation(num).tolist()
            # trainval_pids = sorted(pids[:num // 2])
            # test_pids = sorted(pids[num // 2:])
            test_pids = sorted(pids[:num // 10])
            trainval_pids = sorted(pids[:])
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
