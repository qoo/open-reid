from __future__ import print_function, absolute_import
import os.path as osp
import os

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class AlbumPair(Dataset):
    url = 'http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip'
    md5 = '1c2d9fc1cc800332567a0da25a1ce68c'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(AlbumPair, self).__init__(root, split_id=split_id)

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

        # function
        def read_pairs(pairs_filename):
            pairs = []
            with open(pairs_filename, 'r') as f:
                for line in f.readlines()[1:]:
                    pair = line.strip().split()
                    pairs.append(pair)
            return np.array(pairs)

        def add_extension(path):
            if os.path.exists(path + '.jpg'):
                return path + '.jpg'
            elif os.path.exists(path + '.png'):
                return path + '.png'
            else:
                raise RuntimeError('No file "%s" with extension png or jpg.' % path)

        def get_paths(lfw_dir, pairs):
            nrof_skipped_pairs = 0
            path_list = []
            issame_list = []
            for pair in pairs:
                if len(pair) == 3:
                    path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                    path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
                    issame = True
                elif len(pair) == 4:
                    path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                    path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
                    issame = False
                if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                    path_list += (path0, path1)
                    issame_list.append(issame)
                else:
                    nrof_skipped_pairs += 1
            if nrof_skipped_pairs > 0:
                print('Skipped %d image pairs' % nrof_skipped_pairs)

            return path_list, issame_list
        # exdir: training dataset path
        # exdir='/Users/chenghungyeh/repo/facenet_data/datasets/orb_sample_result_train_160'
        exdir='/root/orb_sample_result_train_160'
        exdir='/root/orb_sample_result_train_all_160'
        exdir = '/Users/chenghungyeh/GoT_frame_160'
        exdir = '/root/GoT_frame_160'
        exdir = '/Users/chenghungyeh/repo/facenet_data/datasets/orb_new_result_160'
        exdir = '/root/orb_new_result_160'


        # Read validatoin data
        validate_dir = exdir

        lfw_pairs = '/Users/chenghungyeh/repo/facenet_data/pairs4.txt'
        lfw_pairs = '/root/pairs4.txt'

        lfw_dir = validate_dir
        pairs = read_pairs(os.path.expanduser(lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame0 = get_paths(os.path.expanduser(lfw_dir), pairs)


        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        cameras=[]
        cameras.append(lfw_paths[0::2])
        cameras.append(lfw_paths[1::2])

        assert len(cameras[0]) == len(cameras[1])
        minLen = min(len(cameras[0]), len(cameras[1]))

        print( 'Min lens of two sets is {:04d} photos'.format(minLen))

        identities = []
        for pid, (cam1, cam2) in enumerate(zip(*cameras)):
            images = []
            # view-0
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 0, 0)
            imsave(osp.join(images_dir, fname), imread(cam1))
            images.append([fname])
            # view-1
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 1, 0)
            imsave(osp.join(images_dir, fname), imread(cam2))
            images.append([fname])
            identities.append(images)

        # minLen = len(lfw_paths)
        # identities = []
        # for pid, cam2 in enumerate(lfw_paths):
        #     images = []
        #     # view-0
        #     fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 0, 0)
        #     imsave(osp.join(images_dir, fname), imread(cam1))
        #     images.append([fname])
        #     # view-1
        #     fname = '{:08d}_{:02d}_{:04d}.jpg'.format(pid, 0, 0)
        #     imsave(osp.join(images_dir, fname), imread(cam2))
        #     images.append([fname])
        #     identities.append(images)


        # Save meta information into a json file
        meta = {'name': 'Albumpair', 'shot': 'single', 'num_cameras': 2,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))
        # # Randomly create ten training and test split
        num = len(identities)
        splits = []
        pids = list(range(num))
        # print("pids")
        # print(pids[0:6])
        test_pids = sorted(pids[:])
        # print(test_pids[0:6])
        trainval_pids = sorted(pids[:])
        split = {'trainval': trainval_pids,
                 'query': test_pids,
                 'gallery': test_pids}
        splits.append(split)
        # for _ in range(10):
        #     pids = np.random.permutation(num).tolist()
        #     # trainval_pids = sorted(pids[:num // 2])
        #     # test_pids = sorted(pids[num // 2:])
        #     test_pids = sorted(pids[:])
        #     trainval_pids = sorted(pids[:])
        #     split = {'trainval': trainval_pids,
        #              'query': test_pids,
        #              'gallery': test_pids}
        #     splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))
