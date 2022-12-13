# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np


class ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    face_mesh -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    emotion -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    """
    def __init__(self, batch_size, face_mesh, emotion,
                 pad=0, shuffle=True, random_seed=1234):
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame) tuples
        for i in range(len(emotion)):
            bounds = np.arange(emotion[i].shape[0]+1)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:])
        # Initialize buffers
        self.batch_mesh = np.empty((batch_size, 1+2*pad, face_mesh[0].shape[-2], face_mesh[0].shape[-1]))
        self.batch_emotion = np.empty((batch_size, 1))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.state = None
        
        self.face_mesh = face_mesh
        self.emotion = emotion
        
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_emotion, end_emotion) in enumerate(chunks):
                    start_mesh = start_emotion - self.pad
                    end_mesh = end_emotion + self.pad

                    # mesh
                    seq_mesh = self.face_mesh[seq_i]
                    low_mesh = max(start_mesh, 0)
                    high_mesh = min(end_mesh, seq_mesh.shape[0])
                    pad_left_mesh = low_mesh - start_mesh
                    pad_right_mesh = end_mesh - high_mesh
                    if pad_left_mesh != 0 or pad_right_mesh != 0:
                        self.batch_mesh[i] = np.pad(seq_mesh[low_mesh:high_mesh],
                            ((pad_left_mesh, pad_right_mesh), (0, 0), (0, 0)), 'edge')
                    else:
                        self.batch_mesh[i] = seq_mesh[low_mesh:high_mesh]

                    # emotion
                    if self.emotion is not None:
                        seq_emotion = self.emotion[seq_i]
                        low_emotion = max(start_emotion, 0)
                        high_emotion = min(end_emotion, seq_emotion.shape[0])
                        pad_left_emotion = low_emotion - start_emotion
                        pad_right_emotion = end_emotion - high_emotion
                        
                        if pad_left_emotion != 0 or pad_right_emotion != 0:
                            self.batch_emotion[i] = np.pad(seq_emotion[low_emotion:high_emotion],
                                ((pad_left_emotion, pad_right_emotion), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_emotion[i] = seq_emotion[low_emotion:high_emotion]
                
                if self.emotion is None:
                    yield self.batch_mesh[:len(chunks)], None
                else:
                    yield self.batch_mesh[:len(chunks)], self.batch_emotion[:len(chunks)]
            
            enabled = False


class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    face_mesh -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    emotion -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    """
    
    def __init__(self, face_mesh, emotion, pad=0):
        self.pad = pad
        self.face_mesh = [] if face_mesh is None else face_mesh
        self.emotion = emotion
        
    def num_frames(self):
        count = 0
        for p in self.emotion:
            count += p.shape[0]
        return count
    
    def next_epoch(self):
        for seq_mesh, seq_emotion in zip_longest(self.face_mesh, self.emotion):
            batch_emotion = None if seq_emotion is None else np.expand_dims(seq_emotion, axis=0)
            batch_mesh = np.expand_dims(np.pad(seq_mesh,
                            ((self.pad, self.pad), (0, 0), (0, 0)),
                            'edge'), axis=0)
            yield batch_mesh, batch_emotion