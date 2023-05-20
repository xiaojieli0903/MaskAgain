import random

import numpy as np


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = f"TubeMaskingGenerator: " \
                   f"total patches {self.total_patches}, " \
                   f"total masks {self.total_masks}"
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.total_masks = int(mask_ratio * self.total_patches)

    def __repr__(self):
        repr_str = f"RandomMaskingGenerator: " \
                   f"total patches {self.total_patches}, " \
                   f"total masks {self.total_masks}"
        return repr_str

    def __call__(self):
        mask_per_seq = np.hstack([
            np.zeros(self.total_patches - self.total_masks),
            np.ones(self.total_masks),
        ])
        np.random.shuffle(mask_per_seq)
        return mask_per_seq


class TimeMaskingGenerator:
    def __init__(self, input_size, mask_ratio, context_ratio=0.25):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_context_patches = int(
            self.frames * context_ratio * self.num_patches_per_frame)
        self.num_future_patches = self.total_patches - self.num_context_patches
        self.total_masks = int(self.num_future_patches * mask_ratio)

    def __repr__(self):
        repr_str = f"TimeMaskingGenerator: " \
                   f"total patches {self.total_patches}, " \
                   f"context patches {self.total_patches}, " \
                   f"future patches {self.total_patches}, " \
                   f"context masks 0, " \
                   f"future masks {self.total_masks}"
        return repr_str

    def __call__(self):
        mask_future_seq = np.hstack([
            np.zeros(self.num_future_patches - self.total_masks),
            np.ones(self.total_masks),
        ])
        np.random.shuffle(mask_future_seq)
        mask_context = np.zeros(self.num_context_patches)
        return np.concatenate((mask_context, mask_future_seq), axis=0)
    
    
class TimeMaskingSplitGenerator:
    def __init__(self, input_size, mask_ratio, context_ratio=0.25):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_context_patches = int(
            self.frames * context_ratio * self.num_patches_per_frame)
        self.num_future_patches = self.total_patches - self.num_context_patches
        self.total_masks = int(self.num_future_patches * mask_ratio)

    def __repr__(self):
        repr_str = f"TimeMaskingGenerator: " \
                   f"total patches {self.total_patches}, " \
                   f"context patches {self.total_patches}, " \
                   f"future patches {self.total_patches}, " \
                   f"context masks 0, " \
                   f"future masks {self.total_masks}"
        return repr_str

    def __call__(self):
        mask_future_seq = np.hstack([
            np.zeros(self.num_future_patches - self.total_masks),
            np.ones(self.total_masks),
        ])
        np.random.shuffle(mask_future_seq)
        mask_all = np.concatenate((np.zeros(self.num_context_patches),
                                   mask_future_seq), axis=0)
        mask_context = np.concatenate((np.zeros(self.num_context_patches),
                                      np.ones(self.num_future_patches)),
                                      axis=0)
        mask_future = np.concatenate((np.ones(self.num_context_patches),
                                     mask_future_seq), axis=0)
        return np.concatenate((mask_all, mask_context, mask_future), axis=0)


class TimeDynamicMaskingGenerator:
    def __init__(self, input_size, mask_ratio=[0.85, 1.0], context_ratio=0.25):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        # context patches is fixed
        self.num_context_patches = int(
            self.frames * context_ratio * self.num_patches_per_frame)
        self.num_future_patches = self.total_patches - self.num_context_patches
        assert mask_ratio[0] <= mask_ratio[1]
        self.masks_range = [int(self.num_future_patches * mask_ratio[0]),
                            int(self.num_future_patches * mask_ratio[1])]
        self.trained_ratio = 0

    def __repr__(self):
        repr_str = f"TimeDynamicMaskingGenerator: " \
                   f"total patches {self.total_patches}, " \
                   f"context patches {self.num_context_patches}, " \
                   f"future patches {self.num_future_patches}, " \
                   f"context masks 0, " \
                   f"future masks in range {self.masks_range} "
        return repr_str

    def set_ratio(self, ratio):
        total_masks = self.masks_range[0] + (
                self.masks_range[1] - self.masks_range[0]) * self.trained_ratio
        self.trained_ratio = ratio
        print(f'trained ratio is set to {ratio}, total masks = {total_masks}')

    def __call__(self):
        total_masks = self.masks_range[0] + int(
            (self.masks_range[1] - self.masks_range[0]) * self.trained_ratio)
        mask_future_seq = np.hstack([
            np.zeros(self.num_future_patches - total_masks),
            np.ones(total_masks),
        ])
        np.random.shuffle(mask_future_seq)
        mask_context = np.zeros(self.num_context_patches)
        return np.concatenate((mask_context, mask_future_seq), axis=0)


class TimeDiffMaskingGenerator:
    def __init__(self,
                 input_size,
                 mask_ratio=[0.65, 0.85],
                 context_ratio=0.5,
                 split=False):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_context_patches = int(
            self.frames * context_ratio * self.num_patches_per_frame)
        self.num_future_patches = self.total_patches - self.num_context_patches
        self.context_masks = int(self.num_context_patches * mask_ratio[0])
        self.future_masks = int(self.num_future_patches * mask_ratio[1])
        self.total_masks = self.context_masks + self.future_masks
        self.split = split

    def __repr__(self):
        repr_str = f"TimeDiffMaskingGenerator: " \
                   f"total patches {self.total_patches}, " \
                   f"context patches {self.num_context_patches}, " \
                   f"future patches {self.num_future_patches}, " \
                   f"context masks {self.context_masks}, " \
                   f"future masks {self.future_masks}"
        return repr_str

    def __call__(self):
        mask_context_seq = np.hstack([
            np.zeros(self.num_context_patches - self.context_masks),
            np.ones(self.context_masks),
        ])
        np.random.shuffle(mask_context_seq)
        mask_future_seq = np.hstack([
            np.zeros(self.num_future_patches - self.future_masks),
            np.ones(self.future_masks),
        ])
        np.random.shuffle(mask_future_seq)
        mask = np.concatenate((mask_context_seq, mask_future_seq), axis=0)
        if self.split:
            mask_context = np.concatenate((mask_context_seq,
                                           np.ones(self.num_future_patches)),
                                          axis=0)
            return np.concatenate((mask, mask_context), axis=0)
        else:
            return mask


class FrameMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_frames = int(mask_ratio * self.frames)
        self.total_masks = self.num_masks_frames * self.num_patches_per_frame

    def __repr__(self):
        repr_str = f"TubeMaskingGenerator: " \
                   f"total patches {self.total_patches}, " \
                   f"total masks {self.total_masks}"
        return repr_str

    def __call__(self):
        frame_id = list(range(self.frames))
        np.random.shuffle(frame_id)
        choose_id = frame_id[:self.num_masks_frames]
        if 0 in choose_id:
            mask = np.ones(self.num_patches_per_frame)
        else:
            mask = np.zeros(self.num_patches_per_frame)

        for i in range(1, self.frames):
            if i in choose_id:
                mask = np.hstack([mask, np.ones(self.num_patches_per_frame)])
            else:
                mask = np.hstack([mask, np.zeros(self.num_patches_per_frame)])
        return mask


class TimeDiffMovingMaskingGenerator:
    def __init__(self,
                 input_size,
                 mask_ratio=[0.65, 0.85],
                 context_ratio=0.50,
                 split=False):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        dense_ratio = context_ratio
        self.num_dense_frames = int(self.frames * dense_ratio)
        self.num_sparse_frames = self.frames - self.num_dense_frames
        self.num_dense_patches = self.num_dense_frames * self.num_patches_per_frame
        self.num_sparse_patches = self.num_sparse_frames * self.num_patches_per_frame
        self.dense_masks = int(self.num_dense_patches * mask_ratio[0])
        self.sparse_masks = int(self.num_sparse_patches * mask_ratio[1])
        self.total_masks = self.dense_masks + self.sparse_masks
        self.split = split

    def __repr__(self):
        repr_str = f"TimeDiffMaskingGenerator: " \
                   f"total patches {self.total_patches}, " \
                   f"dense patches {self.num_dense_patches}, " \
                   f"sparse patches {self.num_sparse_patches}, " \
                   f"dense masks {self.dense_masks}, " \
                   f"sparse masks {self.sparse_masks}"
        return repr_str

    def __call__(self):
        frame_id = list(range(self.frames - self.num_dense_frames))
        np.random.shuffle(frame_id)
        dense_start_id = frame_id[0]
        print(f'dense start id: {dense_start_id}')
        mask_dense_seq = np.hstack([
            np.zeros(self.num_dense_patches - self.dense_masks),
            np.ones(self.dense_masks),
        ])
        np.random.shuffle(mask_dense_seq)
        mask_sparse_seq = np.hstack([
            np.zeros(self.num_sparse_patches - self.sparse_masks),
            np.ones(self.sparse_masks),
        ])
        np.random.shuffle(mask_sparse_seq)
        mask = np.concatenate((
            mask_sparse_seq[:dense_start_id * self.num_patches_per_frame],
            mask_dense_seq,
            mask_sparse_seq[dense_start_id * self.num_patches_per_frame:]),
            axis=0)
        if self.split:
            mask_context = np.concatenate((
                np.ones(dense_start_id * self.num_patches_per_frame),
                mask_dense_seq,
                np.ones(self.num_dense_patches - dense_start_id * self.num_patches_per_frame)),
                axis=0)
            return np.concatenate((mask, mask_context), axis=0)
        else:
            return mask


class RandomTimeDiffMaskingGenerator:
    def __init__(self,
                 input_size,
                 mask_ratio=[0.75, 0.65, 0.85],
                 context_ratio=0.5,
                 split=False,
                 random_prob=0.5):
        assert len(mask_ratio) == 3, f'random_mask_ratio | context_mask_ratio | future_mask_ratio'
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_context_patches = int(
            self.frames * context_ratio * self.num_patches_per_frame)
        self.num_future_patches = self.total_patches - self.num_context_patches
        self.context_masks = int(self.num_context_patches * mask_ratio[1])
        self.future_masks = int(self.num_future_patches * mask_ratio[2])
        self.total_masks = self.context_masks + self.future_masks
        self.random_total_masks = int(mask_ratio[0] * self.total_patches)
        self.split = split
        self.random_prob = random_prob

    def __repr__(self):
        repr_str = f"RandomTimeDiffMaskingGenerator: " \
                   f"total patches {self.total_patches}, " \
                   f"Random masked patches {self.random_total_masks}, " \
                   f"Random masked probability {self.random_prob}, " \
                   f"Timediff: context patches {self.num_context_patches}, " \
                   f"Timediff: future patches {self.num_future_patches}, " \
                   f"Timediff: context masks {self.context_masks}, " \
                   f"Timediff: future masks {self.future_masks}"
        return repr_str

    def __call__(self):
        prob = random.uniform(0., 1.)
        # random mask
        if prob < self.random_prob:
            mask_per_seq = np.hstack([
                np.zeros(self.total_patches - self.random_total_masks),
                np.ones(self.random_total_masks),
            ])
            np.random.shuffle(mask_per_seq)
            return mask_per_seq
        # timediff mask
        mask_context_seq = np.hstack([
            np.zeros(self.num_context_patches - self.context_masks),
            np.ones(self.context_masks),
        ])
        np.random.shuffle(mask_context_seq)
        mask_future_seq = np.hstack([
            np.zeros(self.num_future_patches - self.future_masks),
            np.ones(self.future_masks),
        ])
        np.random.shuffle(mask_future_seq)
        mask = np.concatenate((mask_context_seq, mask_future_seq), axis=0)
        if self.split:
            mask_context = np.concatenate((mask_context_seq,
                                           np.ones(self.num_future_patches)),
                                          axis=0)
            return np.concatenate((mask, mask_context), axis=0)
        else:
            return mask

class TeacherStuDiffMaskingGenerator:
    def __init__(self,
                 input_size,
                 mask_ratio=[0.75, 0.75]):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame
        if mask_ratio[1] < 1 - mask_ratio[0]:
            print('error:teacher and stu have the same visible patch ')
            eixt(0)
        self.num_stu_visible_patches_per_frame = int(self.num_patches_per_frame * (1 -mask_ratio[0]))
        self.num_teacher_visible_patches_per_frame = int(self.num_patches_per_frame * (1 -mask_ratio[1]))
        self.total_visible_patches_per_frame = self.num_stu_visible_patches_per_frame + self.num_teacher_visible_patches_per_frame


    def __repr__(self):
        repr_str = f"TeacherStuDiffMaskingGenerator: " \
                   f"total patches {self.total_patches}, " \
                   f"stu can see patches per frame {self.num_stu_visible_patches_per_frame}, " \
                   f"teacher can see patches per frame {self.num_teacher_visible_patches_per_frame}" 

        return repr_str

    def __call__(self):
        stu = np.ones(self.num_patches_per_frame)
        teacher = np.ones(self.num_patches_per_frame)
        patch_id = list(range(self.num_patches_per_frame))
        np.random.shuffle(patch_id)

        stu[patch_id[:self.num_stu_visible_patches_per_frame]] = 0 
        teacher[patch_id[self.num_stu_visible_patches_per_frame:self.total_visible_patches_per_frame]] = 0

        stu_mask = np.tile(stu, (self.frames, 1)).flatten()
        teacher_mask = np.tile(teacher, (self.frames, 1)).flatten()
        
        return [stu_mask, teacher_mask]


if __name__ == '__main__':

    mask = TeacherStuDiffMaskingGenerator([1,3,4]).__call__()
    print(mask[0])
    print(mask[1])
