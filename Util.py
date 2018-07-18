import numpy as np

def check_bounds(indices, img):
    '''
    Corrects indices and returns a pad sequence that can be used to pad a cropped image
    to maintain the same dimensionality of the original_indices.
    '''

    flat_indices = np.concatenate(list(map(lambda x: [x.start, x.stop], indices)))
    before_pad = list(map(lambda x: 0 if x >= 0 else np.abs(x), flat_indices[::2]))
    max_diff = np.concatenate([np.diff((y,x)) for x,y in zip(img.shape, flat_indices[1::2])])
    after_pad = list(map(lambda x: 0 if x >= 0 else np.abs(x), max_diff))
    pad_sequence = np.stack((before_pad, after_pad), axis=1).tolist()
    correct_start = list(map(lambda x: max(0, x), flat_indices[::2]))
    correct_end = [min(x,y) for x,y in zip(img.shape, flat_indices[1::2])]
    correct_indices = zip(correct_start, correct_end)
    slice_indices = [slice(start,stop) for start,stop in correct_indices]

    return slice_indices, pad_sequence
