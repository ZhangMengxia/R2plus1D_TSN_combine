import os
from pathlib import Path
import glob
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list 
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            directory (str): The path to the directory containing the train/val/test datasets
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'. 
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 8. 
        """

    def __init__(self, directory, im_path_root=None, resize_width=171, resize_height=128, crop_size=112,
            mode='train', clip_len=8):
        #folder = Path(directory)/mode  # get the directory of the specified split
        folder = directory + '/' + mode
        self.clip_len = clip_len

        # the following three parameters are chosen as described in the paper section 4.1
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.crop_size = crop_size
        self.im_path_root = im_path_root
        self.mode=mode
        # obtain all the filenames of files inside all the class folders 
        # going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)     

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)        

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        buffer = self.loadvideoframe(self.fnames[index])
        buffer = self.crop(buffer, self.crop_size)
        buffer = self.normalize(buffer)

        return buffer, self.label_array[index]    
        
        
    def get_im_path_pattern(self, fname):
        vid_name = os.path.basename(fname)[:-4]
        vid_class = fname.split('/')[-2]
        return os.path.join(self.im_path_root, vid_name, 'img_*.jpg')
    def loadvideoframe(self, fname):
        im_path_pattern = self.get_im_path_pattern(fname)
        img_list = sorted(glob.glob(im_path_pattern))
        frame_count = len(img_list)
        #sample_im = cv2.imread(img_list[0])
        #frame_width = sample_im.shape[1]
        #frame_height = sample_im.shape[0]
        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        count = 0
        start_frame = np.random.randint(frame_count - self.clip_len+1, size=1)[0]
        # read in each frame, one at a time into the numpy buffer array
        while (count < self.clip_len):
            frame = cv2.imread(img_list[start_frame+count])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # will resize frames if not already final size
            # NOTE: strongly recommended to resize them during the download process. This script
            # will process videos of any size, but will take longer the larger the video file.
            #if (frame_height != self.resize_height) or (frame_width != self.resize_width):
            frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            buffer[count] = frame
            count += 1


        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        buffer = buffer.transpose((3, 0, 1, 2))

        return buffer 

    def loadvideo(self, fname):
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        print(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(frame_count, frame_width, frame_height)
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True

        # read in each frame, one at a time into the numpy buffer array
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # will resize frames if not already final size
            # NOTE: strongly recommended to resize them during the download process. This script
            # will process videos of any size, but will take longer the larger the video file.
            if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            buffer[count] = frame
            count += 1

        # release the VideoCapture once it is no longer needed
        capture.release()

        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        buffer = buffer.transpose((3, 0, 1, 2))

        return buffer 
    
    def crop(self, buffer, crop_size):
        # randomly select time index for temporal jittering
        #time_index = np.random.randint(buffer.shape[1] - clip_len)
        # randomly select start indices in order to crop the video
        if self.mode == 'train':
            height_index = np.random.randint(buffer.shape[-2] - crop_size)
            width_index = np.random.randint(buffer.shape[-1] - crop_size)
        else:
            height_index = (buffer.shape[-2] - crop_size) // 2
            width_index = (buffer.shape[-1] - crop_size) // 2

        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[...,
                        height_index:height_index + crop_size,
                        width_index:width_index + crop_size]

        return buffer                

    def normalize(self, buffer):
        # Normalize the buffer
        # NOTE: Default values of RGB images normalization are used, as precomputed 
        # mean and std_dev values (akin to ImageNet) were unavailable for Kinetics. Feel 
        # free to push to and edit this section to replace them if found. 
        buffer = (buffer - 128)/128
        return buffer

    def __len__(self):
        return len(self.fnames)


class VideoDataset1M(VideoDataset):
    r"""Dataset that implements VideoDataset, and produces exactly 1M augmented
    training samples every epoch.
        
        Args:
            directory (str): The path to the directory containing the train/val/test datasets
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'. 
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 8. 
        """
    def __init__(self, directory, im_path_root, mode='train', clip_len=8):
        # Initialize instance of original dataset class
        super(VideoDataset1M, self).__init__(directory, im_path_root, mode, clip_len)

    def __getitem__(self, index):
        # if we are to have 1M samples on every pass, we need to shuffle
        # the index to a number in the original range, or else we'll get an 
        # index error. This is a legitimate operation, as even with the same 
        # index being used multiple times, it'll be randomly cropped, and
        # be temporally jitterred differently on each pass, properly
        # augmenting the data. 
        index = np.random.randint(len(self.fnames))

        buffer = self.loadvideoframe(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)

        return buffer, self.label_array[index]    

    def __len__(self):
        return 1000000  # manually set the length to 1 million

class VideoDatasetTSN(VideoDataset):
    """Dataset for TSN-like data input
    Args:
        directory: path to the data containing train/val/test
        im_path_root: path to image frames
        mode: determines which folder to read from. Default to 'train'.
        segment_n: number of segment in a video
        clip_len: number of frames in a clip
        """
    def __init__(self, directory, im_path_root, resize_width=171, resize_height=128, crop_size=112,
            mode='train', segment_n=3, clip_len=8):
        super(VideoDatasetTSN, self).__init__(directory, im_path_root, resize_width, resize_height, 
                crop_size, mode, clip_len)
        self.segment_n = segment_n
    def loadvideoframe(self, fname):
        im_path_pattern = self.get_im_path_pattern(fname)
        img_list = sorted(glob.glob(im_path_pattern))
        frame_count = len(img_list)
        #sample_im = cv2.imread(img_list[0])
        #frame_width = sample_im.shape[1]
        #frame_height = sample_im.shape[0]
        buffer = np.empty((self.segment_n, self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        count = 0
        frame_per_segment = frame_count // self.segment_n
        start_frames = np.random.randint(frame_per_segment - self.clip_len+1, 
                size=self.segment_n)
        # read in each frame, one at a time into the numpy buffer array
        for segment_id, start_frame_off in enumerate(start_frames):
            for count in range(0, self.clip_len):
                start_frame = start_frame_off + segment_id * frame_per_segment 
                frame = cv2.imread(img_list[start_frame+count])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size
                # NOTE: strongly recommended to resize them during the download process. This script
                # will process videos of any size, but will take longer the larger the video file.
                #if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                buffer[segment_id, count] = frame


        # convert from [N, D, H, W, C] format to [N, C, D, H, W] (what PyTorch uses)
        # N = Number of segment, D = Depth (in this case, time), H = Height, W = Width, C = Channels
        buffer = buffer.transpose((0, 4, 1, 2, 3))

        return buffer 
