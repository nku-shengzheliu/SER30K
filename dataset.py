# Code based on the Pyramid Vision Transformer
# https://github.com/whai362/PVT
# Licensed under the Apache License, Version 2.0
import os
import re
from os.path import join
import json
import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
from tqdm import tqdm

from torchvision.transforms import InterpolationMode
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from pytorch_pretrained_bert.tokenization import BertTokenizer

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from mcloader import ClassificationDataset

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class FI():
    def __init__(self, root, mode='train', data_len=None, transform=None):
        self.root = root
        self.mode = mode
        self.nb_classes = 8 # EmotionROI:6  FI:8
        self.transform = transform
            
        with open(join('other_dataset/FI', f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos

        self.imgs = [os.path.join(self.root, self.mode, anno[0]) for anno in
                            tqdm(self.annos[:data_len])]
        self.labels = [int(anno[1]) for anno in self.annos][:data_len]
        self.imgnames = [anno[0] for anno in self.annos]
    
    def __len__(self):
        return len(self.annos)
            
    def __getitem__(self, index):
        img_path, target, imgname = self.imgs[index], self.labels[index], self.imgnames[index]
        img = scipy.misc.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target #, imgname

class EmotionROI():
    def __init__(self, root, mode='train', data_len=None, transform=None):
        self.root = root
        self.mode = mode
        self.nb_classes = 6 # EmotionROI:6  FI:8
        self.transform = transform
            
        with open(join('other_dataset/EmotionROI', f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos

        self.imgs = [os.path.join(self.root, anno[1]) for anno in
                            tqdm(self.annos[:data_len])]
        self.labels = [int(anno[0]) for anno in self.annos][:data_len]
        self.imgnames = [anno[1] for anno in self.annos]
    
    def __len__(self):
        return len(self.annos)
            
    def __getitem__(self, index):
        img_path, target, imgname = self.imgs[index], self.labels[index], self.imgnames[index]
        img = scipy.misc.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target # , imgname

class SER():
    def __init__(self, root, mode='train', data_len=None, transform=None):
        self.root = root
        self.mode = mode
        self.nb_classes = 7
        self.transform = transform
            
        with open(join(root, 'Annotations', 'image-level', f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos['annotations']

        # self.imgs = [scipy.misc.imread(os.path.join(self.root, 'Images', anno['topic'], anno['file_name'])) for anno in
        #                     tqdm(self.annos[:data_len])]
        self.imgs = [os.path.join(self.root, 'Images', anno['topic'], anno['file_name']) for anno in
                            tqdm(self.annos[:data_len])]
        self.labels = [int(anno['anno']-1) for anno in self.annos][:data_len]
        self.imgnames = [join(anno['topic'], anno['file_name']) for anno in self.annos]
            
    def __getitem__(self, index):
        img_path, target, imgname = self.imgs[index], self.labels[index], self.imgnames[index]
        img = scipy.misc.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, target # , imgname

    def __len__(self):
        return len(self.annos)


class SER_Full():
    def __init__(self, root, mode='train', data_len=None, transform=None, max_query_len=30, bert_model='bert-base-uncased'):
        self.root = root
        self.mode = mode
        self.nb_classes = 7
        self.transform = transform
        self.query_len = max_query_len  # 30
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
            
        with open(join(root, 'Annotations', 'image-level', f'{self.mode}.json'), 'r') as f:
            annos = json.load(f)
        self.annos = annos['annotations']

        self.imgs = [os.path.join(self.root, 'Images', anno['topic'], anno['file_name']) for anno in
                            tqdm(self.annos[:data_len])]
        self.labels = [int(anno['anno']-1) for anno in self.annos][:data_len]
        self.imgnames = [join(anno['topic'], anno['file_name']) for anno in self.annos]
        self.sentences = [anno['text'] for anno in self.annos]
            
    def __getitem__(self, index):
        # Read Sample
        img_path, target, imgname, sentence = self.imgs[index], self.labels[index], self.imgnames[index], self.sentences[index]
        img = scipy.misc.imread(img_path)
        # Read Image
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        # Read Sentence
        sentence = sentence.lower()
        ## encode sentence to bert input
        examples = read_examples(sentence, index)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        #word_split = features[0].tokens[1:-1]

        return img, target, np.array(word_id, dtype=int), np.array(word_mask, dtype=int)#, imgname

    def __len__(self):
        return len(self.annos)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.dataset=='SER':
        dataset = SER_Full(root=args.data_path, mode='train' if is_train else 'test', transform=transform)
    elif args.dataset=='SER_V':
        dataset = SER(root=args.data_path, mode='train' if is_train else 'test', transform=transform)
    elif args.dataset=='FI':
        dataset = FI(root=args.data_path, mode='train' if is_train else 'test', transform=transform)
    elif args.dataset=='EmotionROI':
        dataset = EmotionROI(root=args.data_path, mode='train' if is_train else 'test', transform=transform)
    nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
