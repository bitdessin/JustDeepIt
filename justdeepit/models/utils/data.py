import json
from justdeepit.models.abstract import JDIError


class DataClass():

    def __init__(self, class_labels):
        self.class_labels = self.__set_class_labels(class_labels)

    def __len__(self):
        return len(self.class_labels)

    def __getitem__(self, i):
        if isinstance(i, int) or isinstance(i, str):
            return self.__getitem(i)
        elif isinstance(i, list) or isinstance(i, tuple):
            return [self.__getitem(_) for _ in i]
        else:
            raise TypeError(f'Invalid type: {type(i)} '
                            f'The argument must be int, str, list, or tuple.')

    def __getitem(self, i):
        if isinstance(i, int):
            return self.class_labels[i]
        elif isinstance(i, str):
            return self.class_labels.index(i)
    
    def __set_class_labels(self, class_labels):
        cl = None
        if class_labels is None:
            raise JDIError('The `class_labels` is required to build a model.')
        else:
            if isinstance(class_labels, list):
                cl = class_labels
            elif isinstance(class_labels, str):
                cl = []
                with open(class_labels, 'r') as infh:
                    for cl_ in infh:
                        cl.append(cl_.replace('\n', ''))
            else:
                raise JDIError('Unsupported data type of `class_labels`. '
                               'Set a path to a file containing a list of class labels '
                               'or set a list of class labels.')
        return cl



class DataPipeline():

    def __init__(self, with_bbox=True, with_mask=False):
        self.train = [
            dict(type='LoadImageFromFile',
                 backend_args=None),
            dict(type='LoadAnnotations',
                 with_bbox=with_bbox,
                 with_mask=with_mask),
            dict(type='Resize',
                 scale=(1333, 800),
                 keep_ratio=True),
            dict(type='RandomFlip',
                 prob=0.5),
            dict(type='PackDetInputs')
        ]
        self.valid = [
            dict(type='LoadImageFromFile',
                 backend_args=None),
            dict(type='Resize',
                 scale=(1333, 800),
                 keep_ratio=True),
            dict(type='LoadAnnotations',
                 with_bbox=with_bbox,
                 with_mask=with_mask),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id',
                           'img_path',
                           'ori_shape',
                           'img_shape',
                           'scale_factor'))
        ]
        self.test = self.valid
        self.inference = self.valid



class DataLoader():
    def __init__(self,
                 cfg,
                 train, valid=None, test=None,
                 batchsize=8, epoch=100, cpu=8,
                 with_bbox=False, with_mask=False):
        
        self.epoch = epoch
        self.batchsize = batchsize
        self.cpu = cpu
        self.cfg = cfg

        self.pipeline = DataPipeline(with_bbox, with_mask)

        self.__check_metainfo(self.cfg['metainfo']['classes'], train)
        self.cfg.update(self.__set_train_dataloader(train, self.pipeline.train))
        self.cfg.update(self.__set_valid_dataloader(valid, self.pipeline.valid))
        self.cfg.update(self.__set_test_dataloader(test, self.pipeline.test))



    def __check_metainfo(self, cfg_classes, dataset):
        with open(dataset['annotations'], 'r') as infh:
            dataset = json.load(infh)
        coco_classes = [c['name'] for c in dataset['categories']]

        for i in range(len(cfg_classes)):
            assert cfg_classes[i] == coco_classes[i], \
                'The class name in metainfo is not consistent with annotation file.'



    def __set_train_dataloader(self, dataset, pipeline):
        assert dataset is not None, 'The training dataset is required.'
        return dict(
            dataset_type='CocoDataset',
            train_dataloader=dict(
                batch_size=self.batchsize,
                num_workers=self.cpu,
                dataset=self.__set_train_dataset_cfg(dataset, pipeline),
            ),
            train_cfg = dict(
                type='EpochBasedTrainLoop',
                max_epochs=self.epoch,
                val_interval=1,
            ),
        )



    def __set_valid_dataloader(self, dataset, pipeline):
        if dataset is not None:
            return dict(
                val_dataloader=dict(
                    batch_size=self.batchsize,
                    num_workers=self.cpu,
                    dataset=self.__set_dataset_cfg(dataset, pipeline),
                ),
                val_cfg = dict(type='ValLoop'),
                val_evaluator = dict(
                    type='CocoMetric',
                    ann_file=dataset['annotations'],
                    metric='bbox',
                    backend_args=None
                )
            )
        else:
            return dict(val_dataloader=None,
                        val_cfg=None,
                        val_evaluator=None)



    def __set_test_dataloader(self, dataset, pipeline):
        if dataset is not None:
            return dict(
                test_dataloader=dict(
                    batch_size=self.batchsize,
                    num_workers=self.cpu,
                    dataset=self.__set_dataset_cfg(dataset, pipeline),
                ),
                test_cfg = dict(type='ValLoop'),
                test_evaluator = dict(
                    type='CocoMetric',
                    ann_file=dataset['annotations'],
                    metric='bbox',
                    backend_args=None
                )
            )
        else:
            return dict(test_dataloader=None,
                        test_cfg=None,
                        test_evaluator=None)
    
    
 
    def __set_train_dataset_cfg(self, dataset, pipeline):
        if self.cfg.train_dataloader.dataset.type == 'RepeatDataset':
            return dict(
                    type='RepeatDataset',
                    times=1,
                    dataset=self.__set_dataset_cfg(dataset, pipeline),
                )
        else:
            # some architecture does not support RepeatedDataset
            return self.__set_dataset_cfg(dataset, pipeline)



    def __set_dataset_cfg(self, dataset, pipeline=None):
        return dict(
            type='CocoDataset',
            data_root='',
            metainfo=self.cfg['metainfo'],
            ann_file=dataset['annotations'],
            data_prefix=dict(img=dataset['images']),
            pipeline=pipeline
        )
