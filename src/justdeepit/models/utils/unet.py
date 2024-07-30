import torch
import torch.nn as nn


class DualCov(nn.Module):
    
    def __init__(self, input_channels, output_channels, middle_channels=None):
        super().__init__()
        
        middle_channels = output_channels if middle_channels is None
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)




class UNetArch(nn.Module):
    def __init__(self, n_classes, input_channels=3, **kwargs):
        super().__init__()

        n_ch = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.input = DualCov(input_channels, n_ch[0], n_ch[0])
        self.dw1 = DualCov(n_ch[0], n_ch[1], n_ch[1])
        self.dw2 = DualCov(n_ch[1], n_ch[2], n_ch[2])
        self.dw3 = DualCov(n_ch[2], n_ch[3], n_ch[3])
        self.dw4 = DualCov(n_ch[3], n_ch[4], n_ch[4])
        self.up1 = DualCov(n_ch[3] + n_ch[4], n_ch[3], n_ch[3])
        self.up2 = DualCov(n_ch[2] + n_ch[3], n_ch[2], n_ch[2])
        self.up3 = DualCov(n_ch[1] + n_ch[2], n_ch[1], n_ch[1])
        self.up4 = DualCov(n_ch[0] + n_ch[1], n_ch[0], n_ch[0])
        self.output = nn.Conv2d(n_ch[0], n_classes, kernel_size=1)


    def forward(self, x):
        x0 = self.input(x)
        x1 = self.dw1_0(self.pool(x0))
        x2 = self.dw2_0(self.pool(x1))
        x3 = self.dw3_0(self.pool(x2))
        x4 = self.dw4_0(self.pool(x3))
        x = self.up1(torch.cat([x3, self.up(x4)], 1))
        x = self.up2(torch.cat([x2, self.up(x)], 1))
        x = self.up3(torch.cat([x1, self.up(x)], 1))
        x = self.up4(torch.cat([x0, self.up(x)], 1))
        output = self.output(x0_4)
        return output



class NestedUNetArch(nn.Module):
    def __init__(self, n_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        n_ch = [64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.input = DualCov(input_channels, n_ch[0], n_ch[0])
        
        self.conv1_0 = DualCov(n_ch[0], n_ch[1], n_ch[1])
        self.conv2_0 = DualCov(n_ch[1], n_ch[2], n_ch[2])
        self.conv3_0 = DualCov(n_ch[2], n_ch[3], n_ch[3])
        self.conv4_0 = DualCov(n_ch[3], n_ch[4], n_ch[4])
        self.conv0_1 = DualCov(n_ch[0] + n_ch[1], n_ch[0], n_ch[0])
        self.conv1_1 = DualCov(n_ch[1] + n_ch[2], n_ch[1], n_ch[1])
        self.conv2_1 = DualCov(n_ch[2] + n_ch[3], n_ch[2], n_ch[2])
        self.conv3_1 = DualCov(n_ch[3] + n_ch[4], n_ch[3], n_ch[3])
        self.conv0_2 = DualCov(n_ch[0] * 2 + n_ch[1], n_ch[0], n_ch[0])
        self.conv1_2 = DualCov(n_ch[1] * 2 + n_ch[2], n_ch[1], n_ch[1])
        self.conv2_2 = DualCov(n_ch[2] * 2 + n_ch[3], n_ch[2], n_ch[2])
        self.conv0_3 = DualCov(n_ch[0] * 3 + n_ch[1], n_ch[0], n_ch[0])
        self.conv1_3 = DualCov(n_ch[1] * 3 + n_ch[2], n_ch[1], n_ch[1])
        self.conv0_4 = DualCov(n_ch[0] * 4 + n_ch[1], n_ch[0], n_ch[0])

        if self.deep_supervision:
            self.output1 = nn.Conv2d(n_ch[0], n_classes, kernel_size=1)
            self.output2 = nn.Conv2d(n_ch[0], n_classes, kernel_size=1)
            self.output3 = nn.Conv2d(n_ch[0], n_classes, kernel_size=1)
            self.output4 = nn.Conv2d(n_ch[0], n_classes, kernel_size=1)
        else:
            self.output = nn.Conv2d(n_ch[0], n_classes, kernel_size=1)


    def forward(self, x):
        x0_0 = self.input(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            return [self.output1(x0_1), self.output2(x0_2), self.output3(x0_3), self.output4(x0_4)]
        else:
            return self.final(x0_4)





class UNet(ModuleTemplate):
    
    
    def __init__(self, model_weight=None, workspace=None):
        # load model
        self.model = UNetArch(3, 1)
        if model_weight is not None:
            self.model.load_state_dict(torch.load(model_weight, map_location='cpu'))

        # workspace
        self.tempd = None
        if workspace is None:
            self.tempd = tempfile.TemporaryDirectory()
            self.workspace = self.tempd.name
        else:
            self.workspace = workspace
        logger.info('Set workspace at `{}`.'.format(self.workspace))


        # transform
        self.transform_train = None
        self.transform_valid = None

    
    
    def train(self, train_data_fpath, batchsize=8, epoch=10000, lr=0.001, cpu=1, gpu=1):
        
        



def train:
    #config = vars(parse_args())
    #if config['name'] is None:
    #os.makedirs('models/%s' % config['name'], exist_ok=True)
    #with open('models/%s/config.yml' % config['name'], 'w') as f:
    #    yaml.dump(config, f)

    criterion = nn.BCEWithLogitsLoss().cuda()
    cudnn.benchmark = True

    # create model
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    
    model = model.cuda()
    optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    
    
    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batchsize=config['batchsize'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batchsize=config['batchsize'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()



