from torchvision import datasets, transforms
import torch

from mgct.tinyimagenet import TinyImageNet


def _sigmoid_space_to_tanh(x):
    return (2*x) - 1


def _f(x):
    return _sigmoid_space_to_tanh(x)


def get_dataloader(args):
    normalizations = {'mnist': transforms.Normalize((0.1307,), (0.3081,)),
                      'svhn': transforms.Normalize((0.43768206, 0.44376972, 0.47280434),
                                                   (0.19803014, 0.20101564, 0.19703615)),
                      'cifar10': transforms.Normalize((_f(0.4914), _f(0.4822), _f(0.4465)),
                                                      (2*0.2023, 2*0.1994, 2*0.2010)),
                      'cifar100': transforms.Normalize((_f(0.4914), _f(0.4822), _f(0.4465)),
                                                       (2*0.2023, 2*0.1994, 2*0.2010)),
                      'tinyimagenet': transforms.Normalize((_f(0.485), _f(0.456), _f(0.406)),
                                                           (2 * 0.229, 2 * 0.224, 2 * 0.225))
                      }
    if args.input_space == "pre-transform":
        pre_normalization = torch.nn.Identity()
        post_normalization = normalizations[args.dataset.lower()]
    elif args.input_space == "post-transform":
        pre_normalization = normalizations[args.dataset.lower()]
        post_normalization = torch.nn.Identity()
    if args.dataset.lower()=='mnist':
        train_loader = torch.utils.data.DataLoader( 
            datasets.MNIST(args.data_root, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.1307,), (0.3081,))
                               pre_normalization
                           ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.MNIST(args.data_root, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.1307,), (0.3081,))
                               pre_normalization
                           ])),
            batch_size=args.batch_size, shuffle=False, num_workers=2)

    elif args.dataset.lower()=='svhn':
        print("Loading SVHN data")
        train_loader = torch.utils.data.DataLoader( 
            datasets.SVHN(args.data_root, split='train', download=True,
                          transform=transforms.Compose([
                              transforms.Resize((32, 32)),
                              transforms.ToTensor(),
                              #transforms.Normalize((0.43768206, 0.44376972, 0.47280434), (0.19803014, 0.20101564, 0.19703615)),
                              # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                              pre_normalization
                          ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.SVHN(args.data_root, split='test', download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              #transforms.Normalize((0.43768206, 0.44376972, 0.47280434), (0.19803014, 0.20101564, 0.19703615)),
                              # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                              pre_normalization
                          ])),
            batch_size=args.batch_size, shuffle=False, num_workers=2)

    elif args.dataset.lower()=='cifar10':
        # Matched with Data-Free Adversarial Distillation, not true std
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 _sigmoid_space_to_tanh,
                                 #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                 pre_normalization
                             ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(args.data_root, train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                _sigmoid_space_to_tanh,
                                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                pre_normalization
                            ])),
            batch_size=args.batch_size, shuffle=False, num_workers=2)

    elif args.dataset.lower()=='cifar100':
        # Matched with Data-Free Adversarial Distillation, not true mean/std
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_root, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  _sigmoid_space_to_tanh,
                                  #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  pre_normalization
                              ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.data_root, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  _sigmoid_space_to_tanh,
                                  #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  pre_normalization
                              ])),
            batch_size=args.batch_size, shuffle=False, num_workers=2)

    # elif args.dataset.lower() == 'tinyimagenet':
    #     print("Loading TinyImageNet data")
    #     train_loader = torch.utils.data.DataLoader(
    #         TinyImageNet(args.data_root, train=True, transform=transforms.Compose([
    #             transforms.Resize((32, 32)),  # TinyImageNet是64x64的尺寸
    #             transforms.ToTensor(),
    #             pre_normalization
    #         ])),
    #         batch_size=args.batch_size, shuffle=True, num_workers=2)
    #
    #     test_loader = torch.utils.data.DataLoader(
    #         TinyImageNet(args.data_root, train=False, transform=transforms.Compose([
    #             transforms.Resize((32, 32)),
    #             transforms.ToTensor(),
    #             pre_normalization
    #         ])),
    #         batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset.lower() == 'tinyimagenet':
        # 使用TinyImageNet专用归一化参数
        tinyimagenet_mean = [0.480, 0.448, 0.398]
        tinyimagenet_std = [0.277, 0.269, 0.282]
        pre_normalization = transforms.Normalize(tinyimagenet_mean, tinyimagenet_std)

        # 增强的训练数据转换
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            pre_normalization
        ])

        # 测试数据转换
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            pre_normalization
        ])

        train_loader = torch.utils.data.DataLoader(
            TinyImageNet(args.data_root, train=True, transform=train_transform),
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            TinyImageNet(args.data_root, train=False, transform=test_transform),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, post_normalization
