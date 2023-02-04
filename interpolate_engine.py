from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from skimage.color import rgb2yuv, yuv2rgb
from utils.util import setup_logger, print_args
from utils.pytorch_msssim import ssim_matlab
from models.modules import define_G

class InterpolateEngine:
    def __init__(self, args : dict):
        self.args = args
        self.init_device()
        self.model = self.init_model()

    def init_device(self):
        str_ids = self.args.gpu_ids.split(',')
        self.args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.args.gpu_ids.append(id)
        if len(self.args.gpu_ids) > 0:
            torch.cuda.set_device(self.args.gpu_ids[0])

        cudnn.benchmark = True

    def init_model(self):
        args = self.args

        # defaults instead of args from original code that don't need setting
        args.crop_size = 192
        args.dist = False
        args.rank = -1
        args.phase = "test"
        args.resume_flownet = ""
        args.net_name = "VFIformer"

        ## load model
        device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
        args.device = device
        net = define_G(args)
        net = self.load_networks(net, args.model)
        net.eval()
        return net

    def load_networks(self, network, resume, strict=True):
        load_path = resume
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path, map_location=torch.device('cpu'))
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        if 'optimizer' or 'scheduler' in net_name:
            network.load_state_dict(load_net_clean)
        else:
            network.load_state_dict(load_net_clean, strict=strict)
        return network
