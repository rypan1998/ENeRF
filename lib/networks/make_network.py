import os
import imp


def make_network(cfg):
    module = cfg.network_module
    path = cfg.network_path
    # * lib.networks.enerf 中的文件封装了所有各种 Network() 函数，path 不同即可调用不同的网络，也是个小 trick 吧！
    network = imp.load_source(module, path).Network()
    return network
