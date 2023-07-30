from .model import Model

from .alexnet import AlexNet_SS
from .densenet import DenseNet_SS
from .inceptionv3 import Inceptionv3_SS
from .lenet5 import LeNet5_SS
from .linear import Linear_SS
from .mobilenetv2 import MobileNet_SS
from .obc import OBC_SS
from .repvgg import RepVGG_SS
from .resnet import ResNet_SS
from .shufflenet import ShuffleNet_SS


def get_model(name, task, num_ss_classes, device, load_best_fc=True):
    """

    :param name: The name of the backbone model
    :param task: The self-supervision task
    :param num_ss_classes: The number of classes in the self-supervision task
    :param device: The device the model should run on
    :param load_best_fc: True if the best model weights should be loaded, and false otherwise.
    :return: Instance of the model, with backbone model weights preloaded.
    """

    if "resnet" in name:
        version = int(name.replace("resnet", ""))
        model = ResNet_SS(version, num_ss_classes)
    elif name == "repvgg":
        model = RepVGG_SS(num_ss_classes)
    elif name == "mobilenetv2":
        model = MobileNet_SS(num_ss_classes)
    elif "densenet" in name:
        version = int(name.replace("densenet", ""))
        model = DenseNet_SS(version, num_ss_classes)
    elif name == "shufflenet":
        model = ShuffleNet_SS(num_ss_classes)
    elif name == "inception_v3":
        model = Inceptionv3_SS(num_ss_classes)
    elif name == "linear":
        model = Linear_SS(num_ss_classes)
    elif name == "alexnet":
        model = AlexNet_SS(num_ss_classes)
    elif name == "lenet5":
        model = LeNet5_SS(num_ss_classes)
    elif name == "obc":
        model = OBC_SS(num_ss_classes, device)
    else:
        # Absolutely impossible case since this is covered by argparse.
        # If this NameError occurs please check the choices in the arg parser.
        raise NameError(f"Model name {name} does not exist.")

    if load_best_fc:
        # If we are not training the self-supervision FC layer, we should try to load in its best checkpoint
        model.load_ss_fc(f"../model_weights/{name}/{task}/best.pt", is_local=True)

    model.to(device)
    return model
