from .model import Model

from .alexnet import AlexNet_SS
from .densenet import DenseNet_SS
from .inceptionv3 import Inceptionv3_SS
from .lenet5 import LeNet5_SS
from .linear import Linear_SS
from .mobilenetv2 import MobileNet_SS
from .repvgg import RepVGG_SS
from .resnet import ResNet_SS
from .shufflenet import ShuffleNet_SS
from .vgg import VGG_SS


def get_model(name, task, num_ss_classes, device, load_best_fc=True, dataset_name="cifar10", **kwargs):
    """

    :param dataset_name: The name of the dataset.
    :param name: The name of the backbone model
    :param task: The self-supervision task
    :param num_ss_classes: The number of classes in the self-supervision task
    :param device: The device the model should run on
    :param load_best_fc: True if the best model weights should be loaded, and false otherwise.
    :return: Instance of the model, with backbone model weights preloaded.
    """
    kwargs["pretrained"] = True
    kwargs["dataset"] = dataset_name
    kwargs["trust_repo"] = True
    kwargs["num_ss_classes"] = num_ss_classes
    if "resnet" in name:
        version = int(name.replace("resnet", ""))
        del kwargs["dataset"]
        model = ResNet_SS(version, **kwargs)
    elif "vgg" in name and "_bn" in name:
        version = int(name.split("_")[0][3:])
        del kwargs["dataset"]
        model = VGG_SS(version, **kwargs)
    elif name == "repvgg":
        del kwargs["dataset"]
        model = RepVGG_SS(**kwargs)
    elif name == "mobilenetv2":
        del kwargs["dataset"]
        model = MobileNet_SS(**kwargs)
    elif "densenet" in name:
        version = int(name.replace("densenet", ""))
        model = DenseNet_SS(version, **kwargs)
    elif name == "shufflenet":
        del kwargs["dataset"]
        model = ShuffleNet_SS(**kwargs)
    elif name == "inception_v3":
        model = Inceptionv3_SS(**kwargs)
    elif name == "linear":
        model = Linear_SS(**kwargs)
    elif name == "alexnet":
        del kwargs["dataset"]
        model = AlexNet_SS(**kwargs)
    elif name == "lenet5":
        model = LeNet5_SS(**kwargs)
    else:
        # Absolutely impossible case since this is covered by argparse.
        # If this NameError occurs please check the choices in the arg parser.
        raise NameError(f"Model name {name} does not exist.")

    if load_best_fc:
        # If we are not training the self-supervision FC layer, we should try to load in its best checkpoint
        model.load_ss_fc(f"../model_weights/{name}/{dataset_name}/{task}/best.pt", is_local=True)

    model.to(device)
    return model
