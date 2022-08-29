import torchvision.models as models
from fastai.vision import models as fai_models
import torch

def efficientnet_b7(num_classes, pretrained=True):
    model = models.efficientnet_b7(pretrained=pretrained, progress=False)
    linear = torch.nn.Linear(model._modules['classifier'][1].in_features, num_classes, bias=True)
    model._modules['classifier'] = torch.nn.Sequential(model._modules['classifier'][0], linear)
    return model


def efficientnet_b6(num_classes, pretrained=True):
    model = models.efficientnet_b6(pretrained=pretrained, progress=False)
    linear = torch.nn.Linear(model._modules['classifier'][1].in_features, num_classes, bias=True)
    model._modules['classifier'] = torch.nn.Sequential(model._modules['classifier'][0], linear)
    return model


def efficientnet_b5(num_classes, pretrained=True):
    model = models.efficientnet_b5(pretrained=pretrained, progress=False)
    linear = torch.nn.Linear(model._modules['classifier'][1].in_features, num_classes, bias=True)
    model._modules['classifier'] = torch.nn.Sequential(model._modules['classifier'][0], linear)
    return model


def iv3(num_classes, pretrained=True):
    model = models.inception_v3(pretrained=pretrained, progress=False, aux_logits=False)
    model._modules['fc'] = torch.nn.Linear(2048, num_classes, bias=True)
    return model


def resnet152(num_classes, pretrained=True):
    model = models.resnet152(pretrained=pretrained, progress=False)
    model._modules['fc'] = torch.nn.Linear(2048, num_classes, bias=True)
    return model


def resnet50(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained, progress=False)
    model._modules['fc'] = torch.nn.Linear(2048, num_classes, bias=True)
    return model


def resnet34(num_classes, pretrained=True):
    model = models.resnet34(pretrained=pretrained, progress=False)
    model._modules['fc'] = torch.nn.Linear(512, num_classes, bias=True)
    return model


def resnext101(num_classes, pretrained=True):
    model = models.resnext101_32x8d(pretrained=pretrained, progress=False)
    model._modules['fc'] = torch.nn.Linear(2048, num_classes, bias=True)
    return model


def resnet152unet(sample_size, pretrained=True):
    m = torch.nn.Sequential(*list(fai_models.resnet152(pretrained=pretrained, progress=False).children())[:-2])
    model = fai_models.unet.DynamicUnet(m, 1, [sample_size, sample_size], self_attention=True)
    return model


def densenet161(num_classes, pretrained=True):
    model = models.densenet161(pretrained=pretrained, progress=False)
    model._modules['classifier'] = torch.nn.Linear(2208, num_classes, bias=True)
    return model


def densenet169(num_classes, pretrained=True):
    model = models.densenet169(pretrained=pretrained, progress=False)
    model._modules['classifier'] = torch.nn.Linear(1664, num_classes, bias=True)
    return model


def densenet201(num_classes, pretrained=True):
    model = models.densenet201(pretrained=pretrained, progress=False)
    model._modules['classifier'] = torch.nn.Linear(1920, num_classes, bias=True)
    return model


def load_state(model, path, device=None):
    model.load_state_dict(torch.load(path, map_location=device))
    _ = model.eval()
    return model


def clip_last_layer(model):
    return torch.nn.Sequential(*list(model.children())[:-1])


class MixModel(torch.nn.Module):
    def __init__(self, num_classes, m0, m1, add_linear=True, out0=None, out1=None):
        super().__init__()
                
        self._m0 = m0  # torch.nn.Sequential(*list(m0.children())[:-1])
        self._m1 = m1  # torch.nn.Sequential(*list(m1.children())[:-1])

        self.add_linear = add_linear
        self._fc = None
        if self.add_linear:
            self._fc = torch.nn.Linear(out0 + out1, num_classes, bias=True)
        else:
            self._fc = torch.nn.Identity()

    def forward(self, x):
        y0 = self._m0(x[0])
        y1 = self._m1(x[1])
        y = torch.cat((y0, y1), dim=1)
        return self._fc(torch.squeeze(y))


class RandomClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.parameter = torch.nn.Parameter(torch.Tensor([0]))
    
    def forward(self, x):
        return torch.rand(x.shape[0], self.num_classes) + self.parameter[0]
    
    
class DummyClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv2d(3, 30, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='leaky_relu')
        self.bn = torch.nn.BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.bn2 = torch.nn.BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.aap = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(30, num_classes)
        #self._m = torch.nn.Sequential(self.conv1, self.pool, self.aap, self.fc)
        
    def forward(self, x):
        #return self._m(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.bn2(x)
        x = self.aap(x)
        return self.fc(x)
