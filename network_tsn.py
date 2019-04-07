import torch.nn as nn
import torchvision

class TSNClassifier(nn.Module):
    """A reimplementation of TSN model
        Args:
            num_classes: Number of classes
            num_segments: Number of segments
            base_model: backbone model for feature extraction
            dropout: dropout ratio of last fc layer
        """
    def __init__(self, num_classes, num_segments=3, base_model='resnet18', pretrained=True, dropout=0.8):
        super(TSNClassifier,self).__init__()
        self._prepare_base_model(base_model, pretrained)
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.dropout = dropout
        feature_dim = self._prepare_tsn(num_classes)
        self.consensus = lambda x: x.mean(dim=1)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
        self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        nn.init.normal_(self.new_fc.weight, 0, std)
        nn.init.constant_(self.new_fc.bias, 0)
        return feature_dim
    def _prepare_base_model(self, base_model, pretrained=True):

        if hasattr(torchvision.models, base_model):
            self.base_model = getattr(torchvision.models, base_model)(pretrained)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def forward(self, input):
        clip_len = input.size(2)
        channels = input.size(3)
        # reshape into batch of segments
        base_out = self.base_model(input.view((-1, clip_len*channels) + input.size()[-2:]))

        base_out = self.new_fc(base_out)
        # reshape back
        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = self.consensus(base_out)
        return output
