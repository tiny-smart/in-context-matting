from torch import nn
from icm.util import instantiate_from_config

class InContextDecoder(nn.Module):
    '''
    InContextDecoder is the decoder of InContextMatting.

        in-context decoder:

        list get_trainable_params()

        forward(source, reference)
            reference = {'feature': feature_of_reference_image,
                    'guidance': guidance_on_reference_image}

            source = {'feature': feature_of_source_image, 'image': source_images}

    '''

    def __init__(self,
                 cfg_detail_decoder,
                 cfg_in_context_fusion,
                 freeze_in_context_fusion=False,
                 ):
        super().__init__()

        self.in_context_fusion = instantiate_from_config(
            cfg_in_context_fusion)
        self.detail_decoder = instantiate_from_config(cfg_detail_decoder)

        self.freeze_in_context_fusion = freeze_in_context_fusion
        if freeze_in_context_fusion:
            self.__freeze_in_context_fusion()

    def forward(self, source, reference):
        feature_of_reference_image = reference['feature']
        guidance_on_reference_image = reference['guidance']

        feature_of_source_image = source['feature']
        source_images = source['image']

        features = self.in_context_fusion(
            feature_of_reference_image, feature_of_source_image, guidance_on_reference_image)

        output = self.detail_decoder(features, source_images)

        return output, features['mask'], features['trimap']

    def get_trainable_params(self):
        params = []
        params = params + list(self.detail_decoder.parameters())
        if not self.freeze_in_context_fusion:
            params = params + list(self.in_context_fusion.parameters())
        return params

    def __freeze_in_context_fusion(self):
        for param in self.in_context_fusion.parameters():
            param.requires_grad = False
