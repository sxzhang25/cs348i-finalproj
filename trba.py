"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from trba_modules.transformation import TPS_SpatialTransformerNetwork
from trba_modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from trba_modules.sequence_modeling import BidirectionalLSTM
from trba_modules.prediction import Attention


class TRBA(nn.Module):

  def __init__(self, opt):
    super(TRBA, self).__init__()
    self.opt = opt
    self.stages = {
      'Trans': 'TPS', 
      'Feat': 'ResNet',
      'Seq': 'BiLSTM', 
      'Pred': 'Attn'}

    # Transformation.
    self.Transformation = TPS_SpatialTransformerNetwork(
      F=opt.num_fiducial, 
      I_size=(opt.imgH, opt.imgW), 
      I_r_size=(opt.imgH, opt.imgW), 
      I_channel_num=opt.input_channel)

    # FeatureExtraction.
    self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
    self.FeatureExtraction_output = opt.output_channel  # int(imgH / 16 - 1) * 512
    self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH / 16 - 1) -> 1

    # Sequence modeling.
    self.SequenceModeling = nn.Sequential(
      BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
      BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
    self.SequenceModeling_output = opt.hidden_size

    # Prediction.
    self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)

  def forward(self, input, text, is_train=True):
    # print('trba input shape', input.shape)
    # Transformation stage.
    if not self.stages['Trans'] == "None":
      input = self.Transformation(input)
    # print('trba input shape', input.shape)

    # Feature extraction stage.
    visual_feature = self.FeatureExtraction(input)
    visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
    visual_feature = visual_feature.squeeze(3)
    # print('trba visual_feature shape', visual_feature.shape)

    # Sequence modeling stage.
    if self.stages['Seq'] == 'BiLSTM':
      contextual_feature = self.SequenceModeling(visual_feature)
    else:
      contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM
    # print('trba contextual_feature shape', contextual_feature.shape)

    # Prediction stage.
    if self.stages['Pred'] == 'CTC':
      prediction = self.Prediction(contextual_feature.contiguous())
    else:
      prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

    return prediction