# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

## code from https://github.com/HyeonjeongHa/CRoZe/blob/main/zero_cost_methods/pruners/measures/jacob_cov.py

import torch
import numpy as np

from . import indicator


def get_batch_jacobian(net, x, target, device, split_data):
    # net.zero_grad()
    # x = torch.randn([16, 200, 49 * 3]).cuda()
    x.requires_grad_(True)

    N = x.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        y = net(x[st:en])
        if isinstance(y, tuple):
            y, _ = y
        y.backward(torch.ones_like(y))

    jacob = x.grad.detach()
    x.requires_grad_(False)
    return jacob, target.detach()

def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))

@indicator('jacob_cov', bn=True)
def compute_jacob_cov(net, inputs, targets, split_data=1, loss_fn=None, pretrained_model=None, pretrained=False):
    device = inputs.device
    net.zero_grad()
    # Compute gradients (but don't apply them)
    split_data=1
    # if pretrained and pretrained_model is not None:
    #     jacobs_pre, labels_pre = get_batch_jacobian(pretrained_model, inputs, targets, device, split_data=split_data)
    #     jacobs_pre = jacobs_pre.reshape(jacobs_pre.size(0), -1).cpu().numpy()
    jacobs, labels = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    try:
        # if pretrained:
        #     jc_pre = eval_score(jacobs_pre, labels_pre)
        # else:
        #     jc_pre = 0.0
        jc = eval_score(jacobs, labels)
        # print(jc, jc_pre)
    except Exception as e:
        print(e)
        jc = np.nan

    return jc
    # return round(abs(jc - jc_pre), 4)