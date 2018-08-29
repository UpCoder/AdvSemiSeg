from torch import nn
from convolution_lstm import ConvLSTMCell, ConvLSTM
import torch
from torch.autograd import Variable


class BiCLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(BiCLSTM, self).__init__()
        self.forward_clstm = ConvLSTM(input_channels, hidden_channels, kernel_size, step=step,
                                      effective_step=effective_step, bias=bias, name='forward')
        self.backward_clstm = ConvLSTM(input_channels, hidden_channels, kernel_size, step=step,
                                       effective_step=effective_step, bias=bias, name='backward')
        self.step = step
        self.mid_pos = self.step // 2
        self.feature_select = nn.Conv2d(hidden_channels[-1] * 2, hidden_channels[-1], kernel_size=1, stride=1,
                                        padding=0, bias=False)
    def forward(self, input):
        forward_outputs, _ = self.forward_clstm(input)

        splits = torch.split(input, split_size_or_sections=1, dim=1)
        input_reverse = splits[::-1]
        input_reverse = torch.cat(input_reverse, dim=1)
        backward_outputs, _ = self.backward_clstm(input_reverse)

        # forward_output = forward_outputs[self.mid_pos]
        # backward_output = backward_outputs[self.mid_pos]

        forward_output = forward_outputs[-1]
        backward_output = backward_outputs[-1]

        _, channel, _, _ = backward_output.size()
        output = torch.cat([forward_output, backward_output], dim=1)
        output = self.feature_select(output)

        # print('forward_outputs len: ', len(forward_outputs), ' shape of that is ', forward_outputs[0].size())
        # print('backward_outputs len: ', len(backward_outputs), ' shape of that is ', backward_outputs[0].size())
        # print('final output size is ', output.size())
        return output


if __name__ == '__main__':
    # gradient check
    layer_num = 5
    convlstm = BiCLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=layer_num,
                        effective_step=range(layer_num)).cuda()
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(1, layer_num, 512, 64, 32)).cuda()
    target = Variable(torch.randn(1, 32, 64, 32)).double().cuda()

    output = convlstm(input)
    output = output.double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)