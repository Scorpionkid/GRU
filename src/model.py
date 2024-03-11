import torch
import torch.nn as nn
from torch.nn import functional as F
from torcheval.metrics.functional import r2_score
# 逐步实现GRU网络
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, w_ih, w_hh, b_ih, b_hh, device = "cuda"):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.b_ih = b_ih
        self.b_hh = b_hh
        self.reg = nn.Linear(hidden_size, out_size)
        self.device = device


    def gru_forward(self, input, initial_states):
        prev_h = initial_states.to(self.device)
        bs, T, i_size = input.shape
        h_size = self.w_ih.shape[0] // 3

        # 它和w_ih.expand(bs, -1, -1)的区别是：expand不复制数据，tile是增加内存复制数据
        # 如果要维度改变后的变量要修改的话，要用tile，因为expand不会复制数据
        batch_w_ih = self.w_ih.unsqueeze(0).tile(bs, 1, 1).to(self.device)
        batch_w_hh = self.w_hh.unsqueeze(0).tile(bs, 1, 1).to(self.device)

        # 初始化网络输出
        output = torch.zeros(bs, T, h_size).to(self.device)

        # 找到每个时刻的输入
        for t in range(T):
            x = input[:, t, :] # [bs, i_size]
            w_times_x = torch.bmm(batch_w_ih, x.unsqueeze(-1))  # [bs, 3*h_size, 1]
            w_times_x = w_times_x.squeeze(-1)  # [bs, 3*h_size]

            w_times_h_prev = torch.bmm(batch_w_hh, prev_h.unsqueeze(-1))  # [bs, 3*h_size, 1]
            w_times_h_prev = w_times_h_prev.squeeze(-1)  # [bs, 3*h_size]

            # 重置门
            r_t = torch.sigmoid(w_times_x[:, :h_size] + w_times_h_prev[:, :h_size] +
                                self.b_ih[:h_size] + self.b_hh[:h_size])
            # 更新门
            z_t = torch.sigmoid(w_times_x[:, h_size:2*h_size] + w_times_h_prev[:, h_size:2*h_size] +
                                self.b_ih[h_size:2*h_size] + self.b_hh[h_size:2*h_size])
            # 新的候选值
            n_t = torch.tanh(w_times_x[:, 2*h_size:3*h_size] + self.b_ih[2*h_size:3*h_size] +
                             r_t * (w_times_h_prev[:, 2*h_size:3*h_size] + self.b_hh[2*h_size:3*h_size]))
            # 更新隐藏状态
            prev_h = (1 - z_t) * n_t + z_t * prev_h

            # [batch_size, hidden_size]
            output[:, t, :] = prev_h

        # loss = None
        # loss = F.mse_loss(output.view(-1, 2), trg.view(-1, 2))
        # r2_s = r2_score(output.view(-1, 2), trg.view(-1, 2))
        # return out, loss, r2_s

        return output, prev_h

    def forward(self, src):
        N, T, C = src.shape
        h0 = torch.zeros(N, self.hidden_size)
        out, h = self.gru_forward(src, h0)
        out = self.reg(out)
        return out



# 测试
if __name__ == '__main__':
    bs, T, i_size, h_size = 2, 3, 4, 5
    input = torch.randn(bs, T, i_size) # 输入序列
    h0 = torch.randn(bs, h_size) # 初始值不参与训练，api里面维度(D∗num_layers,N,Hout)

    gru_layer = nn.GRU(i_size, h_size, batch_first = True)
    output, h_final = gru_layer(input,h0.unsqueeze(0))
    print(output)
    '''
    weight_ih_l0 torch.Size([15, 4])
    weight_hh_l0 torch.Size([15, 5])
    bias_ih_l0 torch.Size([15])
    bias_hh_l0 torch.Size([15])
    '''
    for k, v in gru_layer.named_parameters():
        print(k, v.shape)

    model = GRU(input, h0, gru_layer.weight_ih_l0, gru_layer.weight_hh_l0, gru_layer.bias_ih_l0, gru_layer.bias_hh_l0)

    output_custom, h_final_custom = model()

    print(output_custom)