"""
ConvLSTM implementation for spatiotemporal sequence prediction.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        h, w = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
        )


class ConvLSTMForecaster(nn.Module):
    """
    Encoder-style stacked ConvLSTM that consumes a sequence and
    autoregressively predicts `pred_len` future frames.
    """

    def __init__(self, input_channels, hidden_dims, kernel_size, num_layers, pred_len):
        super().__init__()
        assert len(hidden_dims) == num_layers

        self.num_layers = num_layers
        self.pred_len = pred_len

        cells = []
        for i in range(num_layers):
            in_dim = input_channels if i == 0 else hidden_dims[i - 1]
            cells.append(ConvLSTMCell(in_dim, hidden_dims[i], kernel_size))
        self.cells = nn.ModuleList(cells)

        self.output_conv = nn.Conv2d(hidden_dims[-1], input_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        device = x.device

        hidden_states = [cell.init_hidden(b, (h, w), device) for cell in self.cells]

        # --- Encode input sequence ---
        for time_step in range(t):
            input_t = x[:, time_step]
            for layer_idx, cell in enumerate(self.cells):
                h_cur, c_cur = hidden_states[layer_idx]
                h_next, c_next = cell(input_t, (h_cur, c_cur))
                hidden_states[layer_idx] = (h_next, c_next)
                input_t = h_next

        # --- Autoregressive decoding ---
        outputs = []
        decoder_input = self.output_conv(hidden_states[-1][0])  # last hidden -> first pred frame

        for _ in range(self.pred_len):
            input_t = decoder_input
            for layer_idx, cell in enumerate(self.cells):
                h_cur, c_cur = hidden_states[layer_idx]
                h_next, c_next = cell(input_t, (h_cur, c_cur))
                hidden_states[layer_idx] = (h_next, c_next)
                input_t = h_next

            frame = self.output_conv(hidden_states[-1][0])
            outputs.append(frame)
            decoder_input = frame

        # (B, pred_len, C, H, W)
        return torch.stack(outputs, dim=1)