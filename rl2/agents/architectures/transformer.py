"""
Implements Transformer architectures for RL^2.
"""

import math

import torch as tc

from rl2.agents.architectures.common import LayerNorm, MultiheadSelfAttention


class FF(tc.nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            connection_style,
            hidden_activation=tc.nn.ReLU(),
            output_activation=None,
            use_ln=True
    ):
        assert connection_style in ['plain', 'residual', 'dense']
        assert input_dim == output_dim or connection_style != 'residual'
        super().__init__()

        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation
        self._connection_style = connection_style
        self._use_ln = use_ln

        if self._use_ln:
            self._ln = LayerNorm(units=input_dim)

        self._lin1 = tc.nn.Linear(
            in_features=self._input_dim,
            out_features=self._hidden_dim,
            bias=True)
        tc.nn.init.xavier_normal_(self._lin1.weight)
        tc.nn.init.zeros_(self._lin1.bias)

        self._lin2 = tc.nn.Linear(
            in_features=self._hidden_dim,
            out_features=self._output_dim,
            bias=True)
        tc.nn.init.xavier_normal_(self._lin2.weight)
        tc.nn.init.zeros_(self._lin2.bias)

    def forward(self, inputs):
        """
        Args:
            inputs: input vec tensor of shape [B, T2, I]

        Returns:
            output tensor of shape [B, T2, O] with output dim O
            determined by self._connection_style
        """
        x = inputs
        if self._use_ln:
            x = self._ln(x)

        x = self._lin1(x)
        x = self._hidden_activation(x)
        x = self._lin2(x)
        if self._output_activation is not None:
            x = self._output_activation(x)

        if self._connection_style == 'plain':
            return x
        elif self._connection_style == 'residual':
            return inputs + x
        elif self._connection_style == 'dense':
            return tc.cat((inputs, x), dim=-1)
        else:
            raise NotImplementedError


class TransformerXLILayer(tc.nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            d_head,
            d_ff,
            attention_style,
            row_len=None
    ):
        assert attention_style == 'full' or row_len is not None
        super().__init__()
        self._d_model = d_model
        self._n_head = n_head
        self._d_head = d_head
        self._d_ff = d_ff
        self._attention_style = attention_style
        self._row_len = row_len

        self._attn = MultiheadSelfAttention(
            input_dim=self._d_model,
            num_heads=self._n_head,
            num_head_features=self._d_head,
            position_encoding_style='rel',
            attention_style=self._attention_style,
            connection_style='residual',
            activation=tc.nn.ReLU(),
            use_ln=True,
            row_len=self._row_len)

        self._ff = FF(
            input_dim=self._d_model,
            hidden_dim=self._d_ff,
            output_dim=self._d_model,
            connection_style='residual',
            hidden_activation=tc.nn.ReLU(),
            output_activation=tc.nn.ReLU(),
            use_ln=True)

    def forward(self, inputs, past_kvs=None):
        """
        Args:
            inputs: input vec tensor of shape [B, T2, I]
            past_kvs: optional past kvs with shape [B, T1, H*F*2]

        Returns:
            output tensor of shape [B, T2, I]
            and new_kvs tensor of shape [B, T1+T2, H*F*2]
        """
        attn_output, new_kvs = self._attn(inputs=inputs, past_kvs=past_kvs)
        ff_output = self._ff(inputs=attn_output)

        return ff_output, new_kvs


class TransformerXLI(tc.nn.Module):
    """
    Implements the Transformer XL-I from Parisotto et al., 2019.
    """
    def __init__(self, input_dim, n_layer, n_head, d_model, d_head):
        super().__init__()
        self._input_dim = input_dim
        self._n_layer = n_layer
        self._n_head = n_head
        self._d_model = d_model
        self._d_head = d_head

        self._lin = tc.nn.Linear(
            self._input_dim, self._d_model, bias=False)
        tc.nn.init.xavier_normal_(self._lin.weight)

        self._transformer_layers = tc.nn.ModuleList([
            TransformerXLILayer(
                d_model=self._d_model,
                n_head=self._n_head,
                d_head=self._d_head,
                d_ff=self._d_model,
                attention_style='full')
            for _ in range(0, self._n_layer)
        ])

        self._ln = LayerNorm(units=self._d_model)

    @property
    def output_dim(self):
        return self._d_model

    def initial_state(self, batch_size):
        return None

    def forward(self, inputs, prev_state=None):
        """
        Args:
            inputs: input vec tensor of shape [B, ..., I]
            prev_state: optional past kvs tensor of shape [L, B, T1, H*F*2]

         Notes:
            '...' must be either one dimensional or must not exist

        Returns:
            output feature tensor of shape [B, ..., d_model]
            and new_kvs tensor
        """
        assert len(list(inputs.shape)) in [2, 3]
        if len(list(inputs.shape)) == 2:
            inputs = inputs.unsqueeze(1)

        past_kvs = [None] * self._n_layer if prev_state is None else prev_state

        inputs = self._lin(inputs)
        inputs = tc.nn.ReLU()(inputs)

        new_kvs_by_layer = []
        for l in range(0, self._n_layer):
            inputs, new_kvs = self._transformer_layers[l](
                inputs=inputs, past_kvs=past_kvs[l])
            new_kvs_by_layer.append(new_kvs)

        features = self._ln(inputs)
        new_kvs = tc.stack(new_kvs_by_layer, dim=0)

        if features.shape[1] == 1:
            features = features.squeeze(1)

        return features, new_kvs


class SparseTransformerXLI(tc.nn.Module):
    """
    Implements a Sparse Transformer (Child et al., 2019) variant,
    using the attention operations introduced by Dhariwal et al., 2020,
    and the relative position encoding from Dai et al., 2019,
    and the reordered layer ordering from Parisotto et al., 2019.
    """
    def __init__(self, input_dim, n_layer, n_head, d_model, d_head, n_context):
        super().__init__()
        self._input_dim = input_dim
        self._n_layer = n_layer
        self._n_head = n_head
        self._d_model = d_model
        self._d_head = d_head
        self._n_context = n_context
        self._attention_styles = ['row', 'column', 'previous_row']

        self._lin = tc.nn.Linear(
            self._input_dim, self._d_model, bias=False)
        tc.nn.init.xavier_normal_(self._lin.weight)

        self._transformer_layers = tc.nn.ModuleList([
            TransformerXLILayer(
                d_model=self._d_model,
                n_head=self._n_head,
                d_head=self._d_head,
                d_ff=self._d_model,
                attention_style=self._attention_styles[l % 3],
                row_len=self._row_len)
            for l in range(0, self._n_layer)
        ])

        self._ln = LayerNorm(units=self._d_model)

    @property
    def _row_len(self):
        small = math.floor(self._n_context ** 0.5)
        big = self._n_context // small
        while small * big != self._n_context:
            small -= 1
            big = self._n_context // small
        return small

    @property
    def output_dim(self):
        return self._d_model

    def initial_state(self, batch_size):
        return None

    def forward(self, inputs, prev_state=None):
        """
        Args:
            inputs: input vec tensor of shape [B, ..., I]
            prev_state: optional past kvs tensor of shape [L, B, T1, H*F*2]

         Notes:
            '...' must be either one dimensional or must not exist

        Returns:
            output feature tensor of shape [B, ..., d_model]
            and new_kvs tensor
        """
        assert len(list(inputs.shape)) in [2, 3]
        if len(list(inputs.shape)) == 2:
            inputs = inputs.unsqueeze(1)

        past_kvs = [None] * self._n_layer if prev_state is None else prev_state

        inputs = self._lin(inputs)
        inputs = tc.nn.ReLU()(inputs)

        new_kvs_by_layer = []
        for l in range(0, self._n_layer):
            inputs, new_kvs = self._transformer_layers[l](
                inputs=inputs, past_kvs=past_kvs[l])
            new_kvs_by_layer.append(new_kvs)

        features = self._ln(inputs)
        new_kvs = tc.stack(new_kvs_by_layer, dim=0)

        if features.shape[1] == 1:
            features = features.squeeze(1)

        return features, new_kvs