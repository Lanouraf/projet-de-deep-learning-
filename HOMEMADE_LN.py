class LayerNorm(nn.Module):
        def __init__(self, features, eps=1e-6):
            super(LayerNorm, self).__init__()
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            return self.gamma * (x - mean) / (std + self.eps) + self.beta

class PytorchLayerNorm(Module):
    r"""
    ## Layer Normalization
    # from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/layer_norm/__init__.py
    Layer normalization $\text{LN}$ normalizes the input $X$ as follows:

    When input $X \in \mathbb{R}^{B \times C}$ is a batch of embeddings,
    where $B$ is the batch size and $C$ is the number of features.
    $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
    $$\text{LN}(X) = \gamma
    \frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}}
    + \beta$$

    When input $X \in \mathbb{R}^{L \times B \times C}$ is a batch of a sequence of embeddings,
    where $B$ is the batch size, $C$ is the number of channels, $L$ is the length of the sequence.
    $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
    $$\text{LN}(X) = \gamma
    \frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}}
    + \beta$$

    When input $X \in \mathbb{R}^{B \times C \times H \times W}$ is a batch of image representations,
    where $B$ is the batch size, $C$ is the number of channels, $H$ is the height and $W$ is the width.
    This is not a widely used scenario.
    $\gamma \in \mathbb{R}^{C \times H \times W}$ and $\beta \in \mathbb{R}^{C \times H \times W}$.
    $$\text{LN}(X) = \gamma
    \frac{X - \underset{C, H, W}{\mathbb{E}}[X]}{\sqrt{\underset{C, H, W}{Var}[X] + \epsilon}}
    + \beta$$
    """

    def __init__(self, normalized_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        """
        * `normalized_shape` $S$ is the shape of the elements (except the batch).
         The input should then be
         $X \in \mathbb{R}^{* \times S[0] \times S[1] \times ... \times S[n]}$
        * `eps` is $\epsilon$, used in $\sqrt{Var[X] + \epsilon}$ for numerical stability
        * `elementwise_affine` is whether to scale and shift the normalized value

        We've tried to use the same names for arguments as PyTorch `LayerNorm` implementation.
        """
        super().__init__()

        # Convert `normalized_shape` to `torch.Size`
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        #
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # Create parameters for $\gamma$ and $\beta$ for gain and bias
        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor):
        """
        `x` is a tensor of shape `[*, S[0], S[1], ..., S[n]]`.
        `*` could be any number of dimensions.
         For example, in an NLP task this will be
        `[seq_len, batch_size, features]`
        """
        # Sanity check to make sure the shapes match
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]

        # The dimensions to calculate the mean and variance on
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        # Calculate the mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X]$
        mean = x.mean(dim=dims, keepdim=True)
        # Calculate the squared mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X^2]$
        mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)
        # Variance of all element $Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$
        var = mean_x2 - mean ** 2

        # Normalize $$\hat{X} = \frac{X - \mathbb{E}[X]}{\sqrt{Var[X] + \epsilon}}$$
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Scale and shift $$\text{LN}(x) = \gamma \hat{X} + \beta$$
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        #
        return x_norm
