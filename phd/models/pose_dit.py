"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import is_torch_version, logging
from diffusers.models.embeddings import get_2d_sincos_pos_embed

from diffusers.models.modeling_utils import ModelMixin
#from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.embeddings import Timesteps, TimestepEmbedding

from .transformer import BasicTransformerBlock


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class BetasProjection(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.

    Args:
        input_dim (`int`): The number of classes.
        hidden_size (`int`): The size of the vector embeddings.
        dropout_prob (`float`): The probability of dropping a label.
    """

    def __init__(self, input_dim, hidden_size, dropout_prob=0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.betas_embedding = nn.Linear(input_dim, hidden_size)
        #self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, betas):
        """
        Drops labels to enable classifier-free guidance.
        """
        drop_ids = torch.rand(betas.shape[0], device=betas.device) < self.dropout_prob
        betas[drop_ids] = 0.
        return betas

    def forward(self, betas):
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout):
            betas = self.token_drop(betas)

        embeddings = self.betas_embedding(betas)
        return embeddings


class PoseDiTTransformer2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D Transformer model as introduced in DiT (https://arxiv.org/abs/2212.09748).

    Parameters:
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        in_channels (int, defaults to 4): The number of channels in the input.
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks.
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        sample_size (int, defaults to 32):
            The width of the latent images. This parameter is fixed during training.
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        num_embeds_ada_norm (int, optional, defaults to 1000):
            Number of embeddings for AdaLayerNorm, fixed during training and affects the maximum denoising steps during
            inference.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-5):
            A small constant added to the denominator in normalization layers to prevent division by zero.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_joints: int = 24,
        num_attention_heads: int = 16,
        attention_head_dim: int = 32,
        in_channels: int = 6,
        out_channels: Optional[int] = None,
        num_layers: int = 20,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        attention_bias: bool = True,
        sample_size: int = 32,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 10,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        use_heatmap: bool = False,
    ):
        super().__init__()

        # Validate inputs.
        if norm_type != "ada_norm_zero":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_zero" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False
        self.use_heatmap = use_heatmap
        self.num_joints = num_joints
        # 2. Initialize the position embedding and transformer blocks.
        self.height = 16
        #self.width = 12
        self.width = 16

        self.patch_size = self.config.patch_size

        self.proj_in_img = nn.Linear(1280, self.inner_dim)
        self.proj_in_pose = nn.Linear(in_channels, self.inner_dim)
        print(num_embeds_ada_norm)

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)
        self.class_embedder = BetasProjection(num_embeds_ada_norm, self.inner_dim, dropout_prob=0.2)

        if self.use_heatmap:
            self.heatmap_projection = nn.Conv2d(17, self.inner_dim, 4, 4)

        self.pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.inner_dim,
            grid_size=(self.height, self.width),
            cls_token=False,
            extra_tokens=0,
            interpolation_scale=1.0,
            base_size=16,
            output_type = "pt").float().unsqueeze(0)

        self.task_tokens = nn.Parameter(torch.zeros(1, num_joints, self.inner_dim))


        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.out_channels)

        self.initialize_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.task_tokens, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        wi = self.proj_in_img.weight.data
        nn.init.xavier_uniform_(wi.view([wi.shape[0], -1]))
        nn.init.constant_(self.proj_in_img.bias, 0)

        w = self.proj_in_pose.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj_in_pose.bias, 0)

        if self.use_heatmap:
            wh = self.heatmap_projection.weight.data
            nn.init.xavier_uniform_(wh.view([wh.shape[0], -1]))
            nn.init.constant_(self.heatmap_projection.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.class_embedder.betas_embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.linear_2.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.proj_out_1.weight, 0)
        nn.init.constant_(self.proj_out_1.bias, 0)
        nn.init.constant_(self.proj_out_2.weight, 0)
        nn.init.constant_(self.proj_out_2.bias, 0)

    def forward(
        self,
        poses: torch.Tensor,
        img_tokens: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        heatmap: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ):
        """
        The [`DiTTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # 1. Input

        input_feature = self.proj_in_img(img_tokens) + self.pos_embed.to(img_tokens.device)

        if heatmap is not None:
            input_feature += self.heatmap_projection(heatmap.to(img_tokens.device)).flatten(2).transpose(1, 2)

        pose_feature = self.proj_in_pose(poses.to(img_tokens.device)) + self.task_tokens.to(img_tokens.device)

        hidden_states = torch.cat([input_feature, pose_feature], dim= 1) # B x (192+24) x C


        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(hidden_states.device))  # (N, D)

        class_emb = self.class_embedder(class_labels)  # (N, D)

        conditioning = timesteps_emb + class_emb  # (N, D)


        #height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        #hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    conditioning,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=conditioning,
                )

        # 3. Output
        #conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=hidden_states.dtype)
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states) # B x (192+24) x C


        output = hidden_states [:, -self.num_joints:, :]
        
        # unpatchify
        #height = width = int(hidden_states.shape[1] ** 0.5)
        #hidden_states = hidden_states.reshape(
        #    shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        #)
        #hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        #output = hidden_states.reshape(
        #    shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        #)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)