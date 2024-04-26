from diffusers import StableDiffusionPipeline
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import DDIMScheduler
import gc
from PIL import Image

from icm.models.feature_extractor.attention_controllers import AttentionStore
import xformers


def register_attention_control(model, controller, if_softmax=True, ensemble_size=1):
    def ca_forward(self, place_in_unet, att_opt_b):

        class MyXFormersAttnProcessor:
            r"""
            Processor for implementing memory efficient attention using xFormers.

            Args:
                attention_op (`Callable`, *optional*, defaults to `None`):
                    The base
                    [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
                    use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
                    operator.
            """

            def __init__(self, attention_op=None):
                self.attention_op = attention_op

            def __call__(
                self,
                attn,
                hidden_states: torch.FloatTensor,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
            ):
                residual = hidden_states

                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)

                input_ndim = hidden_states.ndim

                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(
                        batch_size, channel, height * width).transpose(1, 2)

                batch_size, key_tokens, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )

                attention_mask = attn.prepare_attention_mask(
                    attention_mask, key_tokens, batch_size)
                if attention_mask is not None:
                    # expand our mask's singleton query_tokens dimension:
                    #   [batch*heads,            1, key_tokens] ->
                    #   [batch*heads, query_tokens, key_tokens]
                    # so that it can be added as a bias onto the attention scores that xformers computes:
                    #   [batch*heads, query_tokens, key_tokens]
                    # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
                    _, query_tokens, _ = hidden_states.shape
                    attention_mask = attention_mask.expand(
                        -1, query_tokens, -1)

                if attn.group_norm is not None:
                    hidden_states = attn.group_norm(
                        hidden_states.transpose(1, 2)).transpose(1, 2)

                query = attn.to_q(hidden_states)

                is_cross = False if encoder_hidden_states is None else True

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(
                        encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)

                query = attn.head_to_batch_dim(query).contiguous()
                key = attn.head_to_batch_dim(key).contiguous()
                value = attn.head_to_batch_dim(value).contiguous()

                # controller
                if query.shape[1] in controller.store_res:
                    sim = torch.einsum('b i d, b j d -> b i j',
                                    query, key) * attn.scale

                    if if_softmax:
                        sim = sim / if_softmax
                        my_attn = sim.softmax(dim=-1).detach()
                        del sim
                    else:
                        my_attn = sim.detach()

                    controller(my_attn, is_cross, place_in_unet, ensemble_size, batch_size)

                # end controller

                hidden_states = xformers.ops.memory_efficient_attention(
                    query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
                )
                hidden_states = hidden_states.to(query.dtype)
                hidden_states = attn.batch_to_head_dim(hidden_states)

                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(
                        -1, -2).reshape(batch_size, channel, height, width)

                if attn.residual_connection:
                    hidden_states = hidden_states + residual

                hidden_states = hidden_states / attn.rescale_output_factor

                return hidden_states

        return MyXFormersAttnProcessor(att_opt_b)

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.processor = ca_forward(
                net_, place_in_unet, net_.processor.attention_op)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    # sub_nets = model.unet.named_children()
    sub_nets = model.unet.named_children()
    # for net in sub_nets:
    #     if "down" in net[0]:
    #         cross_att_count += register_recr(net[1], 0, "down")
    #     elif "up" in net[0]:
    #         cross_att_count += register_recr(net[1], 0, "up")
    #     elif "mid" in net[0]:
    #         cross_att_count += register_recr(net[1], 0, "mid")
    for net in sub_nets:
        if "down_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up_blocks" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid_block" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count


class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            # if i > np.max(up_ft_indices):
            #     break

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

            if i in up_ft_indices:
                up_ft[i] = sample.detach()

        output = {}
        output["up_ft"] = up_ft
        return output


class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[
            int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        device = self._execution_device
        latents = (
            self.vae.encode(img_tensor).latent_dist.sample()
            * self.vae.config.scaling_factor
        )
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(
            latents_noisy,
            t,
            up_ft_indices,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        return unet_output


class SDFeaturizer(nn.Module):
    def __init__(self, sd_id='pretrained_models/stable-diffusion-2-1',
                 load_local=True, ):
        super().__init__()
        # sd_id="stabilityai/stable-diffusion-2-1", load_local=False):
        unet = MyUNet2DConditionModel.from_pretrained(
            sd_id,
            subfolder="unet",
            # output_loading_info=True,
            local_files_only=load_local,
            low_cpu_mem_usage=True,
            use_safetensors=False,
            # torch_dtype=torch.float16,
            # device_map="auto",
        )
        onestep_pipe = OneStepSDPipeline.from_pretrained(
            sd_id,
            unet=unet,
            safety_checker=None,
            local_files_only=load_local,
            low_cpu_mem_usage=True,
            use_safetensors=False,
            # torch_dtype=torch.float16,
            # device_map="auto",
        )
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(
            sd_id, subfolder="scheduler"
        )
        gc.collect()

        onestep_pipe = onestep_pipe.to("cuda")

        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        self.pipe = onestep_pipe
        
        # register nn.module for ddp
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        
        # freeze vae and unet
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, img_tensor, prompt='', t=261, up_ft_index=3, ensemble_size=8):
        """
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        """
        img_tensor = img_tensor.repeat(
            ensemble_size, 1, 1, 1).cuda()  # ensem, c, h, w
        prompt_embeds = self.pipe._encode_prompt(
            prompt=prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )  # [1, 77, dim]
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
        )
        unet_ft = unet_ft_all["up_ft"][up_ft_index]  # ensem, c, h, w
        unet_ft = unet_ft.mean(0, keepdim=True)  # 1,c,h,w
        return unet_ft
    # index 0: 1280, 24, 24
    # index 1: 1280, 48, 48
    # index 2: 640, 96, 96
    # index 3: 320, 96，96
    @torch.no_grad()
    def forward_feature_extractor(self, uc, img_tensor, t=261, up_ft_index=[0, 1, 2, 3], ensemble_size=8):
        """
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        """
        batch_size = img_tensor.shape[0]
        
        img_tensor = img_tensor.unsqueeze(1).repeat(1, ensemble_size, 1, 1, 1)
        
        img_tensor = img_tensor.reshape(-1, *img_tensor.shape[2:])
        
        prompt_embeds = uc.repeat(
            img_tensor.shape[0], 1, 1).to(img_tensor.device)
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=up_ft_index,
            prompt_embeds=prompt_embeds,
        )
        unet_ft = unet_ft_all["up_ft"]  # ensem, c, h, w

        return unet_ft


class FeatureExtractor(nn.Module):
    def __init__(self,
                 sd_id='stabilityai/stable-diffusion-2-1',  # 'pretrained_models/stable-diffusion-2-1',
                 load_local=True,
                 if_softmax=False,
                 feature_index_cor=1,
                 feature_index_matting=4,
                 attention_res=32,  # [16, 32],
                 set_diag_to_one=True,
                 time_steps=[0],
                 extract_feature_inputted_to_layer=False,
                 ensemble_size=8):
        super().__init__()
        
        self.dift_sd = SDFeaturizer(sd_id=sd_id, load_local=load_local)
        # register buffer for prompt embedding
        self.register_buffer("prompt_embeds", self.dift_sd.pipe._encode_prompt(
            prompt='',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            device="cuda",
        ))
        # free self.pipe.tokenizer and self.pipe.text_encoder
        del self.dift_sd.pipe.tokenizer
        del self.dift_sd.pipe.text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        self.feature_index_cor = feature_index_cor
        self.feature_index_matting = feature_index_matting
        self.attention_res = attention_res
        self.set_diag_to_one = set_diag_to_one
        self.time_steps = time_steps
        self.extract_feature_inputted_to_layer = extract_feature_inputted_to_layer
        self.ensemble_size = ensemble_size        
        self.register_attention_store(
            if_softmax=if_softmax, attention_res=attention_res)


    def register_attention_store(self, if_softmax=False, attention_res=[16, 32]):
        self.controller = AttentionStore(store_res=attention_res)

        register_attention_control(
            self.dift_sd.pipe, self.controller, if_softmax=if_softmax, ensemble_size=self.ensemble_size)

    def get_trainable_params(self):
        return []

    def get_reference_feature(self, images):
        self.controller.reset()
        batch_size = images.shape[0]
        features = self.dift_sd.forward_feature_extractor(
            self.prompt_embeds, images, t=self.time_steps[0], ensemble_size=self.ensemble_size) # b*e, c, h, w

        features = self.ensemble_feature(
            features, self.feature_index_cor, batch_size)
        
        return features.detach()

    def ensemble_feature(self, features, index, batch_size):
        if isinstance(index, int):

            features_ = features[index].reshape(
                batch_size, self.ensemble_size, *features[index].shape[1:])
            features_ = features_.mean(1, keepdim=False).detach()
        else:
            index = list(index)
            res = ['24','48','96']
            res = res[:len(index)]
            features_ = {}
            for i in range(len(index)):
                features_[res[i]] = features[index[i]].reshape(
                    batch_size, self.ensemble_size, *features[index[i]].shape[1:])
                features_[res[i]] = features_[res[i]].mean(1, keepdim=False).detach()
        return features_

    def get_source_feature(self, images):
        # return {"ft": [B, C, H, W], "attn": [B, H, W, H*W]}

        self.controller.reset()
        torch.cuda.empty_cache()
        batch_size = images.shape[0]
        
        ft = self.dift_sd.forward_feature_extractor(
            self.prompt_embeds, images, t=self.time_steps[0], ensemble_size=self.ensemble_size) # b*e, c, h, w


        attention_maps = self.get_feature_attention(batch_size)

        output = {"ft_cor": self.ensemble_feature(ft, self.feature_index_cor, batch_size),
                  "attn": attention_maps, 'ft_matting': self.ensemble_feature(ft, self.feature_index_matting, batch_size)}
        return output

    def get_feature_attention(self, batch_size):

        attention_maps = self.__aggregate_attention(
            from_where=["down", "mid", "up"], is_cross=False, batch_size=batch_size)

        for attn_map in attention_maps.keys():
            attention_maps[attn_map] = attention_maps[attn_map].permute(0, 2, 1).reshape(
                (batch_size, -1, int(attn_map), int(attn_map)))  # [bs, h*w, h, w]
            attention_maps[attn_map] = attention_maps[attn_map].permute(0, 2, 3, 1)  # [bs, h, w, h*w]
        return attention_maps

    def __aggregate_attention(self, from_where: List[str], is_cross: bool, batch_size: int):
        out = {}
        self.controller.between_steps()
        self.controller.cur_step=1
        attention_maps = self.controller.get_average_attention()
        for res in self.attention_res:
            out[str(res)] = self.__aggregate_attention_single_res(
                from_where, is_cross, batch_size, res, attention_maps)
        return out
    
    def __aggregate_attention_single_res(self, from_where: List[str], is_cross: bool, batch_size: int, res: int, attention_maps):
        out = []
        num_pixels = res ** 2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        batch_size, -1, res, res, item.shape[-1])
                    out.append(cross_maps)
        out = torch.cat(out, dim=1)
        out = out.sum(1) / out.shape[1]
        out = out.reshape(batch_size, out.shape[-1], out.shape[-1])

        if self.set_diag_to_one:
            for o in out:
                o = o - torch.diag(torch.diag(o)) + \
                    torch.eye(o.shape[0]).to(o.device)
        return out
