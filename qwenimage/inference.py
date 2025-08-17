# -*- coding: utf-8 -*-

import cv2
import torch
import argparse
import numpy as np
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from diffusers import QwenImageImg2ImgPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.utils import is_torch_xla_available
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

if is_torch_xla_available():
    # import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

class QwenImageGeneratorEditor(QwenImageImg2ImgPipeline):
    def prepare_latents_edit(self, image, timestep, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        if isinstance(generator, list) and len(generator) != len(image):
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        
        batch_size = len(image)
        shape = (batch_size, 1, num_channels_latents, height, width)

        # If image is [B,C,H,W] -> add T=1. If it's already [B,C,T,H,W], leave it.
        if image.dim() == 4:
            image = image.unsqueeze(2)
        elif image.dim() != 5:
            raise ValueError(f"Expected image dims 4 or 5, got {image.dim()}.")

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        image = image.to(device=device, dtype=dtype)
        if image.shape[1] != self.latent_channels:
            image_latents = self._encode_vae_image(image=image, generator=generator)  # [B,z,1,H',W']
        else:
            image_latents = image
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        image_latents = image_latents.transpose(1, 2)  # [B,1,z,H',W']
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids
    
    def prepare_latents_gen(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        image=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        strength: float = 0.6,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        mode='edit',
    ):
        if mode == 'edit':
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, vae_latent_channels=16)
        else:
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            strength,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Preprocess image
        if image is not None:
            init_image = self.image_processor.preprocess(image, height=height, width=width)
            init_image = init_image.to(dtype=torch.float32,device='cuda:0')

        # 3. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        if mode == 'edit':
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
            if num_inference_steps < 1:
                raise ValueError(
                    f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                    f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
                )
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        if mode == 'edit':
            latents, latent_image_ids = self.prepare_latents_edit(
                init_image,
                latent_timestep,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        elif mode == 'gen':
            latents, latent_image_ids = self.prepare_latents_gen(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        else:
            raise ValueError('Unknown mode.')
            
        img_shapes = [(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)] * batch_size

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        # 6. Denoising loop
        if mode == 'gen':
            self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with self.transformer.cache_context("cond"):    
                    hidden_states = self.transformer.img_in(latents).to("cuda:0")
                    encoder_hidden_states = self.transformer.txt_norm(prompt_embeds).to("cuda:0")
                    encoder_hidden_states = self.transformer.txt_in(encoder_hidden_states)
                    encoder_hidden_states_mask = prompt_embeds_mask.to("cuda:0")
                    # txt_seq_lens = txt_seq_lens.to('cuda:0')
                    
                    # calculate shared parameters
                    timestep_float = timestep.to("cuda:0") / 1000
                    temb = (
                        self.transformer.time_text_embed(timestep_float, hidden_states)
                        if guidance is None
                        else self.transformer.time_text_embed(timestep_float, guidance.to("cuda:0"), hidden_states)
                    )
                    image_rotary_emb = self.transformer.pos_embed(
                        img_shapes, 
                        prompt_embeds_mask.sum(dim=1).tolist(),
                        # txt_seq_lens, 
                        device="cuda:0"
                    )

                    # middle 26 layers
                    temb_c1 = temb.to("cuda:1", non_blocking=True)
                    image_rotary_emb_c1 = tuple(item.to("cuda:1", non_blocking=True) for item in image_rotary_emb)
                    torch.cuda.synchronize("cuda:1")
                    
                    for i, block in enumerate(self.transformer.transformer_blocks[:30]):
                        # 只传输变化的hidden_states和encoder states
                        hidden_states = hidden_states.to("cuda:1", non_blocking=True)
                        encoder_hidden_states = encoder_hidden_states.to("cuda:1", non_blocking=True)
                        encoder_hidden_states_mask = encoder_hidden_states_mask.to("cuda:1", non_blocking=True)
                        torch.cuda.synchronize("cuda:1")
                        
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=temb_c1,
                            image_rotary_emb=image_rotary_emb_c1,
                            joint_attention_kwargs=self.attention_kwargs,
                        )
                    
                    # last 26 layers
                    temb_c2 = temb.to("cuda:2", non_blocking=True)
                    image_rotary_emb_c2 = tuple(item.to("cuda:2", non_blocking=True) for item in image_rotary_emb)
                    torch.cuda.synchronize("cuda:2")
                    
                    for i, block in enumerate(self.transformer.transformer_blocks[30:], start=30):
                        hidden_states = hidden_states.to("cuda:2", non_blocking=True)
                        encoder_hidden_states = encoder_hidden_states.to("cuda:2", non_blocking=True)
                        encoder_hidden_states_mask = encoder_hidden_states_mask.to("cuda:2", non_blocking=True)
                        torch.cuda.synchronize("cuda:2")
                        
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=temb_c2,
                            image_rotary_emb=image_rotary_emb_c2,
                            joint_attention_kwargs=self.attention_kwargs,
                        )
                    
                    hidden_states = hidden_states.to("cuda:0", non_blocking=True)
                    torch.cuda.synchronize("cuda:0")
                    
                    hidden_states = self.transformer.norm_out(hidden_states, temb)
                    noise_pred = self.transformer.proj_out(hidden_states)

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        # 初始化hidden_states在cuda:0 (使用相同的latents)
                        hidden_states = self.transformer.img_in(latents).to("cuda:0")
                        encoder_hidden_states = self.transformer.txt_norm(negative_prompt_embeds).to("cuda:0")
                        encoder_hidden_states = self.transformer.txt_in(encoder_hidden_states)
                        # encoder_hidden_states_mask = negative_prompt_embeds_mask.to("cuda:0")
                        
                        # calcuate shared parameters
                        timestep_float = timestep.to("cuda:0") / 1000
                        temb = (
                            self.transformer.time_text_embed(timestep_float, hidden_states)
                            if guidance is None
                            else self.transformer.time_text_embed(timestep_float, guidance.to("cuda:0"), hidden_states)
                        )
                        image_rotary_emb = self.transformer.pos_embed(
                            img_shapes, 
                            negative_prompt_embeds_mask.sum(dim=1).tolist(),                            device="cuda:0"
                        )

                        # middle 26 layers
                        temb_c1 = temb.to("cuda:1", non_blocking=True)
                        image_rotary_emb_c1 = tuple(item.to("cuda:1", non_blocking=True) for item in image_rotary_emb)
                        torch.cuda.synchronize("cuda:1")
                        
                        for i, block in enumerate(self.transformer.transformer_blocks[:30]):
                            # 只传输变化的hidden_states和encoder states
                            hidden_states = hidden_states.to("cuda:1", non_blocking=True)
                            encoder_hidden_states = encoder_hidden_states.to("cuda:1", non_blocking=True)
                            encoder_hidden_states_mask = encoder_hidden_states_mask.to("cuda:1", non_blocking=True)
                            torch.cuda.synchronize("cuda:1")
                            
                            encoder_hidden_states, hidden_states = block(
                                hidden_states=hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_hidden_states_mask=encoder_hidden_states_mask,
                                temb=temb_c1,
                                image_rotary_emb=image_rotary_emb_c1,
                                joint_attention_kwargs=self.attention_kwargs,
                            )
                        
                        # last 26 layers
                        temb_c2 = temb.to("cuda:2", non_blocking=True)
                        image_rotary_emb_c2 = tuple(item.to("cuda:2", non_blocking=True) for item in image_rotary_emb)
                        torch.cuda.synchronize("cuda:2")
                        
                        for i, block in enumerate(self.transformer.transformer_blocks[30:], start=30):
                            hidden_states = hidden_states.to("cuda:2", non_blocking=True)
                            encoder_hidden_states = encoder_hidden_states.to("cuda:2", non_blocking=True)
                            encoder_hidden_states_mask = encoder_hidden_states_mask.to("cuda:2", non_blocking=True)
                            torch.cuda.synchronize("cuda:2")
                            
                            encoder_hidden_states, hidden_states = block(
                                hidden_states=hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_hidden_states_mask=encoder_hidden_states_mask,
                                temb=temb_c2,
                                image_rotary_emb=image_rotary_emb_c2,
                                joint_attention_kwargs=self.attention_kwargs,
                            )
                        
                        hidden_states = hidden_states.to("cuda:0", non_blocking=True)
                        torch.cuda.synchronize("cuda:0")
                        
                        hidden_states = self.transformer.norm_out(hidden_states, temb)
                        neg_noise_pred = self.transformer.proj_out(hidden_states)

                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    # xm.mark_step()
                    ...

                torch.cuda.empty_cache()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--generate", action='store_true')
    parser.add_argument("--single_edit", action='store_true')
    parser.add_argument("--multi_edit", action='store_true')
    args = parser.parse_args()

    pipe = QwenImageGeneratorEditor.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to("cuda:0")
    pipe.vae.to("cuda:0")
    for i, block in enumerate(pipe.transformer.transformer_blocks):
        if i < 30:
            block.to("cuda:1")  # 前34个block -> GPU 1
        else:
            block.to("cuda:2")  # 后26个block -> GPU 2

    for name, module in pipe.transformer.named_children():
        if name != "transformer_blocks":  # 排除已经手动分配的 blocks
            module.to("cuda:0")
    
    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition.", # for english prompt,
        "zh": "超清，4K，电影级构图" # for chinese prompt,
    }

    def pil_to_numpy(images):
        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)
        return images
    
    if args.generate:
        prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''

        negative_prompt = ""

        # Generate with different aspect ratios
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472)
        }

        width, height = aspect_ratios["16:9"]

        image = pipe(
            prompt=prompt + positive_magic["zh"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42),
            mode='gen'
        ).images[0]
        image.save("generated_result.png")
        print('Testing image generation, saving at generated_result.png')
    
    if args.single_edit:
        init_image = load_image("./cute_cat.png")
        init_image = pil_to_numpy(init_image)
        prompt = "wizard dog, Gandalf-inspired, Lord of the Rings aesthetic, majestic yet cute, Studio Ghibli style"
        negative_prompt = ""
        strengths = [0.8]
        for s in strengths:
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                height=init_image[0].shape[0],
                width=init_image[0].shape[1],
                strength=s,
                num_inference_steps=50,
                true_cfg_scale=4.0,
                generator=[torch.Generator(device="cuda").manual_seed(42)],
                mode='edit'
            )
            out.images[0].save(f"single-img2img-strength-{s}.png")
        print('Testing image generation, saving at single-img2img-{strength}.png')

    if args.multi_edit:
        init_image = load_image("./cute_cat.png")
        init_image1 = load_image("./ref.png").resize((init_image.size[0],init_image.size[1]))
        init_image = [init_image,init_image1]
        init_image = pil_to_numpy(init_image)
        prompt = "Convert the style of the first image to the second image."
        negative_prompt = ""
        strengths = [0.6,0.7,0.8,0.9,1.0]
        for s in strengths:
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                height=init_image[0].shape[0],
                width=init_image[0].shape[1],
                strength=s,
                num_inference_steps=50,
                true_cfg_scale=4.0,
                generator=[torch.Generator(device="cuda").manual_seed(42),torch.Generator(device="cuda").manual_seed(42)],
                mode='edit'
            )
            out.images[0].save(f"multi-img2img-strength-{s}.png")
        print('Testing image generation, saving at multi-img2img-{strength}.png')