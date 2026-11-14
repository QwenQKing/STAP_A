import os
from datetime import datetime
import copy
import pickle
import re
import threading
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from copy import deepcopy
from typing import Any, List, Union
import hashlib
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from tqdm import tqdm
from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils import hf_processor
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    _repeat_interleave,
    _pre_process_inputs,  
    vLLMRollout,
)

from STAP.utils.tools.video_generate import call_video_generate, call_video_generate_batch
from STAP.utils.torch_functional import get_final_eos_mask


def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor:
    assert all([t.ndim == 1 for t in tensor_list])
    max_len = max([t.size(0) for t in tensor_list])
    padded_tensor_list = []
    for t in tensor_list:
        padded_tensor_list.append(
            torch.cat([t, torch.tensor([pad_token_id] * (max_len - t.size(0)), device=t.device, dtype=t.dtype)], dim=0)
        )
    return torch.stack(padded_tensor_list, dim=dim)


class vLLMRollout_MultiTurn_STAP(vLLMRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):

        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)

        self.tokenizer = tokenizer
        self.processor = hf_processor(model_path)

        gen_cfg = config.get('generate', None)

        self.user_prompt_after_generate = None
        after_path = gen_cfg.get('user_prompt_after_generate', None) if gen_cfg else None
        try:
            with open(after_path, 'rb') as file:
                self.user_prompt_after_generate = pickle.load(file)
        except Exception as e:
            print(f"Error: {e} | user_prompt_after_generate default to None")

        self.user_prompt_last_round = None
        last_path = gen_cfg.get('user_prompt_last_round', None) if gen_cfg else None
        try:
            with open(last_path, 'rb') as file:
                self.user_prompt_last_round = pickle.load(file)
        except Exception as e:
            print(f"Error: {e} | user_prompt_last_round falls back to a built-in default")
        if not self.user_prompt_last_round:
            self.user_prompt_last_round = (
                'This is the FINAL round, so you can no longer refine - you must finish now.\n'
                '\n'
                'Assess the video along the following five world-consistency strata:\n'
                'W0 (perceptual and spatiotemporal fidelity): per-frame visual quality, sharpness, artifacts, and temporal flicker.\n'
                'W1 (object and relation structure): whether the required objects, attributes, counts, and spatial relations appear and persist.\n'
                'W2 (event and interaction dynamics): whether the described actions, motions, and interactions happen as described.\n'
                'W3 (intuitive physical plausibility): whether motion, contact, gravity, fluids, and deformation look physically plausible.\n'
                'W4 (temporal-causal coherence): whether events happen in a sensible order with consistent causes and effects.\n'
                '\n'
                'In <reason>...</reason>, give your reasoning. In <evaluate>...</evaluate>, give one line of feedback for each stratum W0 to W4, then name the primary failing stratum.\n'
                '\n'
                'Your ENTIRE response must be EXACTLY this format, with nothing before, after, or between the tags except line breaks:\n'
                '<reason>your reasoning process here</reason>\n'
                '<evaluate>\n'
                'W0: your assessment of perceptual and spatiotemporal fidelity\n'
                'W1: your assessment of object and relation structure\n'
                'W2: your assessment of event and interaction dynamics\n'
                'W3: your assessment of intuitive physical plausibility\n'
                'W4: your assessment of temporal-causal coherence\n'
                'Primary: the single stratum (W0, W1, W2, W3, or W4) with the most severe problem\n'
                '</evaluate>\n'
                '<complete>end</complete>\n'
                '\n'
                'Rules:\n'
                '- You must output <complete>end</complete> and must NOT output <generate>.\n'
                '- Inside <evaluate>, write exactly the six lines W0:, W1:, W2:, W3:, W4:, and Primary:, each non-empty.\n'
                '- Do NOT write anything outside these tags.\n'
            )

        self.t2v_base_url = (gen_cfg.get('base_url', '') if gen_cfg else '')
        self.t2v_model = (gen_cfg.get('model', 'wan22-a14b') if gen_cfg else 'wan22-a14b')
        self.t2v_n_frames = int(gen_cfg.get('n_frames', 4)) if gen_cfg else 4
        self.t2v_threads = int(gen_cfg.get('parallel_tool_call_threads', 8)) if gen_cfg else 8

        print(f"[STAP] T2V backbone: {self.t2v_model} @ {self.t2v_base_url} | n_frames={self.t2v_n_frames}")
        print(f"[Prompt Set] user_prompt_after_generate: {self.user_prompt_after_generate}")

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        rank = int(os.environ.get("RANK", 0))
        cur_step = int(prompts.meta_info.get('global_steps', 0))
        _phase = "val" if prompts.meta_info.get('validate', False) else "step"
        print(f">>> [STAP] vllm_rollout_stap Rollout Starts ({_phase}{cur_step}) ...")

        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']
        batch_size = idx.size(0)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        input_prompt_generation_mask = torch.zeros_like(
            idx, dtype=attention_mask.dtype, device=attention_mask.device
        )

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1,
            }

        n = 1 if prompts.meta_info.get('validate', False) else self.config.n

        vllm_inputs = []
        multi_turn_response_mask = []
        prefix_prompt_lengths = []
        generate_tool_return_videos = []

        if 'multi_modal_data' in non_tensor_batch:
            _multi_modal_data_list = non_tensor_batch['multi_modal_data']
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'), _multi_modal_data_list):
                prefix_length = len(raw_prompt_ids)
                for _ in range(n):
                    vllm_inputs.append({
                        'prompt_token_ids': deepcopy(raw_prompt_ids),
                        'multi_modal_data': deepcopy(multi_modal_data),
                    })
                    multi_turn_response_mask.append([
                        torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)
                    ])
                    prefix_prompt_lengths.append(prefix_length)
                    generate_tool_return_videos.append([])

        else:
            for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids'):
                prefix_length = len(raw_prompt_ids)
                for _ in range(n):
                    vllm_inputs.append({
                        'prompt_token_ids': deepcopy(raw_prompt_ids),
                    })
                    multi_turn_response_mask.append([
                        torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)
                    ])
                    prefix_prompt_lengths.append(prefix_length)
                    generate_tool_return_videos.append([])


        to_generate = list(range(batch_size * n))
        worker_trajs_count = len(to_generate)
        id_img_draw_cnt = [0] * (batch_size * n)
        max_img_draw_round = self.config.generate.get("video_generate_limit", 2)
        save_dir = getattr(
            self.config.generate,
            "video_save_dir",
            os.environ.get("STAP_VIDEO_SAVE_DIR", "outputs/videos"),
        )
        os.makedirs(save_dir, exist_ok=True)

        if 'video_urls' not in non_tensor_batch:
            non_tensor_batch['video_urls'] = np.array([None] * (batch_size * n), dtype=object)
        if 'video_urls_per_round' not in non_tensor_batch:
            non_tensor_batch['video_urls_per_round'] = np.empty(batch_size * n, dtype=object)
            for _k in range(batch_size * n):
                non_tensor_batch['video_urls_per_round'][_k] = []

        with tqdm(total=worker_trajs_count, desc="Worker Rollout Progress", unit="task") as pbar:
            current_iteration = 0
            max_iterations = self.config.max_gen_round
            while current_iteration < max_iterations and len(to_generate) > 0:
                idx_to_gen = []
                for i in to_generate:
                    idx_to_gen.append(vllm_inputs[i])

                print(
                    f"[Round #{current_iteration} Rollout START] For THIS round, We hava {len(idx_to_gen)} trajs to complete ..."
                )

                with self.update_sampling_params(n=1):
                    outputs = self.inference_engine.generate(
                        prompts=idx_to_gen, sampling_params=self.sampling_params, use_tqdm=False
                    )


                response = []
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        _token_ids = output.outputs[sample_id].token_ids
                        filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]
                        if 151645 not in filtered_token_ids:
                            filtered_token_ids[-1] = 151645

                        response.append(filtered_token_ids)

                assert len(to_generate) == len(response)

                idx_to_remove = []
                id_paint_query_mapping = {}
                reached_limit_count = 0
                for i_gen, response_ in zip(to_generate, response):
                    response_ = list(response_)
                    vllm_inputs[i_gen]['prompt_token_ids'] += response_
                    multi_turn_response_mask[i_gen].append(
                        torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device)
                    )

                    decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)

                    generate_match = re.search(r'<generate>(.*?)</generate>', decoded_resp_, re.DOTALL)
                    complete_match = re.search(r'<complete>(.*?)</complete>', decoded_resp_, re.DOTALL)
                    if generate_match and not complete_match:
                        assert str(i_gen) not in id_paint_query_mapping.keys()
                        if id_img_draw_cnt[i_gen] >= max_img_draw_round or current_iteration == max_iterations - 1:
                            idx_to_remove.append(i_gen)
                            reached_limit_count += 1
                            continue
                        paint_prompt = None
                        for match in re.finditer(r'<generate>(.*?)</generate>', decoded_resp_, re.DOTALL):
                            paint_prompt = match.group(1).strip()
                        if paint_prompt:
                            id_paint_query_mapping[str(i_gen)] = paint_prompt
                            id_img_draw_cnt[i_gen] += 1
                        else:
                            print(f"[Round #{current_iteration} Rollout ERROR] Empty <generate> tag, traj {i_gen} will be removed.")
                            idx_to_remove.append(i_gen)
                    else:
                        idx_to_remove.append(i_gen)

                print(
                f"[Round #{current_iteration} Rollout Paint Trigger] "
                f"For THIS round, we need to generate images for {len(id_paint_query_mapping)} trajs ..."
            )
                if reached_limit_count > 0:
                    print(f"[Round #{current_iteration}] {reached_limit_count} trajs reached max_img_draw_round {max_img_draw_round}")


                newly_completed = len(idx_to_remove)
                for x in idx_to_remove:
                    to_generate.remove(x)

                print(
                    f"[Round #{current_iteration} Rollout END] Completed {newly_completed} trajs, "
                    f"{len(to_generate)} trajs remaining for next round ..."
                )

                pbar.update(newly_completed)

                if len(to_generate) == 0:
                    current_iteration += 1
                    break

                
                prompts_batch = [id_paint_query_mapping[str(i)] for i in to_generate]
                print(f"[Round #{current_iteration} Generate START] Batch generating {len(prompts_batch)} videos "
                      f"(max_concurrent={self.t2v_threads}) @ {self.t2v_model} ...")
                generate_result = call_video_generate_batch(
                    prompts=prompts_batch,
                    base_url=self.t2v_base_url,
                    model=self.t2v_model,
                    n_frames=self.t2v_n_frames,
                    max_concurrent=self.t2v_threads,
                )
                success_count = sum(1 for _, _, s in generate_result if s["success"])
                print(f"[Round #{current_iteration} Generate END] success: {success_count}/{len(to_generate)}")

                to_generate_ = to_generate.copy()


                for i_gen_, generate_result_ in zip(to_generate_, generate_result):
                    generate_result_txt, generate_result_frames, tool_stat = generate_result_

                    if tool_stat["success"] and tool_stat.get("mp4_bytes"):
                        rank = int(os.environ.get("RANK", 0))
                        prompt_hash = hashlib.md5(tool_stat["prompt"].encode()).hexdigest()[:8]
                        video_save_path = os.path.join(
                            save_dir,
                            f"{_phase}{cur_step:03d}_{datetime.now().strftime('%m%d_%H%M%S')}_r{current_iteration}_t{i_gen_:03d}_rank{rank:02d}_{prompt_hash}.mp4"
                        )
                        with open(video_save_path, "wb") as _vf:
                            _vf.write(tool_stat["mp4_bytes"])

                        if 'video_urls' not in non_tensor_batch:
                            non_tensor_batch['video_urls'] = np.array([None] * (batch_size * n), dtype=object)
                        non_tensor_batch['video_urls'][i_gen_] = video_save_path

                        if 'video_urls_per_round' not in non_tensor_batch:
                            non_tensor_batch['video_urls_per_round'] = np.empty(batch_size * n, dtype=object)
                            for _k in range(batch_size * n):
                                non_tensor_batch['video_urls_per_round'][_k] = []
                        _cur_list = non_tensor_batch['video_urls_per_round'][i_gen_]
                        if not isinstance(_cur_list, list):
                            _cur_list = list(_cur_list)
                        _cur_list.append(video_save_path)
                        non_tensor_batch['video_urls_per_round'][i_gen_] = _cur_list

                        print(f"[{_phase}{cur_step} Round #{current_iteration}] traj {i_gen_} video saved -> {video_save_path}")

                    next_iter = current_iteration + 1
                    is_last_round = (next_iter >= max_iterations - 1) or (id_img_draw_cnt[i_gen_] >= max_img_draw_round)
                    after_prompt = self.user_prompt_last_round if is_last_round else self.user_prompt_after_generate
                    if (
                        "[Video Generation Results]" in generate_result_txt
                        and "There is an error" not in generate_result_txt
                        and after_prompt is not None
                    ):
                        next_user_message = (
                            generate_result_txt
                            + after_prompt
                        )
                        next_user_message = (
                            "<|im_start|>user\n" + next_user_message + "<|im_end|>\n<|im_start|>assistant\n"
                        )
                    else:
                        next_user_message = (
                            "<|im_start|>user\n" + generate_result_txt + "<|im_end|>\n<|im_start|>assistant\n"
                        )

                    next_turn_prompt_ids = self.tokenizer.encode(next_user_message)

                    if generate_result_frames:
                        if 'multi_modal_data' not in vllm_inputs[i_gen_]:
                            vllm_inputs[i_gen_]['multi_modal_data'] = {'video': []}
                        elif 'video' not in vllm_inputs[i_gen_]['multi_modal_data']:
                            vllm_inputs[i_gen_]['multi_modal_data']['video'] = []

                        video_arr = np.asarray(generate_result_frames)

                        vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids
                        vllm_inputs[i_gen_]['multi_modal_data']['video'].append(video_arr)
                        generate_tool_return_videos[i_gen_].append(video_arr)
                    else:
                        vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids

                    multi_turn_response_mask[i_gen_].append(
                        torch.zeros(
                            len(next_turn_prompt_ids),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                    )

                current_iteration += 1

        response = []
        response_generation_mask = []

        for i_ in range(batch_size * n):
            all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0)
            resp_mask_device = all_response_masks.device

            first_round_prompt_length = prefix_prompt_lengths[i_]
            response_after_prompt = vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]

            if generate_tool_return_videos[i_]:
                generated_video_inputs = self.processor.image_processor(
                    images=None, videos=generate_tool_return_videos[i_], return_tensors='pt'
                )
                generated_video_grid_thw = generated_video_inputs['video_grid_thw']

                if generated_video_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size ** 2
                    video_pad_token, magic_num = 151656, 654321
                    all_response_masks = all_response_masks.tolist()

                    index = 0
                    while video_pad_token in response_after_prompt:
                        pos = response_after_prompt.index(video_pad_token)
                        replicate_count = generated_video_grid_thw[index].prod() // merge_length
                        response_after_prompt[pos: pos + 1] = [magic_num] * replicate_count
                        all_response_masks[pos: pos + 1] = [0] * replicate_count
                        index += 1

                    response_after_prompt = [
                        video_pad_token if x == magic_num else x
                        for x in response_after_prompt
                    ]
                    all_response_masks = torch.tensor(
                        all_response_masks, dtype=torch.int64, device=resp_mask_device
                    )

            response_generation_mask.append(all_response_masks)
            all_response = torch.tensor(response_after_prompt, device=idx.device, dtype=idx.dtype)
            response.append(all_response)
            assert (
                response[i_].shape[0] == response_generation_mask[i_].shape[0]
            ), (
                f"shape mismatched | "
                f"response[i_]: {response[i_].shape[0]} | "
                f"response_generation_mask[i_]: {response_generation_mask[i_].shape[0]}"
            )

        assert len(response) == len(response_generation_mask), \
            "length mismatched between response and response_generation_mask!"

        response = pad_to_max_stack(
            response, self.pad_token_id, dim=0
        )
        response_generation_mask = pad_to_max_stack(
            response_generation_mask, 0, dim=0
        )
        assert all([
            response.size(dim) == response_generation_mask.size(dim)
            for dim in range(response.ndim)
        ])

        if response.shape[1] > self.config.response_length_total:
            response = response[:, :self.config.response_length_total]
            response_generation_mask = response_generation_mask[:, :self.config.response_length_total]
        elif response.shape[1] < self.config.response_length_total:
            response = pad_sequence_to_length(
                response, self.config.response_length_total, self.pad_token_id
            )
            response_generation_mask = pad_sequence_to_length(
                response_generation_mask, self.config.response_length_total, 0
            )

        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n


            input_prompt_generation_mask = _repeat_interleave(
                input_prompt_generation_mask, self.config.n
            )

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_final_eos_mask(
            response_id=response, eos_token=[151645], dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        multi_turn_response_mask = torch.cat(
            [input_prompt_generation_mask, response_generation_mask], dim=-1
        )

        batch = TensorDict(
            {
                'prompts': idx.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous(),
            },
            batch_size=batch_size,
        )

        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        print(">>> vllm_rollout_spmd Rollout Ends ...")
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
