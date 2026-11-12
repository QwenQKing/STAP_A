import json

import torch
from verl import DataProto

from STAP.workers.multimodal.reward.reward_score_mm import _default_compute_score


class STAP_RewardManager:

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score

    def extract_responses_list(
        self,
        tokenizer,
        input_ids: torch.Tensor,
        multi_turn_response_mask: torch.Tensor,
    ) -> list:
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat(
            [multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0
        )
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0] - 1
        segments = []
        for s, e in zip(starts, ends):
            segments.append(input_ids[s : e + 1].tolist())

        decoded_responses = tokenizer.batch_decode(segments, skip_special_tokens=True)
        return decoded_responses

    def __call__(self, data: DataProto):

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        import os
        from concurrent.futures import ThreadPoolExecutor
        already_print_data_sources = {}

        items = []
        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str = [response_str]
            if 'multi_turn_response_mask' in data_item.batch:
                response_str = self.extract_responses_list(
                    self.tokenizer, data_item.batch['input_ids'], data_item.batch['multi_turn_response_mask']
                )

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]
            if 'candidate_answers' in data_item.non_tensor_batch['reward_model']:
                candidate_answers = data_item.non_tensor_batch['reward_model']['candidate_answers']
                if isinstance(candidate_answers, list):
                    ground_truth += candidate_answers
                elif isinstance(candidate_answers, str):
                    ground_truth += json.loads(candidate_answers)
                else:
                    raise TypeError(f"candidate_answers must be a list or a string, but got {type(candidate_answers)}")
            ground_truth = [g for g in ground_truth if isinstance(g, str)]
            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            _ei = dict(extra_info) if extra_info is not None else {}
            if 'video_urls' in data_item.non_tensor_batch:
                _ei['video_urls'] = data_item.non_tensor_batch['video_urls']
            if 'video_urls_per_round' in data_item.non_tensor_batch:
                _ei['video_urls_per_round'] = data_item.non_tensor_batch['video_urls_per_round']

            items.append({'i': i, 'vlen': int(valid_response_length), 'response_str': response_str,
                          'ground_truth': ground_truth, 'data_source': data_source,
                          'extra_info': _ei, 'prompt_str': prompt_str})

        def _work(it):
            it['score'] = self.compute_score(
                data_source=it['data_source'], solution_str=it['response_str'],
                ground_truth=it['ground_truth'], extra_info=it['extra_info'])
            return it

        n_workers = int(os.environ.get('STAP_REWARD_WORKERS', '32'))
        if n_workers > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=min(n_workers, len(items))) as ex:
                items = list(ex.map(_work, items))
        else:
            items = [_work(it) for it in items]

        ei_arr = data.non_tensor_batch.get('extra_info')
        for it in items:
            i = it['i']
            reward_tensor[i, it['vlen'] - 1] = it['score']
            _wm = it['extra_info'].get('_stap_metrics') if isinstance(it['extra_info'], dict) else None
            if _wm is not None and ei_arr is not None and i < len(ei_arr) and isinstance(ei_arr[i], dict):
                ei_arr[i]['_stap_metrics'] = _wm

            ds = it['data_source']
            if ds not in already_print_data_sources:
                already_print_data_sources[ds] = 0
            if already_print_data_sources[ds] < self.num_examine:
                already_print_data_sources[ds] += 1
                print("[prompt]", it['prompt_str'])
                print("[response]", it['response_str'])
                print("[video_urls_per_round]", it['extra_info'].get('video_urls_per_round'))
                print("[ground_truth]", it['ground_truth'])
                print("[score]", it['score'])

        return reward_tensor

