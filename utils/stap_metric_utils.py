import numpy as np

W_LAYERS = ["W0", "W1", "W2", "W3", "W4"]


def extract_stap_metrics(batch) -> list:
    if 'extra_info' not in batch.non_tensor_batch:
        return []
    return [ei['_stap_metrics'] for ei in batch.non_tensor_batch['extra_info']
            if isinstance(ei, dict) and isinstance(ei.get('_stap_metrics'), dict)]


def aggregate_stap_metrics_from_list(all_m, prefix='STAP/') -> dict:
    result = {}
    if not all_m:
        return result
    n = len(all_m)

    for key in ['reward', 'format_score', 'r_Q', 'r_CLIP', 'r_HPS']:
        vals = [m[key] for m in all_m if m.get(key) is not None]
        if vals:
            result[f'{prefix}{key}/mean'] = float(np.mean(vals))
            result[f'{prefix}{key}/max'] = float(np.max(vals))
            result[f'{prefix}{key}/min'] = float(np.min(vals))

    pc = [m.get('generate_count', 0) for m in all_m]
    if pc:
        result[f'{prefix}paint_count/mean'] = float(np.mean(pc))
        result[f'{prefix}paint_count/max'] = float(np.max(pc))

    result[f'{prefix}format_perfect_ratio'] = sum(1 for m in all_m if m.get('format_score', 0.0) >= 1.0) / n

    for w in W_LAYERS:
        vals = [m['r_Q_per_layer'][w] for m in all_m
                if isinstance(m.get('r_Q_per_layer'), dict) and m['r_Q_per_layer'].get(w) is not None]
        if vals:
            result[f'{prefix}by_layer/{w}/mean'] = float(np.mean(vals))

    by_prim = {}
    for m in all_m:
        lay = m.get('layer')
        if lay in W_LAYERS:
            by_prim.setdefault(lay, []).append(m.get('reward', 0.0))
    for w, vals in by_prim.items():
        result[f'{prefix}by_primary/{w}/reward_mean'] = float(np.mean(vals))
        result[f'{prefix}by_primary/{w}/count'] = len(vals)

    result[f'{prefix}num_samples'] = n
    return result


def aggregate_stap_metrics(batch, prefix='STAP/') -> dict:
    return aggregate_stap_metrics_from_list(extract_stap_metrics(batch), prefix)


def qwen_tool_persample(batch) -> dict:
    out = {'input': [], 'output': [], 'total': [], 'wan22_calls': []}
    try:
        n = len(batch)
    except Exception:
        n = 0
    if n == 0:
        return out

    bt = batch.batch
    if bt is not None and 'attention_mask' in bt and 'responses' in bt:
        attn = bt['attention_mask']
        max_resp = bt['responses'].shape[-1]
        input_len = attn[:, :-max_resp].sum(-1)
        if 'multi_turn_response_mask' in bt:
            output_len = (bt['multi_turn_response_mask'] == 1).sum(-1)
        else:
            output_len = attn[:, -max_resp:].sum(-1)

        def _to_list(t):
            for conv in (lambda x: x.detach().cpu().tolist(), lambda x: x.tolist(), list):
                try:
                    return [float(v) for v in conv(t)]
                except Exception:
                    continue
            return []

        input_len = _to_list(input_len)
        output_len = _to_list(output_len)
        out['input'] = input_len
        out['output'] = output_len
        out['total'] = [a + b for a, b in zip(input_len, output_len)]

    vurls = batch.non_tensor_batch.get('video_urls_per_round')
    if vurls is None:
        vurls = batch.non_tensor_batch.get('video_urls')
    if vurls is not None:
        calls = []
        for v in vurls:
            if isinstance(v, (list, tuple)):
                calls.append(len(v))
            elif v is None or isinstance(v, float):
                calls.append(0)
            else:
                calls.append(1)
        out['wan22_calls'] = calls
    return out


def aggregate_qwen_tool(persample, prefix='') -> dict:
    res = {}
    ip = persample.get('input', [])
    op = persample.get('output', [])
    tt = persample.get('total', [])
    wc = persample.get('wan22_calls', [])
    if ip:
        res[f'{prefix}qwen7b/input_tokens/mean'] = float(np.mean(ip))
        res[f'{prefix}qwen7b/output_tokens/mean'] = float(np.mean(op))
        res[f'{prefix}qwen7b/total_tokens/mean'] = float(np.mean(tt))
        res[f'{prefix}qwen7b/num_samples'] = len(ip)
    if wc:
        res[f'{prefix}wan22/calls/mean'] = float(np.mean(wc))
        res[f'{prefix}wan22/calls/total'] = int(np.sum(wc))
    return res

