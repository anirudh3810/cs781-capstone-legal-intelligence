"""Self-contained v3 training, model selection, and packaging stages."""

# %% 6 - verify v2 artifacts and construct a larger span model
import pickle
import numpy as np

V2_CONFIG = read_json(V2_ROOT / 'experiment_config.json')
V2_HASH = hashlib.sha256(json.dumps(V2_CONFIG, sort_keys=True).encode()).hexdigest()
for key, current in [
    ('base_checkpoint_sha256', base_sha), ('model_source_sha256', model_sha),
    ('bpe_source_sha256', bpe_sha),
    ('tokenizer_fingerprint', tokenizer_meta['fingerprint']), ('data_hash', data_hash),
]:
    if V2_CONFIG.get(key) != current:
        raise RuntimeError(f'The v2 experiment has a different {key}')
V2_REPORT = read_json(V2_ROOT / 'report.json')
V2_COMPARISON = read_json(V2_ROOT / 'model_comparison.json')
V2_FINAL = read_json(V2_ROOT / 'final' / 'summary.json')
V2_TEST_PATH = V2_ROOT / 'final' / 'test_candidates.partial.json'
if (V2_REPORT.get('status') != 'READY_FOR_MANUAL_UPLOAD' or
        V2_FINAL.get('status') != 'complete' or not V2_TEST_PATH.is_file()):
    raise RuntimeError('Complete the v2 Run All notebook in this same Kaggle session first')
V2_BASELINE = float(V2_COMPARISON['reranker_calibrated_oof'])
if abs(V2_BASELINE - 0.45297112108361587) > 1e-7:
    raise RuntimeError('Unexpected v2 out-of-fold baseline; inspect artifacts before continuing')


def selected_v2(fold_index):
    folder = V2_ROOT / f'fold{fold_index}'
    best = read_json(folder / 'best.json')
    if best.get('config_hash') != V2_HASH or best.get('fold') != fold_index:
        raise RuntimeError(f'Fold {fold_index}: v2 selected checkpoint provenance mismatch')
    adapter = folder / best['adapter_file']
    candidate_file = folder / best['candidates_file']
    if (file_sha(adapter) != best['adapter_sha256'] or
            file_sha(candidate_file) != best['candidates_sha256']):
        raise RuntimeError(f'Fold {fold_index}: v2 selected artifact changed')
    payload = read_json(candidate_file)
    if payload.get('config_hash') != V2_HASH or set(payload['candidates']) != set(FOLDS[fold_index]):
        raise RuntimeError(f'Fold {fold_index}: v2 candidate provenance changed')
    score = strict_macro(payload['candidates'], FOLDS[fold_index])['macro_f1']
    if abs(score - best['default_macro_f1']) > 1e-8:
        raise RuntimeError(f'Fold {fold_index}: v2 selected score did not reproduce')
    return adapter, payload['candidates'], best


V2_FOLDS = [selected_v2(i) for i in range(3)]
V2_OOF = {}
for _, candidates, _ in V2_FOLDS:
    V2_OOF.update(candidates)
if set(V2_OOF) != set(texts):
    raise RuntimeError('V2 fold candidates do not cover every labeled document')
v2_reproduced = strict_macro(V2_OOF, texts, V2_COMPARISON['v2_thresholds'])['macro_f1']
if abs(v2_reproduced - V2_COMPARISON['v2_calibrated_oof']) > 1e-8:
    raise RuntimeError('The saved v2 calibrated F1 did not reproduce')
print('V2 verified raw OOF:', f'{v2_reproduced:.6f}',
      'v2 reranked OOF:', f'{V2_BASELINE:.6f}')


class V3SpanExtractor(EnhancedSpanExtractor):
    def __init__(self, base):
        super().__init__(base)
        hidden = base.hidden_size
        self.biaffine_start = nn.Linear(hidden, 128, bias=False)
        self.biaffine_end = nn.Linear(hidden, 128, bias=False)
        self.biaffine_out = nn.Linear(128, len(LABELS))
        self.pair_residual = nn.Sequential(
            nn.Linear(5 * hidden + 16, CONFIG['v3_residual_hidden']),
            nn.GELU(), nn.Dropout(0.20),
            nn.Linear(CONFIG['v3_residual_hidden'], len(LABELS)))
        nn.init.zeros_(self.biaffine_out.weight)
        nn.init.zeros_(self.biaffine_out.bias)
        nn.init.zeros_(self.pair_residual[-1].weight)
        nn.init.zeros_(self.pair_residual[-1].bias)
        for block in base.blocks[-CONFIG['v3_unfreeze_top_blocks']:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)

    def score_pairs(self, hidden, pair_index):
        original = super().score_pairs(hidden, pair_index)
        width = hidden.shape[-1]
        def take(index):
            return hidden.gather(1, index.unsqueeze(-1).expand(-1, -1, width))
        start_index, end_index = pair_index[..., 0], pair_index[..., 1]
        start, end = take(start_index), take(end_index)
        before = take((start_index - 1).clamp_min(0))
        after = take((end_index + 1).clamp_max(hidden.shape[1]-1))
        distance = (end_index-start_index).clamp(min=0, max=CONFIG['max_span_tokens'])
        features = torch.cat((start, end, before, after, start*end,
                              self.width_embedding(distance)), dim=-1)
        interaction = self.biaffine_start(start) * self.biaffine_end(end)
        return original + self.biaffine_out(interaction) + self.pair_residual(features)


def v3_model(v2_adapter):
    record = torch.load(BASE_CHECKPOINT, map_location='cpu', weights_only=True)
    if record.get('model_source_sha256') != model_sha:
        raise RuntimeError('The Phase-1 base changed before v3 construction')
    base = model_module.build_model(record['model_config'])
    base.load_state_dict(record['model_state'], strict=True)
    del record
    model = V3SpanExtractor(base)
    saved = torch.load(v2_adapter, map_location='cpu', weights_only=True)
    if saved.get('config_hash') != V2_HASH:
        raise RuntimeError('The v3 warm adapter is not from the verified v2 run')
    current = model.state_dict()
    for name, value in saved['trainable_state'].items():
        if name not in current or current[name].shape != value.shape:
            raise RuntimeError(f'V2 warm state mismatch: {name}')
        current[name] = value
    model.load_state_dict(current, strict=True)
    print('V3 trainable parameters:',
          f'{sum(p.numel() for p in model.parameters() if p.requires_grad):,}',
          'of', f'{sum(p.numel() for p in model.parameters()):,}')
    return model


def verify_v3_best(folder, fold_index):
    path = folder / 'best.json'
    if not path.is_file():
        return None
    best = read_json(path)
    if best.get('config_hash') != CONFIG_HASH or best.get('fold') != fold_index:
        raise RuntimeError(f'Fold {fold_index}: v3 best provenance mismatch')
    for field, sha_field in [('adapter_file', 'adapter_sha256'),
                             ('candidates_file', 'candidates_sha256')]:
        if file_sha(folder / best[field]) != best[sha_field]:
            raise RuntimeError(f'Fold {fold_index}: v3 best SHA mismatch')
    candidates = read_json(folder / best['candidates_file'])
    if candidates.get('config_hash') != CONFIG_HASH or set(candidates['candidates']) != set(FOLDS[fold_index]):
        raise RuntimeError(f'Fold {fold_index}: v3 candidate IDs changed')
    score = strict_macro(candidates['candidates'], FOLDS[fold_index])['macro_f1']
    if abs(score - best['default_macro_f1']) > 1e-8:
        raise RuntimeError(f'Fold {fold_index}: v3 score did not reproduce')
    return best


# %% 7 - two-GPU continuation, ranking loss, and resumable fold training
def v3_parameter_groups(model):
    groups = {'new': [], 'old_head': [], 'lora': [], 'new_encoder': [], 'old_encoder': []}
    cutoff = len(model.base.blocks) - CONFIG['unfreeze_top_blocks']
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(('biaffine_start.', 'biaffine_end.', 'biaffine_out.',
                            'pair_residual.')):
            key = 'new'
        elif not name.startswith('base.'):
            key = 'old_head'
        elif '.a.' in name or '.b.' in name:
            key = 'lora'
        else:
            try:
                layer = int(name.split('.')[2]) if name.startswith('base.blocks.') else -1
            except (ValueError, IndexError):
                layer = -1
            key = 'old_encoder' if layer >= cutoff else 'new_encoder'
        groups[key].append(parameter)
    if any(not values for values in groups.values()):
        raise RuntimeError('One or more v3 trainable parameter groups are empty')
    rates = {'new': 1.2e-4, 'old_head': 4e-5, 'lora': 2e-5,
             'new_encoder': 8e-6, 'old_encoder': 4e-6}
    return [{'params': values, 'lr': rates[name], 'initial_lr': rates[name]}
            for name, values in groups.items()]


def exact_pair_rank_loss(output, batch):
    logits = output['pair'].float()
    positive = batch['pair_targets'].bool() & batch['pair_mask'].unsqueeze(-1)
    negative = (~batch['pair_targets'].bool()) & batch['pair_mask'].unsqueeze(-1)
    hardest = logits.masked_fill(~negative, -1e4).max(dim=1, keepdim=True).values
    values = F.softplus(0.50 + hardest - logits)
    return (values * positive).sum() / positive.sum().clamp_min(1)


def v3_loss(output, batch, class_weights):
    return task_loss(output, batch, class_weights) + 0.18 * exact_pair_rank_loss(output, batch)


def train_v3_fold(fold_index):
    folder = ROOT / f'fold{fold_index}'
    folder.mkdir(parents=True, exist_ok=True)
    summary_path, latest_path = folder/'summary.json', folder/'checkpoint_latest.pt'
    if summary_path.is_file():
        saved_summary = read_json(summary_path)
        if saved_summary.get('config_hash') != CONFIG_HASH:
            raise RuntimeError(f'Fold {fold_index}: incompatible resume summary')
        if saved_summary.get('status') == 'complete':
            verify_v3_best(folder, fold_index)
            print(f'Fold {fold_index}: verified complete')
            return saved_summary
        if saved_summary.get('status') == 'paused' and not latest_path.is_file():
            raise RuntimeError(f'Fold {fold_index}: paused without optimizer checkpoint')
    warm_adapter, old_candidates, old_best = V2_FOLDS[fold_index]
    heldout = FOLDS[fold_index]
    train_ids = set(texts) - set(heldout)
    rows = make_windows(train_ids, training=True)
    model = v3_model(warm_adapter).to('cuda:0')
    parallel = nn.DataParallel(model, device_ids=[0,1], output_device=0)
    optimizer = torch.optim.AdamW(v3_parameter_groups(model), weight_decay=0.01, foreach=False)
    total_steps = MAX_EPOCHS * math.ceil(len(rows)/CONFIG['batch_size'])
    warmup = max(1, int(total_steps*.04))
    def factor(step):
        if step < warmup:
            return max(.10, (step+1)/warmup)
        progress = (step-warmup)/max(1,total_steps-warmup)
        return max(.15, .5*(1+math.cos(math.pi*min(1,progress))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    weights = label_weights(train_ids).to('cuda:0')
    best = verify_v3_best(folder, fold_index)
    best_score = best['default_macro_f1'] if best else float(old_best['default_macro_f1'])
    history, epoch_start, stale = [],0,0
    warm_sha = file_sha(warm_adapter)
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location='cpu', weights_only=False)
        if (saved.get('config_hash') != CONFIG_HASH or saved.get('fold') != fold_index or
                saved.get('warm_sha256') != warm_sha or saved.get('windows') != len(rows)):
            raise RuntimeError(f'Fold {fold_index}: optimizer provenance mismatch')
        load_trainable_state(model, saved['trainable_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        random.setstate(saved['python_rng_state'])
        history, epoch_start, stale = saved['history'], saved['epoch'], saved['stale']
        print(f'Fold {fold_index}: resuming after epoch {epoch_start}')
    paused=False
    try:
        for epoch in range(epoch_start+1, MAX_EPOCHS+1):
            if time.time()-RUN_STARTED > SESSION_GUARD_SECONDS:
                paused=True
                print('Time guard reached; Run All again in the same session to resume.')
                break
            order=list(range(len(rows)))
            random.Random(SEED + 300_000 + fold_index*1000 + epoch).shuffle(order)
            model.train()
            losses=[]
            for begin in range(0,len(order),CONFIG['batch_size']):
                group=[rows[i] for i in order[begin:begin+CONFIG['batch_size']]]
                batch={name:value.to('cuda:0',non_blocking=True)
                       for name,value in collate_windows(group,epoch+900).items()}
                optimizer.zero_grad(set_to_none=True)
                output=(parallel(batch['input_ids'],batch['attention_mask'],batch['pair_index'])
                        if len(group)>1 else
                        model(batch['input_ids'],batch['attention_mask'],batch['pair_index']))
                loss=v3_loss(output,batch,weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Fold {fold_index} epoch {epoch}: nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step();scheduler.step()
                losses.append(float(loss.detach().cpu()))
            entry={'epoch':epoch,'train_loss_mean':sum(losses)/len(losses),
                   'elapsed_seconds':round(time.time()-RUN_STARTED,1)}
            if epoch%VALIDATE_EVERY==0 or epoch==MAX_EPOCHS:
                candidates=predict_documents(model,parallel,heldout,texts)
                score_report=strict_macro(candidates,heldout)
                score=score_report['macro_f1']
                entry['heldout_default_macro_f1']=score
                entry['candidate_recall_macro']=sum(
                    row['exact_candidate_recall'] for row in score_report['per_label'].values())/len(LABELS)
                if score>best_score+0.001:
                    adapter_name=f'selected_epoch{epoch:02d}.pt'
                    candidates_name=f'candidates_epoch{epoch:02d}.json'
                    atomic_torch(folder/adapter_name,{'config_hash':CONFIG_HASH,'fold':fold_index,
                                                      'epoch':epoch,'trainable_state':trainable_state(model),
                                                      'labels':LABELS})
                    atomic_json(folder/candidates_name,{'config_hash':CONFIG_HASH,'fold':fold_index,
                                                        'epoch':epoch,'candidates':candidates})
                    proposed={'config_hash':CONFIG_HASH,'fold':fold_index,'epoch':epoch,
                              'default_macro_f1':score,'adapter_file':adapter_name,
                              'adapter_sha256':file_sha(folder/adapter_name),
                              'candidates_file':candidates_name,
                              'candidates_sha256':file_sha(folder/candidates_name)}
                    previous=best
                    atomic_json(folder/'best.json',proposed)
                    best=verify_v3_best(folder,fold_index)
                    if previous is not None:
                        (folder/previous['adapter_file']).unlink(missing_ok=True)
                        (folder/previous['candidates_file']).unlink(missing_ok=True)
                    best_score=score;stale=0
                else:
                    stale+=1
                print(f'Fold {fold_index} epoch {epoch}: loss {entry["train_loss_mean"]:.4f}, '
                      f'F1 {score:.4f}, best {best_score:.4f}, '
                      f'free {free_gib():.2f} GiB',flush=True)
            else:
                print(f'Fold {fold_index} epoch {epoch}: loss {entry["train_loss_mean"]:.4f}',flush=True)
            history.append(entry)
            atomic_json(folder/'history.json',history)
            if free_gib()<MIN_FREE_GIB:
                raise RuntimeError('Disk reserve reached before v3 optimizer checkpoint')
            atomic_torch(latest_path,{'config_hash':CONFIG_HASH,'fold':fold_index,
                                      'warm_sha256':warm_sha,'windows':len(rows),
                                      'epoch':epoch,'stale':stale,'history':history,
                                      'trainable_state':trainable_state(model),
                                      'optimizer_state':optimizer.state_dict(),
                                      'scheduler_state':scheduler.state_dict(),
                                      'torch_rng_state':torch.get_rng_state(),
                                      'cuda_rng_states':torch.cuda.get_rng_state_all(),
                                      'python_rng_state':random.getstate()})
            if epoch>=MIN_EPOCHS and stale>=PATIENCE:
                print(f'Fold {fold_index}: early stopping after {stale} non-improving validations')
                break
        if paused and not latest_path.is_file():
            raise RuntimeError('Time guard reached before a resumable checkpoint was made')
        best=verify_v3_best(folder,fold_index)
        result={'status':'paused' if paused else 'complete','config_hash':CONFIG_HASH,
                'fold':fold_index,'warm_default_macro_f1':old_best['default_macro_f1'],
                'selected_source':'v3' if best else 'v2',
                'selected_epoch':best['epoch'] if best else 0,
                'selected_default_macro_f1':best_score,'history':history}
        atomic_json(summary_path,result)
        if not paused:latest_path.unlink(missing_ok=True)
        return result
    finally:
        del model,parallel,optimizer,scheduler
        torch.cuda.empty_cache()


FOLD_RESULTS={}
for fold_index in range(3):
    if time.time()-RUN_STARTED>SESSION_GUARD_SECONDS:
        print('Time guard before another fold; Run All again to resume.')
        break
    FOLD_RESULTS[fold_index]=train_v3_fold(fold_index)
    if FOLD_RESULTS[fold_index]['status']!='complete':break


# %% 8 - cross-validated reranking and label-aware joint decoding
def selected_v3_candidates(fold_index):
    best=verify_v3_best(ROOT/f'fold{fold_index}',fold_index)
    if best is None:return V2_FOLDS[fold_index][1]
    return read_json(ROOT/f'fold{fold_index}'/best['candidates_file'])['candidates']


def load_v1_candidates():
    config=read_json(V1_ROOT/'experiment_config.json')
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if (config.get('base_checkpoint_sha256')!=base_sha or
            config.get('data_hash')!=data_hash):
        raise RuntimeError('V1 candidate source differs from the official data or base')
    output={}
    for fold_index in range(3):
        folder=V1_ROOT/f'fold{fold_index}'
        best=read_json(folder/'best.json')
        if best.get('config_hash')!=digest or best.get('fold')!=fold_index:
            raise RuntimeError(f'Fold {fold_index}: V1 candidate provenance mismatch')
        path=folder/best['candidates_file']
        if file_sha(path)!=best['candidates_sha256']:
            raise RuntimeError(f'Fold {fold_index}: V1 candidate SHA mismatch')
        group=read_json(path)['candidates']
        if set(group)!=set(FOLDS[fold_index]):
            raise RuntimeError(f'Fold {fold_index}: V1 candidate IDs mismatch')
        output.update(group)
    return output


def crossfit_reranker(lower_map, upper_map):
    X,y,rows,ranges=build_feature_table(texts,lower_map,upper_map,texts,with_gold=True)
    print('Cross-fit candidate feature table:',X.shape,'positive exact:',int(y.sum()),flush=True)
    output={}
    for fold_index in range(3):
        validation=FOLDS[fold_index]
        training=sorted(set(texts)-set(validation))
        model=train_reranker(X,y,ranges,training)
        output.update(rerank_documents(model,X,rows,ranges,validation))
        del model
        print('Cross-fit reranker fold',fold_index,'done',flush=True)
    if set(output)!=set(texts):
        raise RuntimeError('Cross-fitted reranker is missing labeled documents')
    del X,y,rows,ranges
    return output


def label_source_choices(sources, thresholds):
    reports={key:strict_macro(candidate_map,texts,thresholds[key])
             for key,candidate_map in sources.items()}
    choice={}
    for label in LABELS:
        baseline=reports['v2_reranked']['per_label'][label]['f1']
        candidate=max(sources,key=lambda key:reports[key]['per_label'][label]['f1'])
        advantage=reports[candidate]['per_label'][label]['f1']-baseline
        choice[label]=candidate if advantage>=.015 else 'v2_reranked'
    return choice,reports


def normalized_for_threshold(score,threshold):
    s=min(1-1e-6,max(1e-6,float(score)))
    t=min(1-1e-6,max(1e-6,float(threshold)))
    margin=math.log(s/(1-s))-math.log(t/(1-t))
    return 1/(1+math.exp(-max(-40,min(40,margin))))


def hybrid_candidates(sources,source_thresholds,choices,doc_ids):
    output={}
    for doc_id in doc_ids:
        selected=[]
        for source_name,candidate_map in sources.items():
            thresholds=source_thresholds[source_name]
            for a,b,label,score in candidate_map[doc_id]:
                if choices[label]!=source_name:continue
                confidence=normalized_for_threshold(score,thresholds[label])
                if confidence>=.5:
                    selected.append([int(a),int(b),label,float(confidence)])
        selected.sort(key=lambda row:-row[3])
        output[doc_id]=selected
    return output


SELECTION=None
V3_OOF=None
V2_RERANK_OOF=None
V3_RERANK_OOF=None
if len(FOLD_RESULTS)==3 and all(x['status']=='complete' for x in FOLD_RESULTS.values()):
    V3_OOF={}
    for fold_index in range(3):V3_OOF.update(selected_v3_candidates(fold_index))
    if set(V3_OOF)!=set(texts):
        raise RuntimeError('V3 OOF candidate map is incomplete')
    v3_default=strict_macro(V3_OOF,texts)['macro_f1']
    v3_thresholds,v3_calibrated=calibrate(V3_OOF)
    print('V3 selected-fold F1: default',f'{v3_default:.6f}',
          'calibrated',f'{v3_calibrated:.6f}',flush=True)

    V1_OOF=load_v1_candidates()
    V2_RERANK_OOF=crossfit_reranker(V1_OOF,V2_OOF)
    v2_rr_thresholds=V2_COMPARISON['reranker_thresholds']
    reproduced=fast_macro(V2_RERANK_OOF,v2_rr_thresholds)
    if abs(reproduced-V2_BASELINE)>1e-8:
        raise RuntimeError(f'The saved v2 cross-fit reranker score did not reproduce: {reproduced}')
    del V1_OOF

    V3_RERANK_OOF=crossfit_reranker(V2_OOF,V3_OOF)
    v3_rr_default=fast_macro(V3_RERANK_OOF,{})
    v3_rr_thresholds,v3_rr_calibrated=calibrate(V3_RERANK_OOF)
    sources={'v2_raw':V2_OOF,'v2_reranked':V2_RERANK_OOF,
             'v3_raw':V3_OOF,'v3_reranked':V3_RERANK_OOF}
    source_thresholds={
        'v2_raw':V2_COMPARISON['v2_thresholds'],
        'v2_reranked':v2_rr_thresholds,
        'v3_raw':v3_thresholds,'v3_reranked':v3_rr_thresholds}
    choices,reports=label_source_choices(sources,source_thresholds)
    hybrid=hybrid_candidates(sources,source_thresholds,choices,texts)
    common_thresholds={label:.5 for label in LABELS}
    hybrid_report=strict_macro(hybrid,texts,common_thresholds)
    hybrid_score=hybrid_report['macro_f1']
    alternatives={'v2_reranked':V2_BASELINE,'v3_raw':v3_calibrated,
                  'v3_reranked':v3_rr_calibrated,'hybrid':hybrid_score}
    method=max(alternatives,key=alternatives.get)
    fold_checks=[]
    for fold_index in range(3):
        ids=FOLDS[fold_index]
        fold_checks.append({
            'fold':fold_index,
            'v2_reranked':fast_macro({d:V2_RERANK_OOF[d] for d in ids},v2_rr_thresholds),
            'v3_raw':fast_macro({d:V3_OOF[d] for d in ids},v3_thresholds),
            'v3_reranked':fast_macro({d:V3_RERANK_OOF[d] for d in ids},v3_rr_thresholds),
            'hybrid':fast_macro({d:hybrid[d] for d in ids},common_thresholds)})
    SELECTION={
        'status':'complete','config_hash':CONFIG_HASH,
        'baseline_v2_reranked_oof':V2_BASELINE,
        'v2_raw_calibrated_oof':V2_COMPARISON['v2_calibrated_oof'],
        'v3_default_oof':v3_default,'v3_calibrated_oof':v3_calibrated,
        'v3_reranker_default_oof':v3_rr_default,
        'v3_reranker_calibrated_oof':v3_rr_calibrated,
        'hybrid_oof':hybrid_score,'chosen_method':method,
        'chosen_oof':alternatives[method],
        'v3_thresholds':v3_thresholds,
        'v3_reranker_thresholds':v3_rr_thresholds,
        'hybrid_label_sources':choices,
        'hybrid_source_thresholds':source_thresholds,
        'fold_checks':fold_checks,
        'per_label':{key:report['per_label'] for key,report in reports.items()},
        'per_label_hybrid':hybrid_report['per_label'],
        'caveat':'Fold checkpoint, per-label source, and threshold selection reuse OOF labels; local F1 is optimistic.'}
    atomic_json(ROOT/'model_selection.json',SELECTION)
    print('OOF baseline',f'{V2_BASELINE:.6f}','v3',f'{v3_calibrated:.6f}',
          'v3 reranked',f'{v3_rr_calibrated:.6f}',
          'hybrid',f'{hybrid_score:.6f}','chosen',method,flush=True)
    del hybrid,reports,choices,sources,V2_RERANK_OOF,V3_RERANK_OOF
else:
    print('The three v3 folds are incomplete. Run All in this session to resume.')


# %% 9 - continue the all-data v2 model and generate test candidates
def selected_final_epochs():
    epochs=sorted(int(FOLD_RESULTS[i]['selected_epoch']) for i in range(3)
                  if FOLD_RESULTS[i]['selected_source']=='v3')
    return max(2,min(MAX_EPOCHS,epochs[len(epochs)//2])) if epochs else 0


def train_v3_final(epochs):
    final_dir=ROOT/'final'
    final_dir.mkdir(parents=True,exist_ok=True)
    summary_path=final_dir/'summary.json'
    latest_path=final_dir/'checkpoint_latest.pt'
    adapter_path=final_dir/'selected_adapter.pt'
    warm_path=V2_ROOT/'final'/'selected_adapter.pt'
    warm_sha=file_sha(warm_path)
    if warm_sha!=V2_FINAL['adapter_sha256']:
        raise RuntimeError('V2 full-data warm adapter changed')
    if summary_path.is_file():
        previous=read_json(summary_path)
        if previous.get('config_hash')!=CONFIG_HASH or previous.get('epochs')!=epochs:
            raise RuntimeError('Incompatible v3 final-model resume summary')
        if previous['status']=='complete':
            if file_sha(adapter_path)!=previous['adapter_sha256']:
                raise RuntimeError('The completed v3 final adapter changed')
            print('V3 final fit verified complete')
            return previous
    rows=make_windows(texts,training=True)
    model=v3_model(warm_path).to('cuda:0')
    parallel=nn.DataParallel(model,device_ids=[0,1],output_device=0)
    optimizer=torch.optim.AdamW(v3_parameter_groups(model),weight_decay=.01,foreach=False)
    total_steps=max(1,epochs*math.ceil(len(rows)/CONFIG['batch_size']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(
        optimizer,lambda step:max(.15,.5*(1+math.cos(math.pi*min(1,step/total_steps)))))
    weights=label_weights(texts).to('cuda:0')
    history=[];epoch_start=0
    if latest_path.is_file():
        saved=torch.load(latest_path,map_location='cpu',weights_only=False)
        if (saved.get('config_hash')!=CONFIG_HASH or saved.get('epochs')!=epochs or
                saved.get('warm_sha256')!=warm_sha or saved.get('windows')!=len(rows)):
            raise RuntimeError('Final v3 optimizer checkpoint provenance mismatch')
        load_trainable_state(model,saved['trainable_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        torch.set_rng_state(saved['torch_rng_state'])
        torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        random.setstate(saved['python_rng_state'])
        history,epoch_start=saved['history'],saved['epoch']
        print('Resuming v3 final fit after epoch',epoch_start)
    paused=False
    try:
        for epoch in range(epoch_start+1,epochs+1):
            if time.time()-RUN_STARTED>SESSION_GUARD_SECONDS:
                paused=True
                print('Time guard during final fit; Run All again to resume.')
                break
            order=list(range(len(rows)))
            random.Random(SEED+900_000+epoch).shuffle(order)
            model.train();losses=[]
            for begin in range(0,len(order),CONFIG['batch_size']):
                group=[rows[i] for i in order[begin:begin+CONFIG['batch_size']]]
                batch={name:value.to('cuda:0',non_blocking=True)
                       for name,value in collate_windows(group,epoch+1200).items()}
                optimizer.zero_grad(set_to_none=True)
                output=(parallel(batch['input_ids'],batch['attention_mask'],batch['pair_index'])
                        if len(group)>1 else
                        model(batch['input_ids'],batch['attention_mask'],batch['pair_index']))
                loss=v3_loss(output,batch,weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Final epoch {epoch}: nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step();scheduler.step()
                losses.append(float(loss.detach().cpu()))
            history.append({'epoch':epoch,'train_loss_mean':sum(losses)/len(losses),
                            'elapsed_seconds':round(time.time()-RUN_STARTED,1)})
            atomic_json(final_dir/'history.json',history)
            if free_gib()<MIN_FREE_GIB:
                raise RuntimeError('Disk reserve reached before final optimizer save')
            atomic_torch(latest_path,{'config_hash':CONFIG_HASH,'epochs':epochs,
                                      'warm_sha256':warm_sha,'windows':len(rows),
                                      'epoch':epoch,'history':history,
                                      'trainable_state':trainable_state(model),
                                      'optimizer_state':optimizer.state_dict(),
                                      'scheduler_state':scheduler.state_dict(),
                                      'torch_rng_state':torch.get_rng_state(),
                                      'cuda_rng_states':torch.cuda.get_rng_state_all(),
                                      'python_rng_state':random.getstate()})
            print(f'Final v3 epoch {epoch}/{epochs}: loss {history[-1]["train_loss_mean"]:.4f}, '
                  f'free {free_gib():.2f} GiB',flush=True)
        if not paused:
            atomic_torch(adapter_path,{'config_hash':CONFIG_HASH,'epochs':epochs,
                                       'trainable_state':trainable_state(model),'labels':LABELS})
        result={'status':'paused' if paused else 'complete','config_hash':CONFIG_HASH,
                'epochs':epochs,'history':history,
                'adapter_sha256':file_sha(adapter_path) if not paused else None}
        atomic_json(summary_path,result)
        if not paused:latest_path.unlink(missing_ok=True)
        return result
    finally:
        del model,parallel,optimizer,scheduler
        torch.cuda.empty_cache()


def infer_v3_test(final_summary):
    final_dir=ROOT/'final'
    partial=final_dir/'test_candidates.partial.json'
    provenance_path=final_dir/'test_candidates_provenance.json'
    provenance={'config_hash':CONFIG_HASH,
                'adapter_sha256':final_summary['adapter_sha256'],
                'test_sha256':file_sha(SOURCE_DIR/'fold3-text.json')}
    if provenance_path.is_file() and read_json(provenance_path)!=provenance:
        raise RuntimeError('V3 test cache provenance changed')
    atomic_json(provenance_path,provenance)
    existing=read_json(partial) if partial.is_file() else {}
    if not set(existing)<=set(test_texts):
        raise RuntimeError('V3 partial test inference has unknown document IDs')
    warm_path=V2_ROOT/'final'/'selected_adapter.pt'
    model=v3_model(warm_path).to('cuda:0')
    saved=torch.load(final_dir/'selected_adapter.pt',map_location='cpu',weights_only=True)
    if saved.get('config_hash')!=CONFIG_HASH:
        raise RuntimeError('V3 selected adapter configuration changed')
    load_trainable_state(model,saved['trainable_state'])
    parallel=nn.DataParallel(model,device_ids=[0,1],output_device=0)
    try:
        def save_progress(predictions):
            if len(predictions)%5==0:
                atomic_json(partial,predictions)
                print('V3 test inference',len(predictions),'/',len(test_texts),flush=True)
        raw=predict_documents(model,parallel,test_texts,test_texts,
                              existing=existing,on_document=save_progress)
    finally:
        del model,parallel
        torch.cuda.empty_cache()
    if set(raw)!=set(test_texts):
        raise RuntimeError('V3 inference did not finish all test documents')
    atomic_json(partial,raw)
    return raw


FINAL_RESULT=None
V3_TEST_RAW=None
FINAL_GATE=max(MIN_FINAL_CV_F1,V2_BASELINE+.002)
if SELECTION is not None:
    if SELECTION['chosen_oof']<FINAL_GATE:
        print(f'Selected local F1 {SELECTION["chosen_oof"]:.4f} did not clear '
              f'{FINAL_GATE:.4f}; keep the 0.5069 submission.')
    elif time.time()-RUN_STARTED>SESSION_GUARD_SECONDS:
        print('Time guard before full-data fit; Run All again to continue.')
    else:
        uses_v3=(SELECTION['chosen_method'] in ('v3_raw','v3_reranked') or
                 (SELECTION['chosen_method']=='hybrid' and
                  any(source.startswith('v3_') for source in
                      SELECTION['hybrid_label_sources'].values())))
        epochs=selected_final_epochs() if uses_v3 else 0
        if uses_v3 and epochs:
            FINAL_RESULT=train_v3_final(epochs)
            if FINAL_RESULT['status']=='complete':
                V3_TEST_RAW=infer_v3_test(FINAL_RESULT)
        else:
            FINAL_RESULT={'status':'complete','epochs':0,
                          'note':'Selected decoder uses only verified v2 candidate states.'}
            V3_TEST_RAW=read_json(V2_TEST_PATH)
        if V3_TEST_RAW is not None and set(V3_TEST_RAW)!=set(test_texts):
            raise RuntimeError('Final v3 test candidate IDs are incomplete')


# %% 10 - verify the old score path and package the chosen new predictions
PACKAGE=None
if V3_TEST_RAW is not None and SELECTION is not None:
    v1_test_path=V1_ROOT/'final'/'test_candidates.partial.json'
    v2_test=read_json(V2_TEST_PATH)
    v1_test=read_json(v1_test_path)
    if set(v1_test)!=set(test_texts) or set(v2_test)!=set(test_texts):
        raise RuntimeError('The older test candidate caches are incomplete')

    # Reproduce the exact existing 0.5069 submission before combining sources.
    baseline_meta_path=V2_ROOT/'final'/'document_reranker.pkl'
    if not baseline_meta_path.is_file():
        raise RuntimeError('The saved v2 document reranker is missing')
    with baseline_meta_path.open('rb') as stream:
        baseline_meta=pickle.load(stream)
    X_test,_,baseline_rows,baseline_ranges=build_feature_table(
        test_texts,v1_test,v2_test,test_texts,with_gold=False)
    v2_reranked_test=rerank_documents(
        baseline_meta,X_test,baseline_rows,baseline_ranges,test_texts)
    baseline_predictions=decode(v2_reranked_test,V2_COMPARISON['reranker_thresholds'])
    if baseline_predictions!=read_json(V2_ROOT/'final'/'prediction.json'):
        raise RuntimeError('The existing v2 submission did not reproduce; refusing to mix candidates')
    del baseline_meta,X_test,baseline_rows,baseline_ranges
    print('Existing 0.5069 prediction.json reproduced exactly')

    # The second reranker is trained on all public OOF candidates only.
    X_train,y_train,_,train_ranges=build_feature_table(
        texts,V2_OOF,V3_OOF,texts,with_gold=True)
    new_meta=train_reranker(X_train,y_train,train_ranges,sorted(texts))
    meta_path=ROOT/'final'/'document_reranker.pkl'
    meta_path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=meta_path.parent,suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name)
    try:
        with temporary.open('wb') as stream:
            pickle.dump(new_meta,stream,protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary,meta_path)
    finally:
        temporary.unlink(missing_ok=True)
    del X_train,y_train,train_ranges
    X_test,_,new_rows,new_ranges=build_feature_table(
        test_texts,v2_test,V3_TEST_RAW,test_texts,with_gold=False)
    v3_reranked_test=rerank_documents(new_meta,X_test,new_rows,new_ranges,test_texts)
    del new_meta,X_test,new_rows,new_ranges

    method=SELECTION['chosen_method']
    source_maps={'v2_raw':v2_test,'v2_reranked':v2_reranked_test,
                 'v3_raw':V3_TEST_RAW,'v3_reranked':v3_reranked_test}
    if method=='v3_raw':
        predictions=decode(V3_TEST_RAW,SELECTION['v3_thresholds'])
    elif method=='v3_reranked':
        predictions=decode(v3_reranked_test,SELECTION['v3_reranker_thresholds'])
    elif method=='hybrid':
        combined=hybrid_candidates(
            source_maps,SELECTION['hybrid_source_thresholds'],
            SELECTION['hybrid_label_sources'],test_texts)
        predictions=decode(combined,{label:.5 for label in LABELS})
    else:
        raise RuntimeError('An unchanged v2 method cannot create a new submission')
    if set(predictions)!=set(test_texts):
        raise RuntimeError('The new submission lacks an official test document')
    for doc_id,spans in predictions.items():
        for a,b,label in spans:
            if label not in LABEL_ID or not 0<=a<b<=len(test_texts[doc_id]):
                raise RuntimeError(f'Invalid v3 span in test document {doc_id}')
        if any(spans[i][1]>spans[i+1][0] for i in range(len(spans)-1)):
            raise RuntimeError(f'Overlapping v3 predictions in test document {doc_id}')
    output=ROOT/'final'/'prediction.json'
    atomic_json(output,predictions)
    submission=WORKING/'CS781_Phase2_Span_V3_Submission.zip'
    temporary=submission.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        archive.write(output,arcname='prediction.json')
    os.replace(temporary,submission)
    with zipfile.ZipFile(submission) as archive:
        if (archive.testzip() is not None or archive.namelist()!=['prediction.json'] or
                json.loads(archive.read('prediction.json'))!=predictions):
            raise RuntimeError('V3 submission ZIP validation failed')
    PACKAGE={'status':'READY_FOR_MANUAL_UPLOAD','method':method,
             'submission_zip':str(submission),'zip_sha256':file_sha(submission),
             'test_documents':len(predictions),
             'predicted_spans':sum(map(len,predictions.values()))}
    atomic_json(ROOT/'final'/'package.json',PACKAGE)
    print('New verified submission:',submission)

RECOVERY=None
members=[(ROOT/'experiment_config.json','experiment_config.json')]
for fold_index in range(3):
    best=verify_v3_best(ROOT/f'fold{fold_index}',fold_index)
    if best is not None:
        members.append((ROOT/f'fold{fold_index}'/best['adapter_file'],
                        f'fold{fold_index}/selected_adapter.pt'))
for name in ('selected_adapter.pt','document_reranker.pkl'):
    path=ROOT/'final'/name
    if path.is_file():members.append((path,f'final/{name}'))
if len(members)>1:
    recovery=WORKING/'CS781_Phase2_Span_V3_Recovery.zip'
    temporary=recovery.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1) as archive:
        archive.writestr('README.txt',
                         'V3 trainable states only. The Phase-1 base-03 and v2 adapters are '
                         'required for reconstruction. Use the V3 notebook source.\n')
        for path,name in members:archive.write(path,arcname=name)
    os.replace(temporary,recovery)
    with zipfile.ZipFile(recovery) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('V3 recovery ZIP CRC failure')
    RECOVERY={'zip':str(recovery),'sha256':file_sha(recovery),
              'members':[name for _,name in members]}

status=('READY_FOR_MANUAL_UPLOAD' if PACKAGE else
        'PAUSED_RUN_ALL_TO_RESUME' if SELECTION is None or
        (SELECTION['chosen_oof']>=FINAL_GATE and V3_TEST_RAW is None)
        else 'NO_LOCAL_IMPROVEMENT_KEEP_V2')
REPORT={'version':VERSION,'status':status,'config_hash':CONFIG_HASH,
        'elapsed_seconds':round(time.time()-RUN_STARTED,1),
        'disk_free_gib':round(free_gib(),3),
        'v2_official_score_user_reported':0.506919,
        'v2_baseline_oof':V2_BASELINE,'final_fit_gate':FINAL_GATE,
        'fold_results':FOLD_RESULTS,'selection':SELECTION,
        'final_fit':FINAL_RESULT,'package':PACKAGE,'recovery':RECOVERY,
        'caveats':[
            'The official private F1 for this new submission is unknown.',
            'Best epochs, thresholds, and hybrid label choices were selected using the same OOF labels.',
            'Cross-fitted reranker training uses public labeled documents only.',
            'The 0.506919 baseline was reported by the user; prediction.json was reproduced from saved artifacts.',
            'No outside pretrained weights or private annotations were used.']}
atomic_json(ROOT/'report.json',REPORT)
report_files=[(ROOT/'report.json','report.json'),
              (ROOT/'experiment_config.json','experiment_config.json')]
if (ROOT/'model_selection.json').is_file():
    report_files.append((ROOT/'model_selection.json','model_selection.json'))
for fold_index in range(3):
    folder=ROOT/f'fold{fold_index}'
    for name in ('summary.json','history.json','best.json'):
        path=folder/name
        if path.is_file():report_files.append((path,f'fold{fold_index}/{name}'))
    best=verify_v3_best(folder,fold_index)
    if best is not None:
        report_files.append((folder/best['candidates_file'],
                             f'fold{fold_index}/{best["candidates_file"]}'))
for name in ('summary.json','package.json','test_candidates.partial.json'):
    path=ROOT/'final'/name
    if path.is_file():report_files.append((path,f'final/{name}'))
temporary=REPORT_ZIP.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,
                     compresslevel=6) as archive:
    archive.writestr('README.txt',
                     'Inspect report.json and model_selection.json. '
                     'Submit only when status is READY_FOR_MANUAL_UPLOAD. '
                     'The prior v2 score remains available.\n')
    for path,name in report_files:archive.write(path,arcname=name)
os.replace(temporary,REPORT_ZIP)
with zipfile.ZipFile(REPORT_ZIP) as archive:
    if archive.testzip() is not None:
        raise RuntimeError('V3 report ZIP CRC failure')
print('Report:',REPORT_ZIP,'status:',status)
if PACKAGE:print('Submission:',PACKAGE['submission_zip'])
if RECOVERY:print('Recovery:',RECOVERY['zip'])
print('Keep the old 0.5069 submission until a new official score is known.')
