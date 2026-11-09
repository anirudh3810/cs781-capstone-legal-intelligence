"""Embedded Run-All orchestration for three improved Phase-2 NER submissions.

This cell is appended after the verified Phase-2 archive, data, model, training,
and packaging definitions.  It never mutates the previous Phase-2 workspace.
"""
from contextlib import suppress


IMPROVED_ROOT = Path(WORK_ROOT)
IMPROVED_SLOTS = (1, 3, 5)
CV_FOLDS = 3
CURRENT_STAGE_KEY = 'not_started'
IMPROVEMENT_VERSION = 'phase2-improved-v1'
SAFETY_DEADLINE_SECONDS = 31_200  # leave time before a ten-hour session limit
TRAINING_RESERVE_SECONDS = 2_700  # preserve time for test inference/packaging


def _improved_recovery_name(slot: int) -> str:
    return f'phase2_improved_{CURRENT_STAGE_KEY}_submission{slot:02d}_recovery.zip'


recovery_name = _improved_recovery_name


def _improved_common_module_path() -> Path:
    return IMPROVED_ROOT / 'method_source' / 'phase2_method_common.py'


common_module_path = _improved_common_module_path


def _prepare_improved_sources() -> None:
    common_path = _improved_common_module_path()
    common_path.parent.mkdir(parents=True, exist_ok=True)
    content = IMPROVED_COMMON_SOURCE.encode('utf-8')
    if common_path.exists():
        if common_path.read_bytes() != content:
            raise RuntimeError('Improved shared method source changed after initialization')
    else:
        with tempfile.NamedTemporaryFile(dir=common_path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, common_path)
    for slot in IMPROVED_SLOTS:
        filename = f'phase2_method_{slot:02d}.py'
        source_path = IMPROVED_ROOT / 'method_source' / filename
        expected = IMPROVED_METHOD_SOURCES[slot].encode('utf-8')
        if source_path.exists():
            if source_path.read_bytes() != expected:
                raise RuntimeError(f'{filename} changed after improvement initialization')
        else:
            with tempfile.NamedTemporaryFile(dir=source_path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(expected)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, source_path)
        METHOD_SOURCE_OVERRIDES[slot] = str(source_path)
    print('Improvement method sources:', METHOD_SOURCE_OVERRIDES)


def _folds(data: Mapping[str, Any]) -> list[list[str]]:
    source1 = set(load_json(data['paths']['fold1-text.json']))
    source2 = set(load_json(data['paths']['fold2-text.json']))
    if source1 & source2 or source1 | source2 != set(data['train_text']):
        raise RuntimeError('Fold-source ID audit failed')
    groups = [sorted(source1), sorted(source2)]
    folds = [[] for _ in range(CV_FOLDS)]
    for group_index, group in enumerate(groups):
        random.Random(SEED + 100 + group_index).shuffle(group)
        for position, identifier in enumerate(group):
            folds[position % CV_FOLDS].append(identifier)
    for fold in folds:
        fold.sort()
    if sorted(identifier for fold in folds for identifier in fold) != sorted(data['train_text']):
        raise RuntimeError('Validation folds are incomplete or overlapping')
    manifest = {'version': IMPROVEMENT_VERSION, 'dataset_fingerprint': data['fingerprint'],
                'folds': folds, 'seed': SEED, 'source_stratified': True,
                'official_test_excluded': True}
    path = IMPROVED_ROOT / 'cv_folds.json'
    if path.exists() and load_json(path) != manifest:
        raise RuntimeError('Existing improvement folds differ; refusing unsafe resume')
    atomic_json(path, manifest)
    return folds


def _window_bounds(length: int, size: int, stride: int) -> list[tuple[int, int]]:
    if length <= size:
        return [(0, length)]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return [(start, min(start + size, length)) for start in starts]


def _windowed_training_data(data, train_ids, tokenizer, config, stage_key):
    """Create stable, exhaustive document windows; validation remains untouched."""
    train_text = dict(data['train_text'])
    train_gold = dict(data['train_gold'])
    window_ids = []
    rare = {'APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL', 'JUDGE', 'WIT'}
    rare_ids = []
    uncovered = []
    size, stride = int(config['max_tokens']), int(config['window_stride'])
    # Leave a few BPE positions for retokenization at substring boundaries.
    core_size = size - 8
    for identifier in sorted(train_ids):
        text = data['train_text'][identifier]
        encoding = tokenizer.backend.encode(text, add_special_tokens=False)
        offsets = [(int(a), int(b)) for a, b in encoding.offsets]
        if not offsets:
            continue
        bounds = _window_bounds(len(offsets), core_size, min(stride, core_size))
        gold = data['train_gold'][identifier]
        covered = [False] * len(gold)
        for window_index, (lo, hi) in enumerate(bounds):
            char_lo, char_hi = offsets[lo][0], offsets[hi - 1][1]
            if char_hi <= char_lo:
                continue
            spans = []
            for gold_index, (a, b, label) in enumerate(gold):
                if char_lo <= a and b <= char_hi:
                    spans.append((a - char_lo, b - char_lo, label))
                    covered[gold_index] = True
            window_id = f'{identifier}__{window_index:04d}'
            train_text[window_id] = text[char_lo:char_hi]
            train_gold[window_id] = spans
            window_ids.append(window_id)
            if any(label in rare for _, _, label in spans):
                rare_ids.append(window_id)
        # Add an entity-centred window where a boundary fell through the
        # regular grid.  It keeps the entity complete without using test data.
        for gold_index, okay in enumerate(covered):
            if okay:
                continue
            a, b, label = gold[gold_index]
            touching = [i for i, (x, y) in enumerate(offsets) if y > a and x < b]
            if not touching or touching[-1] - touching[0] + 1 > core_size:
                uncovered.append((identifier, gold[gold_index]))
                continue
            lo = max(0, min(touching[0], len(offsets) - core_size))
            hi = min(len(offsets), lo + core_size)
            char_lo, char_hi = offsets[lo][0], offsets[hi - 1][1]
            if not char_lo <= a or not b <= char_hi:
                uncovered.append((identifier, gold[gold_index]))
                continue
            window_id = f'{identifier}__entity{gold_index:04d}'
            train_text[window_id] = text[char_lo:char_hi]
            train_gold[window_id] = [(x - char_lo, y - char_lo, name)
                                     for x, y, name in gold if char_lo <= x and y <= char_hi]
            window_ids.append(window_id)
            if label in rare:
                rare_ids.append(window_id)
    if uncovered:
        # An entity longer than a model window cannot be learned through this
        # token-tagging setup.  Surface it rather than silently claiming coverage.
        atomic_json(IMPROVED_ROOT / f'{stage_key}_uncovered_spans.json', uncovered)
        print(f'{stage_key}: {len(uncovered)} gold spans not wholly contained in a window')
    # Cap the extra rare-label visits so that ordinary text remains represented.
    extra_count = min(len(rare_ids), max(1, len(window_ids) // 6))
    extra = random.Random(SEED + len(window_ids)).sample(rare_ids, extra_count) if extra_count else []
    for index, source_id in enumerate(extra):
        duplicate_id = f'{source_id}__rare{index:05d}'
        train_text[duplicate_id] = train_text[source_id]
        train_gold[duplicate_id] = train_gold[source_id]
        window_ids.append(duplicate_id)
    if not window_ids:
        raise RuntimeError('No legal-NER training windows were built')
    fingerprint = hashlib.sha256(stable_json({
        'version': IMPROVEMENT_VERSION, 'original': data['fingerprint'],
        'stage': stage_key, 'train_ids': sorted(train_ids), 'window_ids': window_ids,
        'size': size, 'stride': stride,
    }).encode()).hexdigest()
    staged = dict(data)
    staged['train_text'] = train_text
    staged['train_gold'] = train_gold
    staged['fingerprint'] = fingerprint
    return staged, window_ids


def _config_for(slot: int, fold_index: int | None, num_windows: int,
                final_epochs: int | None = None) -> dict[str, Any]:
    config = dict(TRAINING_CONFIGS[slot])
    config.update(max_tokens=384, window_stride=288, batch_size=16,
                  gradient_accumulation_steps=1,
                  outside_class_weight=0.6, inference_window_batch_size=8,
                  confidence_threshold=0.45,
                  seed=SEED + slot * 100 + (fold_index if fold_index is not None else 10))
    if slot == 5:
        config['learning_rate'] = 1e-4
    epochs = final_epochs if final_epochs is not None else {1: 16, 3: 20, 5: 10}[slot]
    steps_per_epoch = math.ceil(num_windows / config['batch_size'])
    config.update(max_steps=max(1, epochs * steps_per_epoch),
                  validation_interval=steps_per_epoch,
                  checkpoint_interval=max(25, steps_per_epoch),
                  recovery_interval=max(50, 2 * steps_per_epoch),
                  improvement_version=IMPROVEMENT_VERSION,
                  exhaustive_window_epochs=epochs,
                  training_window_count=num_windows)
    return config


def _stage_root(stage_key: str) -> Path:
    return IMPROVED_ROOT / 'stages' / stage_key


def _activate_stage(stage_key: str) -> None:
    global WORK_ROOT, CURRENT_STAGE_KEY
    CURRENT_STAGE_KEY = stage_key
    WORK_ROOT = _stage_root(stage_key)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    RESTORED_RECOVERY_SLOTS.clear()


def _run_stage(spec, data, train_ids, val_ids, stage_key, fold_index,
               deadline, final_epochs=None):
    slot = spec['slot']
    _activate_stage(stage_key)
    base, tokenizer, _meta = load_phase1_base(spec['base_id'], 'cpu')
    del base
    provisional = _config_for(slot, fold_index, 1, final_epochs)
    staged, windows = _windowed_training_data(data, train_ids, tokenizer, provisional, stage_key)
    config = _config_for(slot, fold_index, len(windows), final_epochs)
    TRAINING_CONFIGS[slot] = config
    if fold_index is not None:
        sealed_path = WORK_ROOT / 'cv_sealed.json'
        if sealed_path.exists():
            sealed = load_json(sealed_path)
            selected_path = experiment_directory(slot) / 'selected_model.pt'
            if (sealed.get('stage') != stage_key or sealed.get('config') != config or
                    not selected_path.is_file() or
                    sealed.get('selected_sha256') != sha256_file(selected_path)):
                raise RuntimeError(f'{stage_key}: sealed CV checkpoint failed integrity check')
            return staged, config, {'status': 'complete', 'step': sealed['selected_step'],
                                    'sealed_cv': True}
    print(f'{stage_key}: {len(windows)} training windows, '
          f'{config["max_steps"]} maximum steps, {len(val_ids)} validation documents')
    result = train_one_experiment(spec, staged, windows, val_ids, deadline)
    if result['status'] != 'complete':
        return staged, config, result
    selected_path = experiment_directory(slot) / 'selected_model.pt'
    if not selected_path.is_file():
        raise RuntimeError(f'{stage_key}: selected checkpoint missing after training')
    return staged, config, result


def _collect_cv_candidates(spec, staged, config, val_ids, stage_key):
    path = IMPROVED_ROOT / 'cv_candidates' / f'{stage_key}.json'
    directory = experiment_directory(spec['slot'])
    method, method_sha = load_student_method(spec, directory / spec['module_filename'])
    selected_path = directory / 'selected_model.pt'
    selected_sha = sha256_file(selected_path)
    expected = {'stage': stage_key, 'selected_sha256': selected_sha,
                'ids': sorted(val_ids), 'version': IMPROVEMENT_VERSION}
    if path.exists():
        record = load_json(path)
        if record.get('provenance') != expected:
            raise RuntimeError(f'{stage_key}: stale candidate cache')
        candidates = record['candidates']
        if set(candidates) == set(val_ids) and not record.get('incomplete'):
            return candidates
    else:
        candidates = {}
    model, tokenizer, _, _ = load_selected_task(
        spec, staged, config, method, method_sha, 'cuda:0')
    candidate_config = dict(config, return_candidate_scores=True)
    try:
        for identifier in sorted(val_ids):
            if identifier in candidates:
                continue
            candidates[identifier] = method.predict_spans(
                model, staged['train_text'][identifier], tokenizer,
                staged['labels'], candidate_config)
            if len(candidates) % 5 == 0:
                atomic_json(path, {'provenance': expected, 'candidates': candidates,
                                   'incomplete': True})
    finally:
        del model
        torch.cuda.empty_cache()
    atomic_json(path, {'provenance': expected, 'candidates': candidates})
    return candidates


def _seal_cv_stage(spec, config, stage_key):
    """Keep the verified recovery ZIP, selected model, and history; free live optimizer state."""
    directory = experiment_directory(spec['slot'])
    selected_path = directory / 'selected_model.pt'
    candidate_path = IMPROVED_ROOT / 'cv_candidates' / f'{stage_key}.json'
    recovery_path = directory / recovery_name(spec['slot'])
    candidate_record = load_json(candidate_path)
    if candidate_record.get('incomplete') or not recovery_path.is_file():
        raise RuntimeError(f'{stage_key}: refusing to seal incomplete CV artifacts')
    selected = torch.load(selected_path, map_location='cpu', weights_only=True)
    seal = {'stage': stage_key, 'config': config,
            'selected_sha256': sha256_file(selected_path),
            'selected_step': int(selected['step']),
            'candidate_sha256': sha256_file(candidate_path),
            'recovery_sha256': sha256_file(recovery_path)}
    seal_path = WORK_ROOT / 'cv_sealed.json'
    if seal_path.exists() and load_json(seal_path) != seal:
        raise RuntimeError(f'{stage_key}: existing CV seal differs')
    atomic_json(seal_path, seal)
    for filename in ('checkpoint.latest.pt', 'checkpoint.latest.previous_known_good.pt'):
        target = directory / filename
        if target.is_file():
            target.unlink()


def _decode_candidates(candidate_map, thresholds, default=0.45):
    predictions = {}
    for identifier, rows in candidate_map.items():
        accepted = []
        for a, b, label, score in sorted(rows, key=lambda row: -row[3]):
            if score < thresholds.get(label, default):
                continue
            if any(a < other_b and other_a < b for other_a, other_b, _ in accepted):
                continue
            accepted.append((int(a), int(b), label))
        predictions[identifier] = [list(item) for item in sorted(accepted)]
    return predictions


def _calibrate(slot, data, fold_candidates, folds):
    labels = data['labels']
    all_candidates = {identifier: rows for group in fold_candidates for identifier, rows in group.items()}
    if set(all_candidates) != set(data['train_text']):
        raise RuntimeError(f'Slot {slot}: out-of-fold candidate IDs are incomplete')
    gold = data['train_gold']
    grid = (0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80)
    thresholds = {label: 0.45 for label in labels}

    def score(candidate_subset, thresholds_to_use):
        predictions = _decode_candidates(candidate_subset, thresholds_to_use)
        expected = {identifier: gold[identifier] for identifier in candidate_subset}
        return strict_macro_prf(predictions, expected, labels)['strict_macro_f1_local']

    baseline = score(all_candidates, thresholds)
    for label in labels:
        best_threshold, best_score = thresholds[label], score(all_candidates, thresholds)
        for trial in grid:
            proposed = dict(thresholds, **{label: trial})
            measured = score(all_candidates, proposed)
            if measured > best_score + 1e-9:
                best_threshold, best_score = trial, measured
        thresholds[label] = best_threshold
    tuned = score(all_candidates, thresholds)
    fold_scores = []
    for candidate_subset in fold_candidates:
        fold_scores.append({'baseline': score(candidate_subset, {label: 0.45 for label in labels}),
                            'tuned': score(candidate_subset, thresholds)})
    # Avoid a fragile calibration that only helps the pooled score but hurts
    # two of the three held-out document groups.
    hurts = sum(item['tuned'] + 1e-9 < item['baseline'] for item in fold_scores)
    if tuned <= baseline or hurts >= 2:
        thresholds = {label: 0.45 for label in labels}
        tuned = baseline
    report = {'slot': slot, 'candidate_documents': len(all_candidates),
              'baseline_oof_macro_f1': baseline, 'calibrated_oof_macro_f1': tuned,
              'fold_scores': fold_scores, 'confidence_threshold_by_label': thresholds,
              'note': 'Thresholds optimized on OOF predictions; reported tuned score is selection-biased.'}
    atomic_json(IMPROVED_ROOT / f'submission_{slot:02d}_cv_calibration.json', report)
    return report


def _final_epoch_choice(slot, fold_steps, fold_window_counts) -> int:
    epochs = [step / max(1, math.ceil(count / 16))
              for step, count in zip(fold_steps, fold_window_counts)]
    choice = int(round(float(np.median(epochs))))
    return max(3, min(choice, {1: 16, 3: 20, 5: 10}[slot]))


def _promote_final_checkpoint(spec, config, cv_report):
    """Use the CV-chosen training duration, not in-sample final validation."""
    directory = experiment_directory(spec['slot'])
    latest = torch.load(directory / 'checkpoint.latest.pt', map_location='cpu', weights_only=True)
    selected_path = directory / 'selected_model.pt'
    selected = torch.load(selected_path, map_location='cpu', weights_only=True)
    if latest['step'] != config['max_steps'] or selected['training_config'] != config:
        raise RuntimeError('Final checkpoint/config is incomplete or inconsistent')
    result_path = directory / 'training_result.json'
    result = load_json(result_path)
    result['best_strict_macro_f1_local'] = None
    result['selection_note'] = 'Final fit used all public labeled documents; CV selected the duration.'
    result['cv_baseline_oof_macro_f1'] = cv_report['baseline_oof_macro_f1']
    result['cv_calibrated_oof_macro_f1'] = cv_report['calibrated_oof_macro_f1']
    atomic_json(result_path, result)
    if selected.get('selection_note') == 'Full-data final weights at CV-selected epoch; no honest final holdout score':
        if selected['step'] != latest['step']:
            raise RuntimeError('Previously promoted selected checkpoint has wrong step')
        return
    selected['model_state'] = latest['model_state']
    selected['step'] = latest['step']
    selected['strict_macro_f1_local'] = None
    selected['selection_note'] = 'Full-data final weights at CV-selected epoch; no honest final holdout score'
    selected['cv_baseline_oof_macro_f1'] = cv_report['baseline_oof_macro_f1']
    selected['cv_calibrated_oof_macro_f1'] = cv_report['calibrated_oof_macro_f1']
    atomic_torch_save(selected_path, selected)
    export_recovery(spec['slot'])


def _predict_final(spec, staged, train_config, cv_report, deadline):
    slot = spec['slot']
    directory = experiment_directory(slot)
    method, method_sha = load_student_method(spec, directory / spec['module_filename'])
    selected_path = directory / 'selected_model.pt'
    decoding_config = dict(train_config)
    decoding_config['confidence_threshold_by_label'] = cv_report['confidence_threshold_by_label']
    provenance = {'slot': slot, 'selected_sha256': sha256_file(selected_path),
                  'method_sha256': method_sha,
                  'test_file_sha256': sha256_file(staged['paths']['fold3-text.json']),
                  'decoding_thresholds': decoding_config['confidence_threshold_by_label']}
    partial_path = directory / 'predictions.improved.partial.json'
    meta_path = directory / 'predictions.improved.provenance.json'
    if meta_path.exists() and load_json(meta_path) != provenance:
        raise RuntimeError('Partial inference provenance changed; refusing stale predictions')
    if partial_path.exists():
        if not meta_path.exists():
            raise RuntimeError('Partial predictions lack provenance')
        predictions = load_json(partial_path)
    else:
        predictions = {}
        atomic_json(meta_path, provenance)
    if set(predictions) - set(staged['test_text']):
        raise RuntimeError('Partial predictions contain unexpected official IDs')
    for identifier, spans in predictions.items():
        validate_prediction_map({identifier: spans},
                                {identifier: staged['test_text'][identifier]}, staged['labels'])
    if len(predictions) != len(staged['test_text']):
        model, tokenizer, _, _ = load_selected_task(
            spec, staged, train_config, method, method_sha, 'cuda:0')
        try:
            failures = {}
            for attempt in range(3):
                pending = [identifier for identifier in staged['test_text'] if identifier not in predictions]
                if not pending:
                    break
                for identifier in pending:
                    if time.monotonic() >= deadline:
                        export_recovery(slot)
                        return None
                    try:
                        text = staged['test_text'][identifier]
                        spans = [] if not text else method.predict_spans(
                            model, text, tokenizer, staged['labels'], decoding_config)
                        validate_prediction_map({identifier: spans},
                                                {identifier: text}, staged['labels'])
                        predictions[identifier] = spans
                        atomic_json(partial_path, predictions)
                        failures.pop(identifier, None)
                        append_jsonl(directory / 'inference_attempts.jsonl',
                                     {'id': identifier, 'attempt': attempt + 1, 'status': 'success'})
                    except Exception as exc:
                        failures[identifier] = f'{type(exc).__name__}: {exc}'
                        append_jsonl(directory / 'inference_attempts.jsonl',
                                     {'id': identifier, 'attempt': attempt + 1,
                                      'status': 'failure', 'error': failures[identifier]})
                        torch.cuda.empty_cache()
                atomic_json(directory / 'inference_failures.json', failures)
            if len(predictions) != len(staged['test_text']):
                export_recovery(slot)
                raise RuntimeError(f'Slot {slot}: {len(staged["test_text"])-len(predictions)} '
                                   'official documents remain unpredicted; partial state is saved')
        finally:
            del model
            torch.cuda.empty_cache()
    package = make_submission_zip(slot, predictions, staged, provenance)
    top_level = IMPROVED_ROOT / f'submission_{slot:02d}_improved.zip'
    if top_level.exists():
        with zipfile.ZipFile(top_level) as existing:
            if existing.namelist() != ['prediction.json'] or existing.testzip() is not None:
                raise RuntimeError(f'{top_level} is not a valid previous submission')
            old_prediction_sha = hashlib.sha256(existing.read('prediction.json')).hexdigest()
        if old_prediction_sha != package['prediction_sha256']:
            raise RuntimeError(f'{top_level} already exists with different predictions')
    if not top_level.exists():
        with tempfile.NamedTemporaryFile(dir=IMPROVED_ROOT, suffix='.zip', delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(package['zip_path'], temporary)
            os.replace(temporary, top_level)
        finally:
            temporary.unlink(missing_ok=True)
    upload_copy = IMPROVED_ROOT / 'final_submissions' / f'submission_{slot:02d}' / 'submission.zip'
    upload_copy.parent.mkdir(parents=True, exist_ok=True)
    if upload_copy.exists() and sha256_file(upload_copy) != sha256_file(top_level):
        raise RuntimeError(f'{upload_copy} already exists with different bytes')
    if not upload_copy.exists():
        shutil.copyfile(top_level, upload_copy)
    package['download_zip'] = str(top_level)
    package['upload_zip'] = str(upload_copy)
    export_recovery(slot)
    return package


def run_improvement_all():
    global WORK_ROOT
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('This Run-All plan requires two visible Kaggle GPUs')
    IMPROVED_ROOT.mkdir(parents=True, exist_ok=True)
    _prepare_improved_sources()
    data = load_official_phase2()
    folds = _folds(data)
    spec_by_slot = {spec['slot']: spec for spec in EXPERIMENTS}
    started = time.monotonic()
    total_deadline = started + SAFETY_DEADLINE_SECONDS
    training_deadline = total_deadline - TRAINING_RESERVE_SECONDS
    status_path = IMPROVED_ROOT / 'improvement_status.json'
    results = {}
    for slot in IMPROVED_SLOTS:
        spec = spec_by_slot[slot]
        fold_candidates, selected_steps, fold_window_counts = [], [], []
        for fold_index, val_ids in enumerate(folds):
            if time.monotonic() >= training_deadline:
                atomic_json(status_path, {'status': 'TIME_GUARD', 'next': f'cv_{slot:02d}_{fold_index}',
                                          'completed': results})
                return results
            train_ids = sorted(set(data['train_text']) - set(val_ids))
            stage_key = f'cv{fold_index}_slot{slot:02d}'
            staged, config, result = _run_stage(
                spec, data, train_ids, val_ids, stage_key, fold_index, training_deadline)
            if result['status'] != 'complete':
                atomic_json(status_path, {'status': 'TIME_GUARD', 'next': stage_key,
                                          'completed': results})
                return results
            selected = torch.load(experiment_directory(slot) / 'selected_model.pt',
                                  map_location='cpu', weights_only=True)
            selected_steps.append(int(selected['step']))
            fold_window_counts.append(int(config['training_window_count']))
            fold_candidates.append(_collect_cv_candidates(spec, staged, config, val_ids, stage_key))
            _seal_cv_stage(spec, config, stage_key)
            atomic_json(status_path, {'status': 'CV_IN_PROGRESS', 'last_completed': stage_key,
                                      'completed': results})
        cv_report = _calibrate(slot, data, fold_candidates, folds)
        final_epochs = _final_epoch_choice(slot, selected_steps, fold_window_counts)
        stage_key = f'final_slot{slot:02d}'
        if time.monotonic() >= training_deadline:
            atomic_json(status_path, {'status': 'TIME_GUARD', 'next': stage_key,
                                      'completed': results})
            return results
        # One training document is used only for the runner's smoke/periodic
        # diagnostic; every labeled document is included in final training.
        all_ids = sorted(data['train_text'])
        staged, config, result = _run_stage(
            spec, data, all_ids, all_ids[:1], stage_key, None,
            training_deadline, final_epochs=final_epochs)
        if result['status'] != 'complete':
            atomic_json(status_path, {'status': 'TIME_GUARD', 'next': stage_key,
                                      'completed': results})
            return results
        _promote_final_checkpoint(spec, config, cv_report)
        package = _predict_final(spec, staged, config, cv_report, total_deadline)
        if package is None:
            atomic_json(status_path, {'status': 'TIME_GUARD', 'next': f'inference_{slot:02d}',
                                      'completed': results})
            return results
        results[str(slot)] = {'cv': cv_report, 'final_epochs': final_epochs, 'package': package}
        atomic_json(status_path, {'status': 'IN_PROGRESS', 'completed': results})
        print(f'Submission {slot:02d} improved package: {package["download_zip"]}')
    if set(results) != {'1', '3', '5'}:
        raise RuntimeError('Improved completion manifest cannot omit a planned slot')
    hashes = [results[str(slot)]['package']['prediction_sha256'] for slot in IMPROVED_SLOTS]
    if len(set(hashes)) != 3:
        raise RuntimeError('Improved submissions have duplicate prediction contents')
    atomic_json(status_path, {'status': 'READY_FOR_MANUAL_CODABENCH_UPLOAD',
                              'completed': results,
                              'note': 'Official score cannot be known before Codabench evaluation.'})
    return results


try:
    IMPROVEMENT_RESULTS = run_improvement_all()
except Exception as improvement_error:
    atomic_json(IMPROVED_ROOT / 'improvement_status.json',
                {'status': 'ERROR', 'error': f'{type(improvement_error).__name__}: {improvement_error}',
                 'active_stage': CURRENT_STAGE_KEY,
                 'note': 'Earlier checkpoints and recovery ZIPs are preserved; inspect traceback.'})
    raise
print('Phase-2 improvement status:', load_json(IMPROVED_ROOT / 'improvement_status.json')['status'])
