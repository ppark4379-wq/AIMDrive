"""Explicit per-run path warmup/joint policies; switching starts a new optimizer."""
PATH_PREFIXES = ('language_model.path_tokenizer.', 'language_model.navigation_cross_attention.')
MLP_PREFIXES = ('language_model.navigation_mlp.',)


def instruction_module_prefixes(model):
    return MLP_PREFIXES if getattr(model.config, 'path_fusion', 'cross_attention') == 'mlp' else PATH_PREFIXES


def apply_instruction_stage(model, stage='joint'):
    if stage not in ('path_warmup', 'joint'):
        raise ValueError('Instruction stage must be path_warmup or joint')
    if not getattr(model.config, 'path_navigation', False):
        if stage != 'joint':
            raise ValueError('path_warmup requires model.path_navigation=true')
        return
    # Remember the ordinary parameter policy, excluding unused heads/frozen VE.
    # Checkpoints retain these weights even when temporarily frozen for warmup.
    if not hasattr(model, '_stage_base_trainable_names'):
        model._stage_base_trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    prefixes = instruction_module_prefixes(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in model._stage_base_trainable_names and
                                 (stage == 'joint' or name.startswith(prefixes)))
    model.instruction_stage = stage
