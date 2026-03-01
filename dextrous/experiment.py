import torch.cuda

import ezpyzy as ez
# --- compat: published ezpyzy may not expose ez.split ---
if not hasattr(ez, "split"):
    def _ez_split(s, *seps):
        if s is None:
            return [None]
        out = [str(s)]
        for sep in seps:
            new = []
            for part in out:
                new.extend(part.split(sep))
            out = new
        return [x.strip() for x in out if x is not None and str(x).strip() != ""]
    ez.split = _ez_split
# --- end compat ---
import dataclasses as dc
import traceback as tb
import resource as res
import torch as pt
import textwrap as tw
import time
import datetime as dt
import sys
import pathlib as pl
from dataclasses import dataclass as settings; vars().update(settings=ez.settings)
import random as rng

import dextrous.dst_data as dst
import dextrous.preprocessing as dpp
import dextrous.metrics as metrics
import dextrous.utils as utils

from dextrous.tracker import LlamaTracker, LlamaTrackerHyperparameters, T5TrackerHyperparameters


def display_row(table):
    return "\n".join(
        f"{k.replace('_', ' ')}: {v}"
        for k, v in table().dict().items()
    )


@dc.dataclass
class Result(ez.Table):
    joint_goal_accuracy: ez.ColFloat = None
    slot_accuracy: ez.ColFloat = None
    slot_update_accuracy: ez.ColFloat = None
    perplexity: ez.ColFloat = None
    train_perplexity: ez.ColFloat = None

@dc.dataclass
class Performance(ez.Table):
    time_to_preprocess_m: ez.ColFloat = None
    time_to_train_h: ez.ColFloat = None
    time_to_calculate_perplexity_h: ez.ColFloat = None
    time_to_generate_h: ez.ColFloat = None
    ram_usage_before_modelling_gb: ez.ColFloat = None
    ram_usage_after_modelling_gb: ez.ColFloat = None
    gpu_usage_before_modelling_gb: ez.ColFloat = None
    gpu_usage_after_training_gb: ez.ColFloat = None
    gpu_usage_after_perplexity_gb: ez.ColFloat = None
    gpu_usage_after_generation_gb: ez.ColFloat = None
    total_experiment_time_h: ez.ColFloat = None
    crash_stage: ez.ColStr = None
    time_of_crash: ez.ColStr = None
    ram_usage_of_crash: ez.ColFloat = None
    gpu_usage_of_crash: ez.ColFloat = None

@dc.dataclass
class Example(ez.Table):
    prompt: ez.ColStr = None
    value: ez.ColStr = None
    prediction: ez.ColStr = None

    def display(self):
        sep = '='*80
        return sep + sep.join(
            f'\n{prompt}  {v} (a) / {p} (p)\n' for prompt, v, p in self().items()
        ) + sep

def get_examples(data: dst.Data, n=30):
    srng = rng.Random()
    preds = data.predictions.slot_value_id << data.slot_values.slot_value_id
    pos_examples = preds[preds.prompt, preds.value, preds.prediction][preds.value != None]
    neg_examples = preds[preds.prompt, preds.value, preds.prediction][preds.value == None]
    pos_samples = srng.sample(list(range(len(pos_examples))), min(n, len(pos_examples)))
    neg_samples = srng.sample(list(range(len(neg_examples))), min(n, len(neg_examples)))
    examples = Example.of(pos_examples[pos_samples], neg_examples[neg_samples])
    return examples


@settings
class ExperimentHyperparameters(ez.Settings):
    experiment: ez.ColStr = None
    approach: ez.ColStr = None
    train_data: ez.ColStr = ez.Def("data/dsg5k/train")
    eval_data: ez.ColStr = ez.Def("data/mwoz2.4/valid")
    test_domains: ez.ColObj|list[str] = None
    eval_all_slots_per_domain: ez.ColBool = ez.Def(True)
    eval_all_slots_all_domains: ez.ColBool = None
    eval_exclude_speakers: ez.ColStr = ez.Def('bot')
    train_downsample: ez.ColInt = None
    eval_downsample: ez.ColInt = None
    eval_dialogue_downsample: ez.ColInt = None
    do_eval_after_all_training: ez.ColBool = ez.Def(True)
    calculate_eval_perplexity: ez.ColBool = ez.Def(True)
    calculate_eval_gen_metrics: ez.ColBool = ez.Def(True)
    use_train_as_eval: ez.ColBool = ez.Def(False)
    yield_every_x_epochs: ez.ColFloat = ez.Def(1.0)
    rng_seed: ez.ColInt = ez.Def(0)
    debugging: ez.ColBool = ez.Def(False)
    notifications: ez.ColBool = ez.Def(False)
    groupcode: ez.ColStr = None


@settings
class LlamaTrackerExperiment(LlamaTrackerHyperparameters, ExperimentHyperparameters): pass
@settings
class LlamaTrackerExperimentTable(ez.Table, LlamaTrackerExperiment): pass

@settings
class T5TrackerExperiment(T5TrackerHyperparameters, ExperimentHyperparameters): pass
@settings
class T5TrackerExperimentTable(ez.Table, T5TrackerExperiment): pass


@settings
class ExperimentRun(ExperimentHyperparameters):
    def __post_init__(self):
        if isinstance(self, LlamaTrackerExperiment):
            self.ModelHyperparameters = LlamaTrackerHyperparameters
            self.TrackerTable = LlamaTrackerExperimentTable
        else:
            self.ModelHyperparameters = T5TrackerHyperparameters
            self.TrackerTable = T5TrackerExperimentTable
        self.experiment_path = pl.Path(f'ex/{self.approach}/{self.experiment}')
        if 'model_path' in self.settings:
            expt_dict = self.TrackerTable.of(pl.Path(self.model_path).parent / 'experiment.csv')
            expt_dict = expt_dict().dict()
            expt_dict['experiment'] = self.experiment
        else:
            expt_dict = {}
        expt_dict.update({
            k: v for k, v in vars(self).items()
            if k in {f.name for f in dc.fields(self.TrackerTable)}
            and k in self.settings
        })
        self.experiment_hyperparams = self.TrackerTable(**expt_dict) # noqa
        self.experiment_hyperparams().save(self.experiment_path / 'experiment.csv')
        vars(self).update(self.experiment_hyperparams().dict())
        rng.seed(self.rng_seed)
        self.start_time = time.perf_counter()
        self.performance_stats = Performance()
        self.stage: str = 'start'
        if self.debugging:
            self.run()
        else:
            try:
                self.run()
            except Exception as e:
                self.crash()
        print('Done')
    def advance(self, stage):
        self.stage = stage # noqa
        return ez.Timer(stage)
    def run(self):
        print(f'Starting {self.experiment}')
        performance_stats = self.performance_stats
        with self.advance('Loading data'):
            train_data = dst.Data(self.train_data)
            eval_data = dst.Data(self.eval_data)
        if self.test_domains:
            with self.advance('Dropping domains for training'):
                dpp.drop_domains(train_data, self.test_domains, include_specified=False)
            with self.advance('Dropping domains for evaluation'):
                dpp.drop_domains(eval_data, self.test_domains, include_specified=True)
        if self.eval_all_slots_all_domains:
            with self.advance('Adding negative slots cross-domain'):
                dpp.add_neg_slot_targets(eval_data, per_domain=False)
        elif self.eval_all_slots_per_domain:
            with self.advance('Adding negative slots within domain'):
                dpp.add_neg_slot_targets(eval_data, per_domain=True)
        if train_data.predictions is None:
            train_data.predictions = dst.Prediction.of(dict(
                slot_value_id=train_data.slot_values.slot_value_id,
                slot_id=train_data.slot_values.slot_id,
            ), fill=None)
        if eval_data.predictions is None:
            eval_data.predictions = dst.Prediction.of(dict(
                slot_value_id=eval_data.slot_values.slot_value_id,
                slot_id=eval_data.slot_values.slot_id,
            ), fill=None)
        if self.train_downsample:
            dpp.downsample_examples(train_data, self.train_downsample)
        if self.use_train_as_eval:
            assert self.train_data == self.eval_data
            eval_data = train_data.copy()  # MODSF
        if self.eval_exclude_speakers:
            dpp.exclude_speakers(eval_data, {'bot'})
        if self.eval_downsample:
            dpp.downsample_examples(eval_data, self.eval_downsample)
        elif self.eval_dialogue_downsample:
            dpp.downsample_dialogues(eval_data, self.eval_dialogue_downsample)
        approach_args = {
            k: v for k, v in vars(self).items()
            if k in {f.name for f in dc.fields(self.ModelHyperparameters)}
            and k in self.settings
        }
        tracker = LlamaTracker(self.ModelHyperparameters(**approach_args)) # noqa
        training_losses = None
        train_loss_graph = None
        def do_the_evaluation(experiment_path, iteration, train_ppl=None):
            iteration_path = experiment_path / str(iteration)
            if self.calculate_eval_perplexity:
                pt.cuda.reset_peak_memory_stats()
                with self.advance('Calculating perplexity') as ppl_timer:
                    perplexity = tracker.perplexity(eval_data)
                if not performance_stats.time_to_calculate_perplexity_h():
                    performance_stats.time_to_calculate_perplexity_h(max(
                        performance_stats.time_to_calculate_perplexity_h() or 0, ppl_timer.delta / 3600))
                if not performance_stats.gpu_usage_after_perplexity_gb():
                    performance_stats.gpu_usage_after_perplexity_gb(max(
                        performance_stats.gpu_usage_after_perplexity_gb() or 0,
                        pt.cuda.max_memory_allocated() / 1e9))
                pt.cuda.reset_peak_memory_stats()
            else:
                perplexity = None
            if self.calculate_eval_gen_metrics:
                pt.cuda.reset_peak_memory_stats()
                with self.advance('Generating predictions') as gen_timer:
                    tracker.predict(eval_data)
                if not performance_stats.gpu_usage_after_generation_gb():
                    performance_stats.gpu_usage_after_generation_gb(max(
                        performance_stats.gpu_usage_after_generation_gb() or 0,
                        pt.cuda.max_memory_allocated() / 1e9))
                if not performance_stats.time_to_generate_h():
                    performance_stats.time_to_generate_h(max(
                        performance_stats.time_to_generate_h() or 0, gen_timer.delta / 3600))
                pt.cuda.reset_peak_memory_stats()
                predictions_with_turn_info = eval_data.predictions.slot_value_id << eval_data.slot_values.slot_value_id
                predictions_with_turn_info().save(iteration_path / 'predictions.csv')
                with self.advance('Calculating evaluation metrics'):
                    joint_goal_accuracy = metrics.joint_goal_accuracy(
                        eval_data, speakers={'user'}, domains=self.test_domains
                    )
                    slot_accuracy = metrics.slot_accuracy(eval_data)
                    slot_update_accuracy = metrics.slot_update_accuracy(eval_data)
            else:
                joint_goal_accuracy = slot_accuracy = slot_update_accuracy = None
            result = Result(
                joint_goal_accuracy=joint_goal_accuracy,
                slot_accuracy=slot_accuracy,
                slot_update_accuracy=slot_update_accuracy,
                perplexity=perplexity,
                train_perplexity=train_ppl,
            )
            result().save(iteration_path / 'result.csv')
            eval_examples = get_examples(eval_data)
            eval_examples().save(iteration_path / 'examples.csv')
            display = '\n\n'.join([
                display_row(result), display_row(self.experiment_hyperparams),
                display_row(performance_stats), eval_examples.display()
            ])
            print(display)
            if self.notifications:
                ez.email(
                    'jamesfinch293@gmail.com', f"{self.experiment} {iteration}", display, training=train_loss_graph)
            final_time = time.perf_counter()
            performance_stats.total_experiment_time_h(max(
                performance_stats.total_experiment_time_h() or 0, (final_time - self.start_time) / 3600))
            performance_stats.ram_usage_after_modelling_gb(max(
                performance_stats.ram_usage_after_modelling_gb() or 0,
                res.getrusage(res.RUSAGE_SELF).ru_maxrss / 1e6))
            print('\nPerformance stats:')
            print(display_row(performance_stats), '\n')
            performance_stats().save(self.experiment_path / 'performance.csv')
        iteration = 0
        if self.epochs and self.epochs > 0: # noqa
            with self.advance('Preparing training'):
                training = tracker.training(train_data, yield_every_x_epochs=self.yield_every_x_epochs)
            training_exs = get_examples(train_data)
            print(training_exs.display())
            performance_stats.time_to_preprocess_m((time.perf_counter() - self.start_time) / 60)
            performance_stats.ram_usage_before_modelling_gb()
            performance_stats.gpu_usage_before_modelling_gb(pt.cuda.max_memory_allocated() / 1e9)
            pt.cuda.reset_peak_memory_stats()
            train_start_time = time.perf_counter()
            training_losses = []
            for i, ppl in enumerate(training, 1):
                iteration = i
                training_losses.append(ppl)
                performance_stats.time_to_train_h(max(
                    performance_stats.time_to_train_h() or 0, (time.perf_counter()-train_start_time)/3600))
                performance_stats.gpu_usage_after_training_gb(max(
                    performance_stats.gpu_usage_after_training_gb() or 0, pt.cuda.max_memory_allocated()/1e9))
                pt.cuda.reset_peak_memory_stats()
                tracker.save(self.experiment_path / str(iteration))
                if not self.do_eval_after_all_training:
                    do_the_evaluation(self.experiment_path, iteration, ppl)
            train_loss_graph = utils.create_line_graph_image(
                steps=list(range(len(training_losses))), ppl=training_losses)
            print('Training losses')
            print(', '.join(f"{x:.3f}" for x in training_losses))
            if self.do_eval_after_all_training:
                do_the_evaluation(
                    self.experiment_path, iteration,
                    training_losses[-1] if training_losses else None)
        else:
            do_the_evaluation(self.experiment_path, iteration)

    def crash(self):
        error_message = tb.format_exc()
        print(error_message)
        self.performance_stats.crash_stage(self.stage)
        self.performance_stats.time_of_crash(time.perf_counter() - self.start_time)
        self.performance_stats.ram_usage_of_crash(res.getrusage(res.RUSAGE_SELF).ru_maxrss / 1e6)
        self.performance_stats.gpu_usage_of_crash(pt.cuda.max_memory_allocated() / 1e9)
        fail_message = '\n\n'.join((
            error_message,
            display_row(self.experiment_hyperparams),
            display_row(self.performance_stats)
        ))
        if self.notifications:
            ez.email('jamesfinch293@gmail.com', f"{self.experiment} Crash", fail_message)

@settings
class LlamaExperimentRun(ExperimentRun, LlamaTrackerExperiment): pass
@settings
class T5ExperimentRun(ExperimentRun, T5TrackerExperiment): pass


def main():
    experiment = sys.argv[1].strip()
    hyperparameters = ez.File(f'slurm/{experiment}.json').load()
    if hyperparameters['approach'] == 'LlamaTracker':
        LlamaExperimentRun(**hyperparameters)
    else:
        T5ExperimentRun(**hyperparameters)



if __name__ == '__main__':
    from language_model.llama import llama3format
    if len(sys.argv) > 1:
        main()
    else:
        LlamaExperimentRun(
    # --- Model ---
    base="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_reponame="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    format=llama3format,

    # --- Identification ---
    approach="LlamaTracker",
    groupcode="smoke",
    experiment="train_smoke_d0t_1epoch_tiny",

    # --- Data ---
    train_data="data/dsg5k/train",
    eval_data="data/mwoz2.4/valid",
    train_downsample=2000,
    eval_dialogue_downsample=5,      # small for 8GB GPU
    test_domains=None,

    # --- Training (disabled for smoke test) ---
    epochs=1,
    train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=0,
    optimizer="adamw",
    learning_rate=1e-4,
    weight_decay=0.0,

    # --- Sequence lengths (reduced for VRAM safety) ---
    max_sequence_length=128,
    protected_input_length=96,
    max_output_length=8,

    # --- Generation ---
    gen_batch_size=1,
    num_beams=1,
    repetition_penalty=1.0,

    # --- Slot behaviour ---
    train_all_slots_per_domain=True,
    eval_all_slots_per_domain=True,
    exclude_speakers=["bot"],
    eval_exclude_speakers="bot",

    # --- Prompt config ---
    train_prop_add_continuation=1.0,
    train_percent_with_description=1.0,
    train_percent_description_only=0.0,
    train_percent_with_categories=0.0,
    train_percent_with_value_exs=0.0,
    train_percent_value_ex_includes_actual_value=None,
    train_remove_request_token_percent=None,
    train_filters_out_descriptions_with_actual_value=None,

    # --- Evaluation flags ---
    eval_with_categories=False,
    eval_with_value_exs=False,
    prediction_lowercase=True,
    prediction_fuzzy_match_candidates=True,
    uncased=False,

    # --- Quantization / LoRA (disabled in minimal shim) ---
    quantize=None,
    lora=None,
    lora_alpha=None,
    lora_dropout=0.0,

    # --- Misc ---
    rng_seed=21,
    do_eval_after_all_training=True,
    calculate_eval_perplexity=False,
    yield_every_x_epochs=1,
    dynamic_tokenization=True,
    notifications=False,
)
