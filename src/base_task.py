"""
This file provides basic classes and functions to accelerate the development of deep learning tasks.
Some APIs are designed by taking 'pytorch-lightning' and 'transformers' packages as references.
"""

import os
import sys
import json
import random
import shutil
import torch
from collections.abc import Iterable
from abc import ABC, abstractmethod
from datetime import datetime
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
import traceback as tb
import logging as lg
import numpy as np

lg.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=lg.INFO)

# keys to obtain batch output info
LOSS_KEY = 'loss'  # for training loss
BAR_KEY = 'progress_bar'  # for progress bar postfix
SCALAR_LOG_KEY = 'scalar_log'  # for tensorboard
VAL_SCORE_KEY = 'val_score'  # for choosing the best checkpoint
# directories to save checkpoints, model outputs, and tensorboard summaries
CPT_DIR_NAME = 'Checkpoint'  # for checkpoints
OUT_DIR_NAME = 'Output'  # for model outputs
LOG_DIR_NAME = 'Tensorboard'  # for tensorboard summaries
# runtime environment variables
RUNTIME_LOG_DIR = 'RUNTIME_LOG_DIR'  # logging tensorboard info into a runtime directory, and copy to exp dir at last
RUNTIME_MODEL_DIR = 'RUNTIME_MODEL_DIR'  # dumping model checkpoints into a runtime directory, and copy to exp dir at last
# TODO: impl logics for runtime model directory

# Helper functions
def dump_json_to(obj, fpath, indent=2, ensure_ascii=False, **kwargs):
    """The helper for dumping json into the given file path"""
    with open(fpath, 'w') as fout:
        json.dump(obj, fout, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def load_json_from(fpath, **kwargs):
    """The helper for loading json from the given file path"""
    with open(fpath, 'r') as fin:
        obj = json.load(fin, **kwargs)

    return obj


def strtobool(str_val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    str_val = str_val.lower()
    if str_val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif str_val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (str_val,))


def add_config_to_argparse(config, arg_parser):
    """The helper for adding configuration attributes to the argument parser"""
    for key, val in config.to_dict().items():
        if isinstance(val, bool):
            arg_parser.add_argument('--' + key, type=strtobool, default=val)
        elif isinstance(val, (int, float, str)):
            arg_parser.add_argument('--' + key, type=type(val), default=val)
        else:
            raise Exception('Do not support value ({}) type ({})'.format(val, type(val)))


class BaseConfig:
    """The config class to set up the task"""

    def __init__(self):
        # the maximum number of epochs to run
        self.max_epochs = 1000
        # the initial random seed for data sampler and parameter initialization
        self.random_seed = 0
        # the number of gradient accumulation steps (for large-batch training with limited memory)
        self.grad_accum_steps = 1
        # whether to use cuda for GPU training
        self.use_cuda = True
        # -1: single-gpu training, positive integer: the local rank for distributed training
        self.local_rank = -1
        # distributed bankend, 'gloo' is slower than 'nccl'
        self.dist_backend = 'nccl'
        # only permit the master node to log information
        self.only_master_log = True
        # the directory to save checkpotins and model outputs
        self.exp_dir = os.path.join(os.getcwd(), 'Exp')
        # the file name of the config json
        self.cfg_fname = 'config.json'
        # the file name of the model checkpoint
        self.cpt_fname = 'model.cpt'
        # the file name of the model output
        self.out_fname = 'out.cpt'
        # Options for dumping checkpoints and model outputs
        # 0: no dump, 1: latest, 2: latest & best, 3: latest & best & every epoch
        self.dump_option = 2
        # whether trying to recover from the latest checkpoints before the start of training
        self.try_recover = False
        # whether to skip training
        self.skip_train = False
        # whether to evaluate on test set during the model fitting
        self.eval_test_dur_fit = False
        # enable early stopping
        self.enable_early_stop = True
        # the maximum number of epochs with no validation improvement before early stopping
        self.early_stop_epochs = 30
        # for DistributedDataParallel
        self.find_unused_parameters = False

    def update_by_dict(self, config_dict):
        for key, val in config_dict.items():
            setattr(self, key, val)

    def to_dict(self):
        return dict(self.__dict__)


class BasePytorchTask(ABC):
    """The task class to support typical deep learning workflows based on Pytorch"""

    def __init__(self, config):
        assert isinstance(config, BaseConfig)
        self.config = config

        # init distributed backend
        if self.in_distributed_mode:
            if not dist.is_initialized():
                # only set backend here, other settings either adopt defaults or inherit from env variables
                dist.init_process_group(self.config.dist_backend)
            # one process occupies one GPU exclusively
            torch.cuda.set_device(self.config.local_rank)
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.log('Distributed Mode [Global Rank {}, Local Rank {}, World Size {}]'.format(
                self.global_rank, self.config.local_rank, self.world_size
            ))
        else:
            self.global_rank = None
            self.world_size = None
            self.log('Non-distributed Mode')

        # set device
        if self.config.use_cuda:
            if self.config.local_rank >= 0:
                # multi-process multi-GPU (recommended)
                self.device = torch.device('cuda', self.config.local_rank)
                self.gpu_num = 1
            else:
                # single-process multi-GPU
                self.device = torch.device('cuda')
                self.gpu_num = torch.cuda.device_count()
        else:
            self.device = torch.device('cpu')
            self.gpu_num = 0
        self.log('Device {} #GPU {}'.format(self.device, self.gpu_num))

        self.init_exp_dir()
        self.set_random_seed()
        self.log('Random Seed {}'.format(self.config.random_seed))

        self.model = None
        self.optimizer = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.summary_writer = None
        self._log_dir_path = None  # the full directory path to log tensorboard summaries
        self._set_fitting_state()

        self.dump_config()

    @property
    def in_distributed_mode(self):
        """The property to identify the distributed mode"""
        return self.config.local_rank >= 0

    @property
    def is_master_node(self):
        """The property to identify the master node"""
        if self.in_distributed_mode:
            return dist.get_rank() == 0
        return True

    def log(self, message, level=lg.INFO):
        """[summary]
            Unify the command line message logging
        Arguments:
            message {[str]} -- [message to be printed]
        Keyword Arguments:
            level {[int]} -- [follow logging.INFO] (default: {lg.INFO})
        """
        if self.in_distributed_mode:
            message = 'Rank {} {}'.format(dist.get_rank(), message)

        if self.config.only_master_log:
            if self.is_master_node:
                lg.log(level, message)
        else:
            lg.log(level, message)

    def set_random_seed(self, seed=None):
        """[summary]
            Set random seeds for packages with random functions,
            use input seed when available, if not, use self.config.random_seed
        Keyword Arguments:
            seed {[int]} -- [input random seed] (default: {None})
        """
        if seed is None:
            seed = self.config.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.gpu_num > 0:
            torch.cuda.manual_seed_all(seed)

    def init_exp_dir(self):
        """[summary]
            Initialize experimental directory structures
            Exp_dir/
                CPT_DIR_NAME/
                OUT_DIR_NAME/
        """
        if self.is_master_node:
            if os.path.exists(self.config.exp_dir):
                self.log('Experimental directory {} already exists, overwrite it...'.format(
                    self.config.exp_dir))
            else:
                os.makedirs(self.config.exp_dir, exist_ok=True)
                self.log('Create experimental directory {}'.format(self.config.exp_dir))

            # build sub-directory structures
            for dir_name in [CPT_DIR_NAME, OUT_DIR_NAME, LOG_DIR_NAME]:
                dir_path = os.path.join(self.config.exp_dir, dir_name)
                os.makedirs(dir_path, exist_ok=True)
                self.log('Creat sub-directory {}'.format(dir_path))

    def has_parallel_wrapper(self, model):
        """[summary]
            Judge whether the model has been wrapped for parallel training
        Arguments:
            model {[torch.nn.Module]} -- [the input model]
        Returns:
            [bool] -- []
        """
        return isinstance(model, (DataParallel, DistributedDataParallel))

    def decorate_model(self, model):
        """[summary]
            Move the model to a proper device and add it with wrappers for parallel training
        Arguments:
            model {[torch.nn.Module]} -- [the input model]
        Returns:
            [torch.nn.Module] -- [decorated model]
        """
        if self.has_parallel_wrapper(model):
            model = model.module
        model.to(self.device)
        if self.in_distributed_mode:
            model = DistributedDataParallel(
                model, device_ids=[self.config.local_rank], output_device=self.config.local_rank,
                find_unused_parameters=self.config.find_unused_parameters
            )
        elif self.gpu_num > 1:
            model = DataParallel(model)

        return model

    def decorate_batch(self, batch):
        """[summary]
            Decorate the input batch with a proper device
        Arguments:
            batch {[torch.Tensor | list | dict]} -- [the input batch,
            where the list or dict can contain non-tensor objects]
        Raises:
            Exception: [Unsupported data type]
        Returns:
            [torch.Tensor | list | dict] -- [maintain the same structure as the input batch,
            but with tensors moved to a proper device]
        """
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
            return batch
        elif isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
                elif isinstance(value, dict) or isinstance(value, list):
                    batch[key] = self.decorate_batch(value)
                # retain other value types in the batch dict
            return batch
        elif isinstance(batch, list):
            new_batch = []
            for value in batch:
                if isinstance(value, torch.Tensor):
                    new_batch.append(value.to(self.device))
                elif isinstance(value, dict) or isinstance(value, list):
                    new_batch.append(self.decorate_batch(value))
                else:
                    # retain other value types in the batch list
                    new_batch.append(value)
            return new_batch
        else:
            raise Exception('Unsupported batch type {}'.format(type(batch)))

    def dump_output(self, obj, out_fname=None):
        if self.is_master_node and obj is not None:
            if out_fname is None:
                out_fname = self.config.out_fname
            out_fpath = os.path.join(self.config.exp_dir, OUT_DIR_NAME, out_fname)
            torch.save(obj, out_fpath)
            return out_fpath
        else:
            return None

    def dump_config(self, cfg_fname=None):
        if self.is_master_node:
            if cfg_fname is None:
                cfg_fname = self.config.cfg_fname
            cfg_fpath = os.path.join(self.config.exp_dir, cfg_fname)
            dump_json_to(self.config.to_dict(), cfg_fpath)
            return cfg_fname
        else:
            return None

    def dump_checkpoint(self, cpt_fname=None, epoch=None):
        if self.is_master_node:
            if cpt_fname is None:
                cpt_fname = self.config.cpt_fname
            cpt_fpath = os.path.join(self.config.exp_dir, CPT_DIR_NAME, cpt_fname)
            self.dump_checkpoint_to(cpt_fpath, epoch=epoch)
            return cpt_fpath
        else:
            return None

    def dump_checkpoint_to(self, cpt_fpath, epoch=None):
        # only master node can dump checkpoint
        if self.is_master_node:
            dump_dict = {
                'config': self.config.to_dict(),
            }
            if self.model is not None:
                if self.has_parallel_wrapper(self.model):
                    model_state = self.model.module.state_dict()
                else:
                    model_state = self.model.state_dict()
                dump_dict['model'] = model_state
            if isinstance(self.optimizer, Iterable):
                dump_dict['optimizer'] = [opt.state_dict() for opt in self.optimizer]
            elif self.optimizer is not None:
                dump_dict['optimizer'] = self.optimizer.state_dict()
            if epoch is not None:
                dump_dict['epoch'] = epoch
            torch.save(dump_dict, cpt_fpath)

    def resume_checkpoint(self, cpt_fname=None, resume_model=True, resume_opt=True, resume_epoch=True, strict=True):
        if cpt_fname is None:
            cpt_fname = self.config.cpt_fname
        cpt_fpath = os.path.join(self.config.exp_dir, CPT_DIR_NAME, cpt_fname)
        return self.resume_checkpoint_from(
            cpt_fpath, resume_model=resume_model, resume_opt=resume_opt, resume_epoch=resume_epoch, strict=strict
        )

    def resume_checkpoint_from(self, cpt_fpath, resume_model=True, resume_opt=True, resume_epoch=True, strict=True):
        store_dict = torch.load(cpt_fpath, map_location=self.device)
        if resume_model:
            if 'model' in store_dict:
                if self.has_parallel_wrapper(self.model):
                    self.model.module.load_state_dict(store_dict['model'])
                else:
                    self.model.load_state_dict(store_dict['model'])
            elif strict:
                raise Exception('The checkpoint did not contain model')
        if resume_opt:
            if 'optimizer' in store_dict:
                if isinstance(self.optimizer, Iterable):
                    for self_opt, dump_opt in zip(self.optimizer, store_dict['optimizer']):
                        self_opt.load_state_dict(dump_opt)
                else:
                    self.optimizer.load_state_dict(store_dict['optimizer'])
            elif strict:
                raise Exception('The checkpoint did not contain optimizer')
        if resume_epoch:
            if 'epoch' in store_dict:
                self._passed_epoch = store_dict['epoch']
            elif strict:
                raise Exception('The checkpoint did not contain epoch')
        return store_dict

    def resume_best_checkpoint(self):
        best_cpt_fname = '{}.best'.format(self.config.cpt_fname)
        return self.resume_checkpoint(cpt_fname=best_cpt_fname)

    def build_summary_writer(self):
        # only master node can build summary writer
        if self.is_master_node:
            # cur_time = datetime.now().strftime('%b%d_%H-%M-%S')
            # dir_name = '{}-{}'.format(LOG_DIR_PREFIX, cur_time)
            # sum_dpath = os.path.join(self.config.exp_dir, dir_name)
            self._log_dir_path = os.path.join(self.config.exp_dir, LOG_DIR_NAME)
            if RUNTIME_LOG_DIR in os.environ:
                # write summaries into a runtime hot storage
                self.summary_writer = SummaryWriter(os.environ[RUNTIME_LOG_DIR])
            else:
                self.summary_writer = SummaryWriter(self._log_dir_path)
            return self.summary_writer.logdir
        else:
            return None

    def close_summary_writer(self):
        if self.is_master_node:
            self.summary_writer.close()
            if RUNTIME_LOG_DIR in os.environ:
                # copy logged summaries from the runtime hot storage to experiment directory
                runtime_log_dir = os.environ[RUNTIME_LOG_DIR]
                for fn in os.listdir(runtime_log_dir):
                    raw_fp = os.path.join(runtime_log_dir, fn)
                    shutil.copy(raw_fp, self._log_dir_path)
                    self.log('Copy tensorboard {} to {}'.format(raw_fp, self._log_dir_path))

    def dump(self, val_out=None, test_out=None, epoch_idx=None, is_best=False, dump_option=1):
        # only the master node can dump outputs
        if not self.is_master_node or dump_option <= 0:
            return 0

        val_ftemp = 'val.{}'
        test_ftemp = 'test.{}'

        # dump the latest checkpoints and outputs with default names
        cpt_fpath = self.dump_checkpoint(epoch=epoch_idx+1)  # the number of epochs passed
        val_fpath = self.dump_output(val_out, val_ftemp.format(self.config.out_fname))
        test_fpath = self.dump_output(test_out, test_ftemp.format(self.config.out_fname))

        if dump_option <= 1:
            return 1

        # update the best checkpoints and outputs, name: '{}.{}'.format(default_name, 'best')
        if is_best:
            for fpath in [cpt_fpath, val_fpath, test_fpath]:
                if fpath is not None:
                    new_fpath = '{}.{}'.format(fpath, 'best')
                    shutil.copyfile(fpath, new_fpath)

        if dump_option <= 2:
            return 2

        # dump per each epoch, name: '{}.{}'.format(default_name, epoch_idx+1)
        assert isinstance(epoch_idx, int)
        for fpath in [cpt_fpath, val_fpath, test_fpath]:
            if fpath is not None:
                new_fpath = '{}.{}'.format(fpath, epoch_idx+1)
                shutil.copyfile(fpath, new_fpath)

        return 3

    def dump_cumstom(self, val_out=None, test_out=None, epoch_idx=None, is_best=False, dump_option=1, name=''):
        # only the master node can dump outputs
        if not self.is_master_node or dump_option <= 0:
            return 0

        val_ftemp = 'val.{}'
        test_ftemp = 'test.{}'

        # dump the latest checkpoints and outputs with default names
        cpt_fpath = self.dump_checkpoint(epoch=epoch_idx+1)  # the number of epochs passed
        val_fpath = self.dump_output(val_out, val_ftemp.format(self.config.out_fname))
        test_fpath = self.dump_output(test_out, test_ftemp.format(self.config.out_fname))

        if dump_option <= 1:
            return 1

        # update the best checkpoints and outputs, name: '{}.{}'.format(default_name, 'best')
        if is_best:
            for fpath in [cpt_fpath, val_fpath, test_fpath]:
                if fpath is not None:
                    new_fpath = '{}.{}'.format(fpath, 'best')
                    shutil.copyfile(fpath, new_fpath)

        if dump_option <= 2:
            return 2

        # dump per each epoch, name: '{}.{}'.format(default_name, epoch_idx+1)
        assert isinstance(epoch_idx, int)
        for fpath in [cpt_fpath, val_fpath, test_fpath]:
            if fpath is not None:
                new_fpath = '{}.{}'.format(fpath, epoch_idx+1)
                shutil.copyfile(fpath, new_fpath)

        return 3

    def try_recover_before_train(self):
        cpt_fpath = os.path.join(self.config.exp_dir, CPT_DIR_NAME, self.config.cpt_fname)
        if os.path.exists(cpt_fpath):
            self.resume_checkpoint_from(
                cpt_fpath, resume_model=True, resume_opt=True, resume_epoch=True, strict=True
            )
            self.log('Recover from the latest checkpoint (epoch={})'.format(self._passed_epoch))
        else:
            self.log('No latest checkpoint, train from scratch (epoch={})'.format(self._passed_epoch))

    def process_log_dict(self, log_dict, log_step):
        # only master node can write log dict
        if self.is_master_node:
            for key, val in log_dict.items():
                self.summary_writer.add_scalar(key, val, global_step=log_step)

    def process_bar_info(self, bar_info, pbar):
        if isinstance(bar_info, dict):
            pbar.set_postfix(**bar_info)
        elif isinstance(bar_info, str):
            pbar.set_postfix_str(bar_info)
        else:
            raise Exception('Unsupported bar info type {}'.format(type(bar_info)))

    def handle_func_out_keys(self, step_out, pbar, log_step, inc_pbar=True):
        # non-master nodes can also update their own progress bars if only_master_log == False
        if self.config.only_master_log and not self.is_master_node:
            return

        if inc_pbar:
            pbar.update(n=1)

        # if BAR_KEY in step_out and isinstance(pbar, tqdm):
        if BAR_KEY in step_out:
            bar_info = step_out[BAR_KEY]
            self.process_bar_info(bar_info, pbar)

        if SCALAR_LOG_KEY in step_out and log_step is not None:
            # assume log_dict format: { key_string: scalar_variable}
            # Note that we only support adding scalar tensors currently
            # skip logging if log_step is None
            log_dict = step_out[SCALAR_LOG_KEY]
            self.process_log_dict(log_dict, log_step)

    def _set_fitting_state(self):
        self._current_epoch = -1
        self._passed_epoch = 0  # the passed epoch of the resumed checkpoint
        self._global_step = 0  # the number of gradient updating steps so far
        self._best_val_score = - float('inf')  # the best validation score so far
        self._best_val_epoch = -1  # the epoch for the best validation score

    def _judge_best_epoch(self, val_out, epoch):
        is_best_epoch = False
        is_early_stop = False
        if VAL_SCORE_KEY in val_out:
            val_score = val_out[VAL_SCORE_KEY]
            if val_score > self._best_val_score:
                self._best_val_score = val_score
                self._best_val_epoch = epoch
                is_best_epoch = True
            else:
                if epoch - self._best_val_epoch >= self.config.early_stop_epochs:
                    is_early_stop = True
        elif self.config.enable_early_stop:
            raise Exception("Enabled early-stopping, but did not set 'VAL_SCORE_KEY'.")

        return is_best_epoch, is_early_stop

    def init_model_and_optimizer(self, model):
        self.model = self.decorate_model(model)
        self.optimizer = self.build_optimizer(self.model)

    def fit(self, model=None):
        """[summary]
            Fit the given model
        Arguments:
            model {[torch.nn.Module]} -- []
        """
        # preparations before training

        self.set_random_seed()  # reset the random seed for each call
        if model is not None:
            self.init_model_and_optimizer(model)
        assert self.model is not None and self.optimizer is not None

        self.model.zero_grad()
        self.model.train()
        self.build_summary_writer()  # build summary writer for tensorboard
        self.train_dataloader = self.build_train_dataloader()
        self._set_fitting_state()

        # user-defined preparations
        self.on_train_start()

        # useful for preemption-style cluster scheduling
        # try to resume the latest checkpoint,
        # including model, optimizer, and the number of passed epochs
        if self.config.try_recover:
            self.try_recover_before_train()

        main_pbar = tqdm(
            leave=True, dynamic_ncols=True, file=sys.stdout,
            disable=self.config.only_master_log and not self.is_master_node
        )

        # enter the training epoch
        if self.config.use_cuda:
            torch.cuda.empty_cache()

        # for epoch in trange(self._passed_epoch, self.config.max_epochs, desc='Epoch'):
        for epoch in range(self._passed_epoch, self.config.max_epochs):
            self._current_epoch = epoch
            bar_desc = "Epoch {}".format(epoch+1)
            if self.in_distributed_mode:
                bar_desc = "{} (Rank {})".format(bar_desc, self.global_rank)
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    # set the same epoch seed for all dataloaders
                    # add random seed to enable different kinds of distributed data loading
                    self.train_dataloader.sampler.set_epoch(self.config.random_seed + epoch)
                if hasattr(self.train_dataloader.dataset, 'set_epoch'):
                    # For iterable dataset with user-defined iteration logics
                    self.train_dataloader.dataset.set_epoch(self.config.random_seed + epoch)

            if not main_pbar.disable:
                main_pbar.reset(total=len(self.train_dataloader))
            main_pbar.set_description(desc=bar_desc)

            # enter the training batch
            for batch_idx, batch in enumerate(self.train_dataloader):
                batch = self.decorate_batch(batch)
                # user-defined train step function
                train_out = self.train_step(batch, batch_idx)

                loss = train_out[LOSS_KEY]  # train_out must include the 'loss' key
                if self.gpu_num > 1:
                    loss = loss.mean()  # average for the dp mode
                if self.config.grad_accum_steps > 1:
                    loss = loss / self.config.grad_accum_steps
                loss.backward()  # will trigger allreduce in the ddp mode

                if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                    # user-defined preparations before optimizer step
                    self.before_optimizer_step(self.optimizer)
                    # self.optimizer.step()
                    self.optimizer_step()
                    self.model.zero_grad()
                    self._global_step += 1

                log_step = self._global_step * self.config.grad_accum_steps + \
                    (batch_idx + 1) % self.config.grad_accum_steps
                self.handle_func_out_keys(train_out, main_pbar, log_step, inc_pbar=True)

            # evaluate the validation set
            val_out = self.val_eval(epoch=epoch+1)
            is_best_epoch, is_early_stop = self._judge_best_epoch(val_out, epoch)

            # evaluate the test set when necessary
            if self.config.eval_test_dur_fit:
                test_out = self.test_eval(epoch=epoch+1)
            else:
                test_out = None

            # dump checkpoints and model outputs
            self.dump(
                val_out=val_out,
                test_out=test_out,
                epoch_idx=epoch,
                is_best=is_best_epoch,
                dump_option=self.config.dump_option
            )

            if self.config.enable_early_stop and is_early_stop:
                msg = 'Meet early-stopping requirements, exit at epoch {}'.format(epoch+1)
                self.log(msg)
                return
            if self.in_distributed_mode:
                dist.barrier()

        # processing at the end
        main_pbar.close()
        self.close_summary_writer()

    def eval(self, model, dataloader, step_func, epoch_end_func, pbar_desc, epoch=None):
        # At present, we do not support distributed dataloader for evaluation,
        # which means that even in the distributed mode,
        # every process will run over the whole dataset
        if isinstance(dataloader.sampler, DistributedSampler):
            raise Exception('At present, we do not support distributed dataloader for evaluation.')
        # enter evaluation mode
        model.zero_grad()
        model.eval()

        if epoch is not None:
            pbar_desc = '{} Epoch {}'.format(pbar_desc, epoch)
        if self.in_distributed_mode:
            pbar_desc = '{} (Rank {})'.format(pbar_desc, self.global_rank)

        eval_pbar = tqdm(
            leave=True, dynamic_ncols=True, file=sys.stdout,
            disable=self.config.only_master_log and not self.is_master_node
        )
        eval_pbar.set_description(desc=pbar_desc)
        if not eval_pbar.disable:
            eval_pbar.reset(total=len(dataloader))

        eval_outs = []
        for batch_idx, batch in enumerate(dataloader):
            batch = self.decorate_batch(batch)

            with torch.no_grad():
                # user-defined function for each evaluation step
                eval_out = step_func(batch, batch_idx)
            eval_outs.append(eval_out)

            # set log_step = None to avoid writing tensorboard events
            self.handle_func_out_keys(eval_out, eval_pbar, None, inc_pbar=True)

        # user-defined collection function at the end of the epoch
        final_eval_out = epoch_end_func(eval_outs)

        self.handle_func_out_keys(final_eval_out, eval_pbar, epoch, inc_pbar=False)
        eval_pbar.close()

        # back to training mode
        model.train()

        return final_eval_out

    def val_eval(self, model=None, epoch=None):
        if model is None:
            model = self.model
        dataloader = self.build_val_dataloader()
        pbar_desc = 'Val'
        eval_out = self.eval(model, dataloader, self.val_step, self.val_epoch_end, 'Val', epoch=epoch)
        return eval_out

    def test_eval(self, model=None, epoch=None):
        if model is None:
            model = self.model
        dataloader = self.build_test_dataloader()
        eval_out = self.eval(model, dataloader, self.test_step, self.test_epoch_end, 'Test', epoch=epoch)
        return eval_out

    def on_train_start(self, *args, **kwargs):
        """Things to do before the training procedure starts"""
        return None

    def before_optimizer_step(self, *args, **kwargs):
        """ # pseudocode for optimizer
            loss.backward()
            optimizer_step_start(optimizer)
            optimizer.step()
        """
        return None

    def optimizer_step(self):
        if isinstance(self.optimizer, Iterable):
            for opt in self.optimizer:
                opt.step()
        else:
            self.optimizer.step()



    @abstractmethod
    def build_val_dataloader(self, *args, **kwargs):
        """[summary]
            Build a dataloader for validation
        """

    @abstractmethod
    def build_test_dataloader(self, *args, **kwargs):
        """[summary]
            Build a dataloader for testing
        """

    @abstractmethod
    def build_optimizer(self, model):
        """[summary]
            Building an optimizer for a given model,
            where the optimizer can be a group of sub-optimizers.

        Arguments:
            model {[torch.nn.Module]} -- []

        Return:
            optimizer {[torch.optim.Optimizer | List[torch.optim.Optimizer]]} -- []
        """

    @abstractmethod
    def train_step(self, *args, **kwargs):
        """ # pseudocode for validation calls
            for batch_idx, train_batch in enumerate(train_dataloader):
                out = train_step(train_batch, batch_idx)
                loss = out['loss']
                bar_info = out['progress_bar']
        """

    @abstractmethod
    def val_step(self, *args, **kwargs):
        """ # pseudocode for validation calls
            for batch_idx, val_batch in enumerate(val_dataloader):
                out = val_step(val_batch, batch_idx)
        """

    @abstractmethod
    def val_epoch_end(self, *args, **kwargs):
        """ # pseudocode for validation calls
            val_outs = []
            for batch_idx, val_batch in enumerate(val_dataloader):
                out = val_step(val_batch, batch_idx)
                val_outs.append(out)
            val_epoch_out = val_epoch_end(val_outs)
        """

    @abstractmethod
    def test_step(self, *args, **kwargs):
        """ # pseudocode for validation calls
            for batch_idx, test_batch in enumerate(test_dataloader):
                out = test_step(test_batch, batch_idx)
        """

    @abstractmethod
    def test_epoch_end(self, *args, **kwargs):
        """ # pseudocode for test calls
            test_outs = []
            for batch_idx, test_batch in enumerate(test_dataloader):
                out = test_step(test_batch, batch_idx)
                test_outs.append(out)
            test_epoch_out = test_epoch_end(test_outs)
        """