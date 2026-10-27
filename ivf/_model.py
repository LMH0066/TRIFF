import io
import itertools
import os
from shutil import rmtree
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import rich
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from anndata import AnnData
from captum.attr import IntegratedGradients
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalObsField,
    LayerField,
    NumericalObsField,
    ObsmField,
)
from scvi.dataloaders import DataSplitter
from scvi.model.base import BaseModelClass
from scvi.train import SaveCheckpoint, TrainRunner
from scvi.utils import setup_anndata_dsp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ivf._data import AnnDataSplitter
from ivf._module import NET
from ivf._train import IVFTrainingPlan
from ivf._utils import _logger


def _get_field(adata: AnnData, layer: str = None, obsm_column: str = None):
    if layer:
        if layer == "X":
            _logger.info("Using data from `adata.X`.")
            FIELD = LayerField(registry_key=layer, layer=None)
        else:
            if layer not in adata.layers:
                raise KeyError(
                    f"{layer} is not a valid key in `adata.layers`.")
            _logger.info(f"Using data from adata.layers[{layer!r}]")
            FIELD = LayerField(
                registry_key=layer,
                layer=layer,
            )
    elif obsm_column:
        if obsm_column not in adata.obsm.keys():
            raise KeyError(
                f"{obsm_column} is not a valid key in `adata.obsm`.")
        _logger.info(f"Using data from adata.obsm[{obsm_column!r}]")
        if adata.obsm[obsm_column].dtype in [int, float]:
            FIELD = ObsmField(registry_key=obsm_column, attr_key=obsm_column)
        else:
            raise KeyError(
                f"Type {adata.obsm[obsm_column].dtype} is not supported.")
    return FIELD


def _ddp_inference(self, rank, world_size, queue, model, adata, indices, batch_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    model.eval()

    adata = self._validate_anndata(adata)
    adata_manager = self.get_anndata_manager(adata)
    indices = np.arange(
        adata_manager.adata.shape[0]) if indices is None else indices
    dataset = adata_manager.create_torch_dataset(indices=indices)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    data_loader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, shuffle=False
    )

    results = []
    for tensors in data_loader:
        index = tensors["ind_x"]
        tensors["X"] = tensors["X"].to(rank)
        with torch.inference_mode():
            output: torch.Tensor = self.module.forward(
                tensors, compute_loss=False)
            output = output.detach().cpu()
            output = output.unsqueeze(0) if output.dim() != 2 else output
        results.extend(list(zip(index.tolist(), output.tolist())))
    queue.put(results)

    dist.destroy_process_group()


class ModifiedSaveCheckpoint(SaveCheckpoint):
    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        """Removes model saves that are no longer needed."""
        super(SaveCheckpoint, self)._remove_checkpoint(trainer, filepath)

        model_path = filepath.split(".ckpt")[0]
        if (
            trainer.is_global_zero
            and os.path.exists(model_path)
            and os.path.isdir(model_path)
        ):
            rmtree(model_path)

    def _update_best_and_save(
        self,
        current: torch.Tensor,
        trainer: pl.Trainer,
        monitor_candidates: dict[str, torch.Tensor],
    ) -> None:
        """Replaces Lightning checkpoints with our model saves."""
        super(SaveCheckpoint, self)._update_best_and_save(
            current, trainer, monitor_candidates
        )

        if trainer.is_global_zero:
            if os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)
            self.best_model_path = self.best_model_path.split(".ckpt")[0]


class IVF(BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        model_name: Optional[str] = None,
        module_params: Dict[str, Any] = None,
        split_key: Optional[str] = None,
        train_split: str = "train",
        valid_split: str = "validate",
        test_split: str = "test",
    ):
        super().__init__(adata)

        torch.set_float32_matmul_precision("high")
        self._set_attributes_maps()

        self.split_key = split_key
        self.scores = {}

        self._module = None
        self._training_plan = None
        self._data_splitter = None

        train_indices, valid_indices, test_indices = None, None, None
        if split_key is not None:
            train_indices = np.where(
                adata.obs.loc[:, split_key] == train_split)[0]
            valid_indices = np.where(
                adata.obs.loc[:, split_key] == valid_split)[0]
            test_indices = np.where(
                adata.obs.loc[:, split_key] == test_split)[0]

        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.test_indices = test_indices
        self.n_samples = adata.n_obs

        module_params = module_params if isinstance(
            module_params, Dict) else {}

        self._model_summary_string = self.__class__.__name__
        self._model_name = model_name

        self.module = NET(
            main_loc=self.main_loc,
            target_loc=self.target_loc,
            **module_params,
        ).float()

        self.init_params_ = self._get_init_params(locals())
        self.epoch_history = None

    def _set_attributes_maps(self):
        """Set attributes' maps."""
        self.main_loc = self.registry_["setup_args"]["main_layer"]
        self.target_loc = self.registry_["setup_args"]["target_column"]

    @property
    def training_plan(self):
        """The model's training plan."""
        return self._training_plan

    @training_plan.setter
    def training_plan(self, plan):
        self._training_plan = plan

    @property
    def data_splitter(self):
        """Data splitter."""
        return self._data_splitter

    @data_splitter.setter
    def data_splitter(self, data_splitter):
        self._data_splitter = data_splitter

    @property
    def module(self):
        """Model's module."""
        return self._module

    @module.setter
    def module(self, module):
        self._module = module

    @property
    def model_name(self) -> str:
        """Model's name."""
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str):
        self._model_name = model_name

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        main_layer: str = "X",
        target_column: str = "target",
        **kwargs: Any,
    ) -> None:
        """Setup function.

        Parameters
        ----------
        adata
            Annotated data object.
        layer
            Expression layer in :attr:`anndata.AnnData.layers` to use. If :obj:`None`, use :attr:`anndata.AnnData.X`.
        kwargs
            Keyword arguments for :meth:`~scvi.data.AnnDataManager.register_fields`.

        Returns
        -------
        Nothing, just sets up ``adata``.
        """

        MAIN_FIELD = _get_field(adata, layer=main_layer)
        TARGET_FIELD = _get_field(adata, obsm_column=target_column)

        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs)
        anndata_fields = [
            MAIN_FIELD,
            TARGET_FIELD,
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]

        adata_manager = AnnDataManager(
            fields=anndata_fields,
            setup_method_args=setup_method_args,
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_dataset(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Processes :class:`~anndata.AnnData` object into valid input tensors for the model.

        Parameters
        ----------
        adata
            Annotated data object.
        indices
            Optional indices.

        Returns
        -------
        A dictionary of tensors which can be passed as input to the model.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=len(indices), shuffle=False
        )
        return list(scdl)[0]

    @torch.no_grad()
    def predict(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = 4,
    ) -> AnnData:
        """The model's prediction for a given :class:`~anndata.AnnData` object.

        Parameters
        ----------
        adata
            Annotated data object.
        indices
            Optional indices.
        batch_size
            Batch size to use.

        Returns
        -------
        Two :class:`~anndata.AnnData` objects representing the model's prediction of the expression mean and variance respectively.
        """
        self.module.eval()

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, shuffle=False
        )
        outputs = None
        for tensors in scdl:
            output: torch.Tensor = self.module.forward(
                tensors,
                compute_loss=False,
            )
            output = output.detach().cpu()
            output = (
                output.unsqueeze(0) if output.dim() != 2 else output
            )  # [batch_size, score]
            output = output.numpy()
            outputs = output if outputs is None else np.concatenate(
                (outputs, output))

        _adata = adata[indices] if indices is not None else adata
        pred_adata = AnnData(X=outputs, obsm=_adata.obsm.copy())
        pred_adata.obs_names = _adata.obs_names

        return pred_adata

    def parallel_predict(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = 4,
        world_size: int = torch.cuda.device_count(),
        MASTER_ADDR: str = "localhost",
        MASTER_PORT: str = "12355",
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["MASTER_ADDR"] = MASTER_ADDR
        os.environ["MASTER_PORT"] = MASTER_PORT

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        for i in range(world_size):
            p = ctx.Process(
                target=_ddp_inference,
                args=(
                    self,
                    i,
                    world_size,
                    queue,
                    self.module,
                    adata,
                    indices,
                    batch_size,
                ),
            )
            p.start()
        results = [queue.get() for _ in range(world_size)]
        results = list(itertools.chain(*results))

        results = pd.DataFrame(results, columns=["index", "output"])
        results["index"] = results["index"].apply(lambda x: x[0])
        results = results.drop_duplicates("index").sort_values(by="index")
        outputs = np.array(results["output"].tolist())

        _adata = adata[indices] if indices is not None else adata
        pred_adata = AnnData(X=outputs, obsm=_adata.obsm.copy())
        pred_adata.obs_names = _adata.obs_names

        return pred_adata

    @torch.no_grad()
    def interpret(
        self,
        data: np.ndarray,
        internal_batch_size: int = 8,
    ):
        self.module.eval()
        ig = IntegratedGradients(self.module.inference)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        tensors = {self.main_loc: torch.Tensor(data)}
        inference_inputs = self.module._get_inference_input(tensors)
        attributions = ig.attribute(
            inference_inputs["main"],
            target=(0),
            internal_batch_size=internal_batch_size,
        )
        return attributions

    def save(
        self,
        dir_path: Optional[str] = None,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_save_kwargs: Any,
    ) -> None:
        """Save the model.

        Parameters
        ----------
        dir_path
            Directory where to save the model. If :obj:`None`, it will be determined automatically.
        overwrite
            Whether to overwrite an existing model.
        save_anndata
            Whether to also save :class:`~anndata.AnnData`.
        anndata_save_kwargs
            Keyword arguments :meth:`scvi.model.base.BaseModelClass.save`.

        Returns
        -------
        Nothing, just saves the model.
        """
        if dir_path is None:
            dir_path = (
                f"./{self.__class__.__name__}_model/"
                if self.model_name is None
                else f"./{self.__class__.__name__}_{self.model_name}_model/"
            )
        super().save(
            dir_path=dir_path,
            overwrite=overwrite,
            save_anndata=save_anndata,
            **anndata_save_kwargs,
        )

        if isinstance(self.training_plan.epoch_history, dict):
            self.epoch_history = pd.DataFrame().from_dict(
                self.training_plan.epoch_history
            )
            self.epoch_history.to_csv(
                os.path.join(dir_path, "history.csv"), index=False
            )

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
        accelerator: str = "auto",
        device: Union[int, list[int], str] = "auto",
        **kwargs: Any,
    ) -> "IVF":
        """Load a saved model.

        Parameters
        ----------
        dir_path
            Directory where the model is saved.
        adata
            AnnData organized in the same way as data used to train model.
        accelerator
            Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu",
            "mps, "auto") as well as custom accelerator instances.
        device
            The device to use. Can be set to a positive number (int or str), or ``"auto"``
            for automatic selection based on the chosen accelerator.
        kwargs
            Keyword arguments for :meth:`scvi`

        Returns
        -------
        The saved model.
        """
        model = super().load(
            dir_path, adata, accelerator=accelerator, device=device, **kwargs
        )

        fname = os.path.join(dir_path, "history.csv")
        if os.path.isfile(fname):
            model.epoch_history = pd.read_csv(fname)
        else:
            _logger.warning(f"The history file `{fname}` was not found")

        return model

    def train(
        self,
        max_epochs: Optional[int] = None,
        accelerator: str = "auto",
        device: Union[int, list[int], str] = "auto",
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        plan_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 4,
        early_stopping: bool = False,
        **trainer_kwargs: Any,
    ) -> None:
        """Train the :class:`~ivf.IVF` model.

        Parameters
        ----------
        max_epochs
            Maximum number of epochs for training.
        accelerator
            Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu",
            "mps, "auto") as well as custom accelerator instances.
        device
            The device to use. Can be set to a positive number (int or str), or ``"auto"``
            for automatic selection based on the chosen accelerator.
        train_size
            Fraction of training data in the case of randomly splitting dataset to train/validation
            if :attr:`split_key` is not set in model's constructor.
        validation_size
            Fraction of validation data in the case of randomly splitting dataset to train/validation
            if :attr:`split_key` is not set in model's constructor.
        batch_size
            Size of mini-batches for training.
        early_stopping
            If `True`, early stopping will be used during training on validation dataset.
        plan_kwargs
            Keyword arguments for :class:`~scvi.train.TrainingPlan`.
        trainer_kwargs
            Keyword arguments for :class:`~scvi.train.TrainRunner`.

        Returns
        -------
        Nothing, just trains the :class:`~ivf.IVF` model.
        """
        plan_kwargs = plan_kwargs if plan_kwargs is not None else {}
        # self.module = torch.compile(self.module)
        self.training_plan = IVFTrainingPlan(
            module=self.module,
            **plan_kwargs,
        )

        monitor = trainer_kwargs.pop("monitor", "val_metric")
        save_ckpt_every_n_epoch = trainer_kwargs.pop(
            "save_ckpt_every_n_epoch", 20)
        save_top_k = trainer_kwargs.pop("save_top_k", -1)
        enable_checkpointing = trainer_kwargs.pop("enable_checkpointing", True)

        trainer_kwargs["callbacks"] = (
            []
            if "callbacks" not in trainer_kwargs.keys()
            else trainer_kwargs["callbacks"]
        )
        if enable_checkpointing:
            checkpointing_callback = ModifiedSaveCheckpoint(
                monitor=monitor,
                every_n_epochs=save_ckpt_every_n_epoch,
                save_top_k=save_top_k,
            )
            trainer_kwargs["callbacks"] += [checkpointing_callback]
            trainer_kwargs["checkpointing_monitor"] = monitor

        num_workers = trainer_kwargs.pop("num_workers", 0)
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        manual_splitting = (
            (self.valid_indices is not None)
            and (self.train_indices is not None)
            and (self.test_indices is not None)
        )
        if manual_splitting:
            self.data_splitter = AnnDataSplitter(
                self.adata_manager,
                train_indices=self.train_indices,
                valid_indices=self.valid_indices,
                test_indices=self.test_indices,
                batch_size=batch_size,
                accelerator=accelerator,
                num_workers=num_workers,
            )
        else:
            self.data_splitter = DataSplitter(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys(
            ) else trainer_kwargs[es]
        )

        trainer_kwargs["check_val_every_n_epoch"] = trainer_kwargs.get(
            "check_val_every_n_epoch", 1
        )
        trainer_kwargs["early_stopping_patience"] = trainer_kwargs.get(
            "early_stopping_patience", 20
        )

        root_dir = settings.logging_dir
        root_dir = (
            os.path.join(root_dir, f"{self.__class__.__name__}/")
            if self.model_name is None
            else os.path.join(root_dir, f"{self.model_name}_{self.__class__.__name__}/")
        )

        trainer_kwargs["default_root_dir"] = trainer_kwargs.pop(
            "default_root_dir", root_dir
        )

        runner = TrainRunner(
            self,
            training_plan=self.training_plan,
            data_splitter=self.data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=device,
            early_stopping_monitor=monitor,
            early_stopping_mode="max",
            enable_checkpointing=enable_checkpointing,
            strategy=DDPStrategy(process_group_backend="gloo"),
            **trainer_kwargs,
        )

        return runner()

    def __repr__(self) -> str:
        buffer = io.StringIO()
        summary_string = f"{self._model_summary_string} training status: "
        summary_string += "{}".format(
            "[green]Trained[/]" if self.is_trained else "[red]Not trained[/]"
        )
        console = rich.console.Console(file=buffer)
        with console.capture() as capture:
            console.print(summary_string)
        return capture.get()
