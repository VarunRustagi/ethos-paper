from functools import partial
from multiprocessing import set_start_method
import traceback
from typing import Optional

import numpy as np
from ethos.datasets.admission_mortality import AdmissionMortalityDataset
from ethos.datasets.mimic import DrgPredictionDataset, ICUMortalityDataset, ICUReadmissionDataset, SofaPredictionDataset
from ethos.datasets.mortality import MortalityDataset, SingleAdmissionMortalityDataset
from ethos.datasets.readmission import ReadmissionDataset
from ethos.inference.constants import Test
from ethos.tokenize.special_tokens import SpecialToken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from ethos.inference import run_inference, profile_inference
from ethos.utils import load_data, load_model_from_checkpoint, get_logger
from ethos.tokenize import Vocabulary
from ethos.constants import ADMISSION_STOKEN, DISCHARGE_STOKEN, ICU_ADMISSION_STOKEN, ICU_DISCHARGE_STOKEN, PROJECT_DATA
import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Subset

app = FastAPI()
logger = get_logger()

# Data model for incoming requests
class InferenceRequest(BaseModel):
    test: str
    model: str
    data: str
    vocab: str
    n_tokens: int = None
    n_jobs: int = 1
    n_gpus: int = 1
    suffix: str = None
    device: str = "cuda"
    no_compile: bool = False
    output: str = "results"
    no_time_offset: bool = False
    mode: str = "infer"

@app.post("/inference/")
async def inference(request: InferenceRequest):
    vocab = Vocabulary(PROJECT_DATA / "tokenized_datasets/mimic_vocab_t4367.pkl")  # Modify path as needed
    model_path = Path(request.model)
    device = "cpu"
    model, block_size = load_model_from_checkpoint(model_path, "cpu", for_training=False)
    model_name = model_path.stem
    test = Test(request.test)
    try:
        # Prepare data
        data_path = PROJECT_DATA / request.data
        data = {
            "model": model,
            "device": request.device,
            "vocab": vocab,
            "results_dir": Path(request.output),
            "test": request.test,
            "suffix": request.suffix,
            "no_compile": request.no_compile,
        }

        stoi = [SpecialToken.DEATH, SpecialToken.TIMELINE_END]
        if test == Test.READMISSION:
            dataset_cls = ReadmissionDataset
            stoi = [ADMISSION_STOKEN] + stoi
        elif test == Test.ADMISSION_MORTALITY:
            dataset_cls = AdmissionMortalityDataset
            stoi = [DISCHARGE_STOKEN] + stoi
        elif test == Test.MORTALITY:
            dataset_cls = MortalityDataset
        elif test == Test.SINGLE_ADMISSION:
            # todo: move the hardcoded values to options of the script
            dataset_cls = partial(SingleAdmissionMortalityDataset, admission_idx=1000, num_reps=35)
            stoi = [DISCHARGE_STOKEN] + stoi
        elif test == Test.SOFA_PREDICTION:
            dataset_cls = SofaPredictionDataset
            stoi += SpecialToken.DECILES
        elif test == Test.ICU_MORTALITY:
            dataset_cls = partial(ICUMortalityDataset, use_time_offset=not request.no_time_offset)
            stoi = [ICU_DISCHARGE_STOKEN] + stoi
        elif test == Test.DRG_PREDICTION:
            dataset_cls = DrgPredictionDataset
            drg_stokens = list(vocab.get_q_storage("DRG_CODE").values())
            assert drg_stokens, "No DRG stokens found in the vocabulary"
            stoi += drg_stokens
        elif test == Test.ICU_READMISSION:
            dataset_cls = ICUReadmissionDataset
            stoi = [ICU_ADMISSION_STOKEN, DISCHARGE_STOKEN] + stoi
        else:
            raise ValueError(f"Unknown test: {test}, available")
        
        logger.info("Inference API called successfully")
        logger.info(f"Model loaded (block_size={block_size})")

        data_path = PROJECT_DATA / request.data
        fold = data_path.stem.split("_")[1]
        logger.info(f"Fold: {fold}")
        token_suffix = f"_{request.n_tokens}M_tokens" if request.n_tokens is not None else ""
        logger.info(f"Token suffix: {token_suffix}")
        results_dir = Path(request.output) / f"{test.value}_{model_name}_{fold}{token_suffix}"
        logger.info(f"Results directory: {results_dir}")
        results_dir.mkdir(parents=True, exist_ok=True)
        data = load_data(data_path, n_tokens=request.n_tokens * 1_000_000 if request.n_tokens is not None else None)
        data["times"].share_memory_()
        data["tokens"].share_memory_()
        data["patient_context"].share_memory_()

        dataset = dataset_cls(data=data, encode=vocab.encode, block_size=block_size)
        logger.info(f"Dataset size: {len(dataset):,}")


        data = model, device, vocab, stoi, results_dir, test, request.suffix, request.no_compile
        indices = np.arange(len(dataset))
        subsets = (
            Subset(dataset, subset_indices) for subset_indices in np.array_split(indices, request.n_jobs)
        )
        loaders = [
            DataLoader(
                subset,
                batch_size=None,
                pin_memory=request.device == "cuda",
                batch_sampler=None,
                pin_memory_device=f"{device}:{i % request.n_gpus}" if device == "cuda" else "",
            )
            for i, subset in enumerate(subsets)
        ]
        if(request.mode == "infer"):
            result = Parallel(n_jobs=request.n_jobs)(delayed(run_inference)(loader, data, request.n_gpus) for loader in loaders)
        else:
            result = Parallel(n_jobs=request.n_jobs)(delayed(profile_inference)(loader, data, request.n_gpus) for loader in loaders)
        # Example of combining results:
        total_tokens = sum(r['total_tokens'] for r in result)
        avg_latency = sum(r['average_latency'] for r in result) / len(result)
        return {"status": "success", "total_tokens": total_tokens, "average_latency": avg_latency, "results":result,}
    except Exception as e:
        error_details = traceback.format_exc()  # Captures the full traceback
        logger.error(f"Error during inference:\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "ETHOS Inference API is running"}
