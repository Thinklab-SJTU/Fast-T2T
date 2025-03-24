# Training-to-Testing Solving Framework for Combinatorial Optimization

Official implementation of **NeurIPS 2024** paper "[Fast T2T: Optimization Consistency Speeds Up Diffusion-Based Training-to-Testing Solving for Combinatorial Optimization](https://openreview.net/pdf?id=xDrKZOZEOc)" and **NeurIPS 2023** paper "[T2T: From Distribution Learning in Training to Gradient Search in Testing for Combinatorial Optimization](https://openreview.net/forum?id=JtF0ugNMv2)".

![fig1](figs/fig1.png)

<img src="figs/fig2.png" alt="fig2" style="zoom: 50%;" />

To bridge the gap between training on historical instances and solving new instances, the training-to-testing (T2T) framework first leverages the generative modeling to estimate the high-quality solution distribution for each instance during training, and then conducts a gradient-based search within the solution space during testing. Initially, T2T uses a diffusion model for this process, based on $p\_\theta(\mathbf{x}\_{t-1}|\mathbf{x}\_t,G)$. However, the iterative sampling process, which requires denoising across multiple noise levels, incurs significant computational overhead. To improve both effectiveness and efficiency, Fast T2T is introduced to learn direct mappings from different noise levels to the optimal solution for a given instance, i.e., $f\_\theta(\mathbf{x}\_{0}|\mathbf{x}\_t,t,G)$, facilitating high-quality generation with minimal shots. This is achieved through an optimization consistency training protocol, which, for a given instance, minimizes the difference among samples originating from varying generative trajectories and time steps relative to the optimal solution.  The neural search paradigms consistently disrupts the local structure of the given solution by introducing noise and reconstructs a lower-cost solution guided by the optimization objective, yet are independently developed based on denoising functions and consistency functions, respectively.

## Setup

The implementation is primarily developed on top of [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO)'s code foundation.

```bash
conda env create -f environment.yml
conda activate fastt2t
```

Running TSP experiments requires installing the additional cython package for merging the diffusion heatmap results:

```bash
cd diffusion/utils/cython_merge
python setup.py build_ext --inplace
cd -
```

## Data and Pretrained Checkpoints

We provide the evaluation data for TSP in `data/tsp`. For other details, please refer to `data` folder. Please download the pretrained model checkpoints of Fast T2T from [here](https://drive.google.com/drive/folders/107mbgDsRqp-0Vf16_xDRGVFeDzeRNr-P?usp=sharing). TSP-50 and TSP-100 models are trained with parameter initialization from [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO).

## Reproduction

The training and evaluation scripts for Fast T2T can be directly found in `scripts` folder.

For instance, a simple evaluation of Fast T2T on TSP50 data:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
  --task "tsp" \
  --project_name "consistency_co_test" \
  --wandb_logger_name "tsp_50" \
  --do_test \
  --storage_path "./" \
  --test_split "your_data_path" \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 1 \
  --two_opt_iterations 5000 \
  --ckpt_path 'tsp50.ckpt' \
  --consistency \
  --resume_weight_only \
  --parallel_sampling 1 \
  --sequential_sampling 1 \
  --rewrite \
  --guided \
  --rewrite_steps 1 \
  --rewrite_ratio 0.2
```

The training and evaluation scripts for T2T can be found in [T2T](https://github.com/Thinklab-SJTU/T2TCO).

For instance, a simple evaluation of Fast T2T on TSP50 data:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u diffusion/train.py \
  --task "tsp" \
  --project_name "consistency_co_test" \
  --wandb_logger_name "tsp_50" \
  --do_test \
  --storage_path "./" \
  --test_split "your_data_path" \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50 \
  --two_opt_iterations 5000 \
  --ckpt_path "ckpts/tsp50_categorical.ckpt" \
  --resume_weight_only \
  --parallel_sampling 1 \
  --sequential_sampling 1 \
  --rewrite \
  --norm \
  --rewrite_steps 3 \
  --rewrite_ratio 0.4
```

## Reference

If you found this codebase useful, please consider citing the Fast T2T paper and the T2T paper:

```
@inproceedings{li2024fastt2t,
  title={Fast t2t: Optimization consistency speeds up diffusion-based training-to-testing solving for combinatorial optimization},
  author={Li, Yang and Guo, Jinpei and Wang, Runzhong and Zha, Hongyuan and Yan, Junchi},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}

@inproceedings{li2023t2t,
  title = {T2T: From Distribution Learning in Training to Gradient Search in Testing for Combinatorial Optimization},
  author = {Li, Yang and Guo, Jinpei and Wang, Runzhong and Yan, Junchi},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023}
}
```

